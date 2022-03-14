from functools import partial
from pathlib import Path

import hydra
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from ..points.match import match
from ..points.utils import draw_groundtruth_and_predictions
from ..points.metrics import detection_and_counting
from .metrics import counting_yx, ssim, counting
from .target_builder import DensityTargetBuilder
from .utils import density_map_to_points, normalize_map


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    criterion = hydra.utils.instantiate(cfg.optim.loss)

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target = sample[0].to(device)
        # split channels to get input and target maps
        n_channels = input_and_target.shape[1]
        x_channels = cfg.model.module.in_channels
        y_channels = n_channels - x_channels
        images, gt_dmaps = input_and_target.split((x_channels, y_channels), dim=1)

        # computing pred dmaps
        pred_dmaps = model(images)

        # computing loss and backwarding it
        loss = criterion(pred_dmaps, gt_dmaps)
        loss.backward()

        batch_metrics = {'loss': loss.item()}
        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.optim.batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.optim.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


def _save_image_and_density_maps(image, image_id, pred_dmap, gt_dmap, cfg):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image_id = Path(image_id)

    image = (255 * image.cpu().squeeze().numpy()).astype(np.uint8)
    pil_image = Image.fromarray(image).convert("RGB")
    pil_image.save(debug_dir / image_id)

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, cfg.misc.font_size)

    def _annotate_density_map(density_map, prefix):
        count = density_map.sum()
        density_map = normalize_map(density_map)
        density_map = (255 * density_map).astype(np.uint8)

        pil_density_map = Image.fromarray(density_map)
        draw = ImageDraw.Draw(pil_density_map)
        text = f"{prefix} Num of Cells: {count}"
        draw.text((cfg.misc.text_pos, cfg.misc.text_pos), text=text, font=font, fill=191)
        return pil_density_map

    for i in range(pred_dmap.shape[2]):
        pil_pred_dmap = _annotate_density_map(pred_dmap[:, :, i], 'Det')
        pil_pred_dmap.save(debug_dir / f'{image_id.stem}_pred_dmap_cls{i}.png')

        pil_gt_dmap = _annotate_density_map(gt_dmap[:, :, i], 'GT')
        pil_gt_dmap.save(debug_dir / f'{image_id.stem}_gt_dmap_cls{i}.png')


@torch.no_grad()
def _process_fn(batch, model, device, cfg, has_target=True, return_input=True, return_target=True):
    batch = batch.to(device)

    if has_target:
        # split channels to get input and target
        n_channels = batch.shape[1]
        x_channels = cfg.model.module.in_channels
        y_channels = n_channels - x_channels
        patches, gt_dmaps = batch.split((x_channels, y_channels), dim=1)
    else:
        patches = batch

    # pad to mitigate border errors
    pad = cfg.optim.border_pad
    padded_patches = F.pad(patches, pad)

    # compute predictions
    predicted_density_patches = model(padded_patches)

    # rescale target and predicted pixel values 
    scale_factor = cfg.data.validation.target_params.target_normalize_scale_factor
    predicted_density_patches /= scale_factor
    if has_target:
        gt_dmaps /= scale_factor
    
    # unpad
    h, w = predicted_density_patches.shape[2:]
    predicted_density_patches = predicted_density_patches[:, :, pad:(h-pad), pad:(w-pad)]

    # prepare data
    processed_batch = (predicted_density_patches,)
    if return_target and has_target:
        processed_batch = (gt_dmaps,) + processed_batch
    if return_input:
        processed_batch = (patches,) + processed_batch

    processed_batch = [x.movedim(1, -1).to(device) for x in processed_batch]
    return processed_batch


@torch.no_grad()
def _collate_fn(image_info, image_patches, device, has_input=True, has_target=True, return_numpy=False):
    image_id, image_hw = image_info

    # image = None
    # target_density_map = None
    predicted_density_map = None
    normalization_map = None

    # build full maps from patches
    for (patch_hw, start_yx), datum in image_patches:

        if has_input and has_target:
            patch, target_density_patch, predicted_density_patch = datum
        elif has_input:
            patch, predicted_density_patch = datum
        elif has_target:
            target_density_patch, predicted_density_patch = datum
        else:
            predicted_density_patch, = datum
        
        if predicted_density_map is None:
            out_channels = predicted_density_patch.shape[-1]
            out_hwc = image_hw + (out_channels,)

            if has_input:
                in_channels = patch.shape[-1]
                in_hwc = image_hw + (in_channels,)
                image = torch.empty(in_hwc, dtype=torch.float32, device=device)

            if has_target:
                target_density_map = torch.zeros(out_hwc, dtype=torch.float32, device=device)

            predicted_density_map = torch.zeros(out_hwc, dtype=torch.float32, device=device)
            normalization_map = torch.zeros(out_hwc, dtype=torch.float32, device=device)

        (y, x), (h, w) = start_yx, patch_hw
        if has_input:
            image[y:y+h, x:x+w] = patch[:h, :w]
        if has_target:
            target_density_map[y:y+h, x:x+w] += target_density_patch[:h, :w]
        predicted_density_map[y:y+h, x:x+w] += predicted_density_patch[:h, :w]
        normalization_map[y:y+h, x:x+w] += 1.0

    if has_target:
        target_density_map /= normalization_map
    predicted_density_map /= normalization_map
    del normalization_map

    if return_numpy:
        if has_input:
            image = image.cpu().numpy()
        if has_target:
            target_density_map = target_density_map.cpu().numpy()
        predicted_density_map = predicted_density_map.cpu().numpy()

    result = (image_id, image_hw)
    if has_input:
        result += (image,)
    if has_target:
        result += (target_density_map,)
    result += (predicted_density_map,)

    return result


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device
    criterion = hydra.utils.instantiate(cfg.optim.loss)
    
    process_fn = partial(_process_fn, model=model, device=validation_device, cfg=cfg, has_target=True, return_input=True, return_target=True)
    collate_fn = partial(_collate_fn, device=validation_device, has_input=True, has_target=True, return_numpy=False)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, max_prefetch=1)
    
    metrics = []
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image_hw, image, target_density_map, predicted_density_map in progress:
        # threshold-free metrics
        loss = criterion(predicted_density_map, target_density_map)

        target_density_map = target_density_map.cpu().numpy().squeeze()
        predicted_density_map = predicted_density_map.cpu().numpy().squeeze()
        drange = predicted_density_map.max() - predicted_density_map.min()
        val_ssim = ssim(target_density_map, predicted_density_map, data_range=drange)

        count_metrics = counting(target_density_map, predicted_density_map)

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'density/mse_loss': loss.item(),
            **val_ssim,
            **count_metrics,
        })

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_and_density_maps(image, image_id, predicted_density_map, target_density_map, cfg)

        progress.set_description('EVAL (patches)')

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=0):
    """ Make predictions on data. """
    model.eval()
    
    process_fn = partial(_process_fn, model=model, device=device, cfg=cfg, has_target=False, return_input=True, return_target=False)
    collate_fn = partial(_collate_fn, device=device, has_input=True, has_target=False, return_numpy=True)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, max_prefetch=1)

    all_metrics = []
    all_yx_metrics = []
    all_dmap_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]
    density_map_builder = DensityTargetBuilder(**cfg.data.validation.target_params)

    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for n_i, (image_id, image_hw, image, density_map) in enumerate(progress):
        n_classes = density_map.shape[2]
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[[image_id]].copy()
        if 'AV' in groundtruth.columns:  # for the PNN dataset only
            groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and n_i < debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            for i in range(n_classes):
                normalized_dmap = normalize_map(density_map[:, :, i])
                normalized_dmap = (255 * normalized_dmap).astype(np.uint8)
                io.imsave(outdir / f'dmap_cls{i}_{image_id}', normalized_dmap)
                del normalized_dmap

        # compute dmap metrics (no thresholding or peak finding)
        gt_dmap = density_map_builder.build(image_hw, groundtruth, n_classes=n_classes)
        # rescale target pixel values 
        gt_dmap /= cfg.data.validation.target_params.target_normalize_scale_factor
    
        dmap_metrics = counting(gt_dmap, density_map)
        dmap_metrics['imgName'] = image_id
        all_dmap_metrics.append(dmap_metrics)

        yx_metrics = counting_yx(groundtruth, density_map)
        yx_metrics['imgName'] = image_id
        all_yx_metrics.append(yx_metrics)

        min_distance = int(cfg.data.validation.target_params.sigma)
        # TODO instead of using target.sigma better to use estimated object radius
        tolerance = 1.25 * cfg.data.validation.target_params.sigma  # min distance to match points

        def _compute_thr_metrics(thr, density_map, min_distance, groundtruth, tolerance):
            localizations = density_map_to_points(density_map, min_distance, thr)

            # match groundtruths and predictions
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr

            # compute metrics
            metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            metrics['thr'] = thr
            metrics['imgName'] = image_id

            return groundtruth_and_predictions, metrics

        job_args = (density_map, min_distance, groundtruth, tolerance)
        jobs = [delayed(_compute_thr_metrics)(thr, *job_args) for thr in thrs]
        image_gt_and_preds, image_metrics = zip(*Parallel(n_jobs=-1)(jobs))

        all_metrics.extend(image_metrics)
        all_gt_and_preds.extend(image_gt_and_preds)

        if outdir and n_i < debug:
            outdir.mkdir(parents=True, exist_ok=True)

            # pick a threshold and draw that prediction set
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3/macro'].idxmin()
            gp = pd.concat(image_gt_and_preds, ignore_index=True)
            gp = gp[gp.thr == best_thr]

            radius = cfg.data.validation.target_params.sigma
            for i in range(n_classes):
                gp_i = gp[gp['class'] == i]
                image = draw_groundtruth_and_predictions(image, gp_i, radius=radius, marker='circle')
                io.imsave(outdir / f'annot_cls{i}_{image_id}', image)

    all_metrics = pd.DataFrame(all_metrics)
    all_yx_metrics = pd.DataFrame(all_yx_metrics)
    all_dmap_metrics = pd.DataFrame(all_dmap_metrics)
    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_gp.to_csv(outdir / 'all_gt_preds.csv.gz')
        all_metrics.to_csv(outdir / 'all_metrics.csv.gz')
        all_yx_metrics.to_csv(outdir / 'yx_metrics.csv.gz')
        all_dmap_metrics.to_csv(outdir / 'dmap_metrics.csv.gz')


@torch.no_grad()
def predict_points(dataloader, model, device, threshold, cfg):
    """ Predict and find points. """
    model.eval()
    
    process_fn = partial(_process_fn, model=model, device=device, cfg=cfg, has_target=False, return_input=False, return_target=False)
    collate_fn = partial(_collate_fn, device=device, has_input=False, has_target=False, return_numpy=True)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, progress=True, max_prefetch=1)

    all_localizations = []
    min_distance = int(cfg.data.validation.target_params.sigma)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, image_hw, density_map in progress:
        localizations = density_map_to_points(density_map, min_distance, threshold)
        localizations['imgName'] = image_id
        localizations['thr'] = threshold
        all_localizations.append(localizations)
    
    all_localizations = pd.concat(all_localizations, ignore_index=True)
    return all_localizations
