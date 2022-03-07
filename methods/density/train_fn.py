from pathlib import Path

import hydra
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
        images, gt_dmaps = input_and_target.split((n_channels - 1, 1), dim=1)

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

    pil_pred_dmap = _annotate_density_map(pred_dmap, 'Det')
    pil_pred_dmap.save(debug_dir / f'{image_id.stem}_pred_dmap.png')

    pil_gt_dmap = _annotate_density_map(gt_dmap, 'GT')
    pil_gt_dmap.save(debug_dir / f'{image_id.stem}_gt_dmap.png')


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    criterion = hydra.utils.instantiate(cfg.optim.loss)

    @torch.no_grad()
    def process_fn(batch):
        input_and_target = batch.to(device)

        # split channels to get input and target
        n_channels = input_and_target.shape[1]
        patches, gt_dmaps = input_and_target.split((n_channels - 1, 1), dim=1)
            
        # pad to mitigate border errors
        pad = cfg.optim.border_pad
        padded_patches = F.pad(patches, pad)

        # Computing predictions
        predicted_density_patches = model(padded_patches)
        
        # Eventually rescale target and predicted pixel values 
        gt_dmaps /= cfg.data.validation.target_params.target_normalize_scale_factor
        predicted_density_patches /= cfg.data.validation.target_params.target_normalize_scale_factor

        # unpad
        h, w = predicted_density_patches.shape[2:]
        predicted_density_patches = predicted_density_patches[:, :, pad:(h-pad), pad:(w-pad)]

        # prepare data for validation
        processed_batch = (patches, gt_dmaps, predicted_density_patches)
        processed_batch = [x.movedim(1, -1).to(validation_device) for x in processed_batch]
        return processed_batch

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info

        image = None
        target_density_map = None
        predicted_density_map = None
        normalization_map = None

        # build full maps from patches
        for (patch_hw, start_yx), (patch, target_density_patch, predicted_density_patch) in image_patches:

            if predicted_density_map is None:
                in_channels = patch.shape[-1]
                out_channels = predicted_density_patch.shape[-1]

                in_hwc = image_hw + (in_channels,)
                out_hwc = image_hw + (out_channels,)

                image = torch.empty(in_hwc, dtype=torch.float32, device=validation_device)
                target_density_map = torch.zeros(out_hwc, dtype=torch.float32, device=validation_device)
                predicted_density_map = torch.zeros(out_hwc, dtype=torch.float32, device=validation_device)
                normalization_map = torch.zeros(out_hwc, dtype=torch.float32, device=validation_device)

            (y, x), (h, w) = start_yx, patch_hw
            image[y:y+h, x:x+w] = patch[:h, :w]
            target_density_map[y:y+h, x:x+w] += target_density_patch[:h, :w]
            predicted_density_map[y:y+h, x:x+w] += predicted_density_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        target_density_map /= normalization_map
        predicted_density_map /= normalization_map
        del normalization_map

        return image_id, image, predicted_density_map, target_density_map

    metrics = []
    
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image, predicted_density_map, target_density_map in progress:
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
            'density/ssim': val_ssim,
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
def predict(dataloader, model, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()
    pad = cfg.optim.border_pad
    density_map_builder = DensityTargetBuilder(**cfg.data.validation.target_params)

    @torch.no_grad()
    def process_fn(patches):
        # pad to mitigate border errors
        padded_patches = F.pad(patches, pad)
        # predict
        predicted_density_patches = model(padded_patches.to(device))
        # Eventually rescale prediction pixel values 
        predicted_density_patches /= cfg.data.validation.target_params.target_normalize_scale_factor
        # unpad
        h, w = predicted_density_patches.shape[2:]
        predicted_density_patches = predicted_density_patches[:, :, pad:(h - pad), pad:(w - pad)]
        # prepare
        processed_patches = (patches, predicted_density_patches)
        processed_patches = [x.movedim(1, -1) for x in processed_patches]
        return processed_patches

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info
        
        image = None
        predicted_density_map = None
        normalization_map = None

        for (patch_hw, start_yx), (patch, predicted_density_patch) in image_patches:

            if image is None:
                in_channels = patch.shape[-1]
                out_channels = predicted_density_patch.shape[-1]

                in_hwc = image_hw + (in_channels,)
                out_hwc = image_hw + (out_channels,)

                image = torch.empty(in_hwc, dtype=torch.float32, device=device)
                predicted_density_map = torch.zeros(out_hwc, dtype=torch.float32, device=device)
                normalization_map = torch.zeros(out_hwc, dtype=torch.float32, device=device)

            (y, x), (h, w) = start_yx, patch_hw
            image[y:y+h, x:x+w] = patch[:h, :w]
            predicted_density_map[y:y+h, x:x+w] += predicted_density_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        predicted_density_map /= normalization_map
        del normalization_map

        image = image.cpu().numpy()
        predicted_density_map = predicted_density_map.cpu().numpy()

        return image_id, image_hw, image, predicted_density_map

    all_metrics = []
    all_yx_metrics = []
    all_dmap_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, image_hw, image, density_map in progress:
        image = (255 * image).astype(np.uint8)
        density_map = density_map.squeeze()

        groundtruth = dataloader.dataset.annot.loc[[image_id]].copy()
        if 'AV' in groundtruth.columns:  # for the PNN dataset only
            groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            normalized_dmap = normalize_map(density_map)
            normalized_dmap = (255 * normalized_dmap).astype(np.uint8)
            io.imsave(outdir / f'dmap_{image_id}', normalized_dmap)
            del normalized_dmap

        # compute dmap metrics (no thresholding or peak finding)
        gt_points = groundtruth[['Y', 'X']].values
        gt_dmap = density_map_builder.build(image_hw, gt_points)
        # Eventually rescale target pixel values 
        gt_dmap /= cfg.data.validation.target_params.target_normalize_scale_factor
        dmap_metrics = counting(gt_dmap, density_map)
        dmap_metrics['imgName'] = image_id
        all_dmap_metrics.append(dmap_metrics)

        yx_metrics = counting_yx(gt_points, density_map)
        yx_metrics['imgName'] = image_id
        all_yx_metrics.append(yx_metrics)

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')

            min_distance = int(cfg.data.validation.target_params.sigma)
            localizations = density_map_to_points(density_map, min_distance, thr)

            # match groundtruths and predictions
            # TODO instead of using target.sigma better to use estimated obecjt radius
            tolerance = 1.25 * cfg.data.validation.target_params.sigma  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr
            image_gt_and_preds.append(groundtruth_and_predictions)

            # compute metrics
            metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            metrics['thr'] = thr
            metrics['imgName'] = image_id

            image_metrics.append(metrics)

        all_metrics.extend(image_metrics)
        all_gt_and_preds.extend(image_gt_and_preds)

        if outdir and debug:
            outdir.mkdir(parents=True, exist_ok=True)

            # pick a threshold and draw that prediction set
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3'].idxmin()
            gp = pd.concat(image_gt_and_preds, ignore_index=True)
            gp = gp[gp.thr == best_thr]

            image = draw_groundtruth_and_predictions(image, gp, radius=cfg.data.validation.target_params.sigma, marker='circle')
            io.imsave(outdir / f'annot_{image_id}', image)

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
    pad = cfg.optim.border_pad

    @torch.no_grad()
    def process_fn(patches):
        patches_rgb = patches.expand(-1, 3, -1, -1)  # gray to RGB
        # pad to mitigate border errors
        padded_patches = F.pad(patches_rgb, pad)
        # predict
        predicted_density_patches = model(padded_patches.to(device))
        # unpad
        h, w = predicted_density_patches.shape[2:]
        predicted_density_patches = predicted_density_patches[:, 0, pad:(h - pad), pad:(w - pad)]
        return (predicted_density_patches,)

    def collate_fn(image_info, image_patches):
        image_id, image_hwc = image_info
        image_hw = image_hwc[:2]
        predicted_density_map = torch.zeros(image_hw, dtype=torch.float32, device=device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=device)

        for (patch_hw, start_yx), (predicted_density_patch,) in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            predicted_density_map[y:y+h, x:x+w] += predicted_density_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        predicted_density_map /= normalization_map
        del normalization_map

        predicted_density_map = predicted_density_map.cpu().numpy()
        return image_id, predicted_density_map

    all_localizations = []

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, progress=True)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, density_map in progress:

        min_distance = int(cfg.data.validation.target_params.sigma)
        localizations = density_map_to_points(density_map, min_distance, threshold)
        localizations['imgName'] = image_id
        localizations['thr'] = threshold

        all_localizations.append(localizations)
    
    all_localizations = pd.concat(all_localizations, ignore_index=True)
    return all_localizations
