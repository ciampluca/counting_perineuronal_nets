from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from .metrics import counting
from .target_builder import CountmapTargetBuilder
from .utils import normalize_map


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
        # split channels to get input and target
        images, gt_targets = input_and_target.split(1, dim=1)
        # retrieving target pad
        target_patch_size = cfg.data.train.target_params['target_patch_size']
        # removing pad to images 
        images = images[:, :, int(target_patch_size/2):images.shape[2]-int(target_patch_size/2), int(target_patch_size/2):images.shape[3]-int(target_patch_size/2)]
        # expanding images to 3 channels
        images = images.expand(-1, 3, -1, -1)
        
        # computing outputs
        preds = model(images)

        # computing loss and backwarding it
        loss = criterion(preds, gt_targets)
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
    # TODO adjust csv
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


def _save_image_and_count_maps(image, image_id, pred_cmap, gt_cmap, cfg):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image_id = Path(image_id)

    pil_image = F.to_pil_image(image.cpu()).convert("RGB")
    pil_image.save(debug_dir / image_id)

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font_size = cfg.misc.font_size // cfg.data.validation.target_params['scale']
    text_pos = cfg.misc.text_pos // cfg.data.validation.target_params['scale']
    font = ImageFont.truetype(font_path, font_size)

    def _annotate_count_map(count_map, prefix):
        target_patch_size = cfg.data.validation.target_params['target_patch_size']
        count = (count_map / (target_patch_size ** 2.0)).sum()
        count_map = normalize_map(count_map)
        count_map = (255 * count_map).astype(np.uint8)

        pil_count_map = Image.fromarray(count_map)
        draw = ImageDraw.Draw(pil_count_map)
        text = f"{prefix} Num of Cells: {round(count, 2)}"
        draw.text((text_pos, text_pos), text=text, font=font, fill=191)
        return pil_count_map

    pil_pred_cmap = _annotate_count_map(pred_cmap, 'Det')
    pil_pred_cmap.save(debug_dir / f'{image_id.stem}_pred_cmap.png')

    pil_gt_cmap = _annotate_count_map(gt_cmap, 'GT')
    pil_gt_cmap.save(debug_dir / f'{image_id.stem}_gt_cmap.png')


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
        patches, gt_cmaps = input_and_target.split(1, dim=1)
        # retrieving target pad
        target_patch_size = cfg.data.validation.target_params['target_patch_size']
        # removing pad to patches 
        patches = patches[:, :, int(target_patch_size/2):patches.shape[2]-int(target_patch_size/2), int(target_patch_size/2):patches.shape[3]-int(target_patch_size/2)]
        # expanding patches to 3 channels
        patches = patches.expand(-1, 3, -1, -1)

        # Computing predictions
        predicted_cmap_patches = model(patches)

        # prepare data for validation
        processed_batch = (patches, gt_cmaps, predicted_cmap_patches)
        processed_batch = [x[:, 0].to(validation_device) for x in processed_batch]
        return processed_batch

    def collate_fn(image_info, image_patches):
        # TODO: add support for rebuilding count maps from patches (not sure that it makes sense with norm map)
        # TODO: check and eventually add support for image divided in patches
        image_id, image_hw = image_info   
        # retrieving scale and target pad
        scale, target_patch_size = cfg.data.validation.target_params['scale'], cfg.data.validation.target_params['target_patch_size']
        # eventually scale image_hw
        image_hw = image_hw[0] // scale, image_hw[1] // scale
        # compute target size adding target pad
        target_hw = image_hw[0] + target_patch_size, image_hw[1] + target_patch_size
        
        image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        target_count_map = torch.zeros(target_hw, dtype=torch.float32, device=validation_device)
        predicted_count_map = torch.zeros(target_hw, dtype=torch.float32, device=validation_device)
        # normalization_map = torch.zeros(target_hw, dtype=torch.float32, device=validation_device)

        # build full maps from patches
        for (patch_hw, start_yx), (patch, target_cmap_patch, predicted_cmap_patch) in image_patches:
            scale = cfg.data.validation.target_params['scale']
            target_patch_size = cfg.data.validation.target_params['target_patch_size']
            patch_hw = patch_hw // scale
            target_patch_hw = patch_hw + target_patch_size
            
            (y, x), (h, w) = start_yx, patch_hw
            image[y:y+h, x:x+w] = patch[:h, :w]
            (y, x), (h, w) = start_yx, target_patch_hw
            target_count_map[y:y+h, x:x+w] += target_cmap_patch[:h, :w]
            predicted_count_map[y:y+h, x:x+w] += predicted_cmap_patch[:h, :w]
            # normalization_map[y:y+h, x:x+w] += 1.0

        # target_count_map /= normalization_map
        # predicted_count_map /= normalization_map
        # del normalization_map

        return image_id, image, predicted_count_map, target_count_map

    metrics = []
    
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image, predicted_count_map, target_count_map in progress:
        # threshold-free metrics
        loss = criterion(predicted_count_map, target_count_map)

        target_count_map = target_count_map.cpu().numpy()
        predicted_count_map = predicted_count_map.cpu().numpy()

        count_metrics = counting(target_count_map, predicted_count_map, patch_size=cfg.data.validation.target_params['target_patch_size'])

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'density/mae_loss': loss.item(),
            **count_metrics,
        })

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_and_count_maps(image, image_id, predicted_count_map, target_count_map, cfg)

        progress.set_description('EVAL (patches)')

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    # TODO adjust csv
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()
    # pad = cfg.optim.border_pad
    count_map_builder = CountmapTargetBuilder(**cfg.data.validation.target_params)

    @torch.no_grad()
    def process_fn(patches):
        # retrieving scale
        scale = cfg.data.validation.target_params['scale']
        # TODO probably it should be better to resize images before they are tensors
        h, w = patches.shape[2] // scale, patches.shape[3] // scale
        patches = F.resize(patches, size=(h,w), antialias=True)
        # expanding patches to 3 channels
        patches = patches.expand(-1, 3, -1, -1)

        # predict
        predicted_cmap_patches = model(patches.to(device))
        
        # prepare
        processed_patches = (patches, predicted_cmap_patches)
        processed_patches = [x[:, 0] for x in processed_patches]
        return processed_patches

    def collate_fn(image_info, image_patches):
        # TODO: add support for rebuilding count maps from patches (not sure that it makes sense with norm map)
        # TODO: check and eventually add support for image divided in patches
        image_id, image_hw = image_info
        # retrieving scale and target pad
        scale, target_patch_size = cfg.data.validation.target_params['scale'], cfg.data.validation.target_params['target_patch_size']
        # eventually scale image_hw
        image_hw = image_hw[0] // scale, image_hw[1] // scale
        # compute target size adding target pad
        target_hw = image_hw[0] + target_patch_size, image_hw[1] + target_patch_size
        
        image = torch.empty(image_hw, dtype=torch.float32, device=device)
        predicted_count_map = torch.zeros(target_hw, dtype=torch.float32, device=device)
        # normalization_map = torch.zeros(target_hw, dtype=torch.float32, device=device)

        for (patch_hw, start_yx), (patch, predicted_cmap_patch) in image_patches:
            scale = cfg.data.validation.target_params['scale']
            target_patch_size = cfg.data.validation.target_params['target_patch_size']
            patch_hw = patch_hw // scale
            target_patch_hw = patch_hw + target_patch_size
            
            (y, x), (h, w) = start_yx, patch_hw
            image[y:y+h, x:x+w] = patch[:h, :w]
            (y, x), (h, w) = start_yx, target_patch_hw
            predicted_count_map[y:y+h, x:x+w] += predicted_cmap_patch[:h, :w]
            # normalization_map[y:y+h, x:x+w] += 1.0

        # predicted_density_map /= normalization_map
        # del normalization_map

        image = image.cpu().numpy()
        predicted_count_map = predicted_count_map.cpu().numpy()

        return image_id, image_hw, image, predicted_count_map

    all_cmap_metrics = []

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, image_hw, image, count_map in progress:
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[[image_id]].copy()

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            normalized_cmap = normalize_map(count_map)
            normalized_cmap = (255 * normalized_cmap).astype(np.uint8)
            io.imsave(outdir / f'cmap_{image_id}', normalized_cmap)
            del normalized_cmap

        # compute cmap metrics
        gt_points = groundtruth[['Y', 'X']].values
        scale = cfg.data.validation.target_params['scale']
        original_image_hw = image_hw[0] * scale, image_hw[1] * scale
        gt_cmap = count_map_builder.build(original_image_hw, gt_points)
        target_patch_size = cfg.data.validation.target_params['target_patch_size']
        cmap_metrics = counting(gt_cmap, count_map, target_patch_size)
        cmap_metrics['imgName'] = image_id
        all_cmap_metrics.append(cmap_metrics)

    all_cmap_metrics = pd.DataFrame(all_cmap_metrics)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_cmap_metrics.to_csv(outdir / 'cmap_metrics.csv.gz')
