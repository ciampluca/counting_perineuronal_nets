import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from prefetch_generator import BackgroundGenerator
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from .metrics import ssim, counting
from .utils import normalize_map

from utils import unbatch


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
        # split channels to get input, target, and loss weights
        images, gt_dmaps = input_and_target.split(1, dim=1)
        # expanding images to 3 channels
        images = images.expand(-1, 3, -1, -1)

        # computing pred dmaps
        pred_dmaps = model(images)
        if cfg.model.name == "UNet":
            pred_dmaps /= 1000

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


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    criterion = hydra.utils.instantiate(cfg.optim.loss)

    @torch.no_grad()
    def _predict(batch):
        input_and_target, patch_hw, start_yx, image_hw, image_id = batch
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, gt_dmaps = input_and_target.split(1, dim=1)
        # expanding images to 3 channels
        images = images.expand(-1, 3, -1, -1)

        # Padding image to mitigate problems when reconstructing
        pad = cfg.optim.border_pad
        padded_images = F.pad(images, pad)

        # Computing predictions
        pred_dmaps = model(padded_images)

        # Removing previously added pad
        h, w = pred_dmaps.shape[2:]
        pred_dmaps = pred_dmaps[:, :, pad:(h-pad), pad:(w-pad)]

        # prepare data for validation
        images, gt_dmaps, pred_dmaps = map(lambda x: x[:, 0].to(validation_device), (images, gt_dmaps, pred_dmaps))

        processed_batch = (image_id, image_hw, images, gt_dmaps, pred_dmaps, patch_hw, start_yx)
        return processed_batch

    processed_batches = map(_predict, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=7500)  # prefetch batches using threading
    processed_samples = unbatch(processed_batches)

    metrics = []

    # group by image_id (and image_hw for convenience) --> iterate over full images
    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)

    n_images = len(dataloader.dataset)
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        n_images = len(dataloader.dataset.datasets)

    progress = tqdm(groups, total=n_images, desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in progress:
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        full_gt_dmap = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_pred_dmap = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        # build full maps from patches
        progress.set_description('EVAL (patches)')
        for _, _, patch, gt_dmap, pred_dmap, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y+h, x:x+w] = patch[:h, :w]
            full_gt_dmap[y:y+h, x:x+w] += gt_dmap[:h, :w]
            full_pred_dmap[y:y+h, x:x+w] += pred_dmap[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        full_gt_dmap /= normalization_map
        full_pred_dmap /= normalization_map

        ## threshold-free metrics
        loss = criterion(full_pred_dmap, full_gt_dmap)

        full_gt_dmap = full_gt_dmap.cpu().numpy()
        full_pred_dmap = full_pred_dmap.cpu().numpy()
        drange = full_pred_dmap.max() - full_pred_dmap.min()
        val_ssim = ssim(full_gt_dmap, full_pred_dmap, data_range=drange)

        count_metrics = counting(full_gt_dmap, full_pred_dmap)

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'density/mse_loss': loss.item(),
            'density/ssim': val_ssim,
            **count_metrics,
        })

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_and_density_maps(full_image, image_id, full_pred_dmap, full_gt_dmap)

        # Cleaning
        del normalization_map
        del full_gt_dmap
        del full_pred_dmap
        del full_image

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


def _save_image_and_density_maps(image, image_id, pred_dmap, gt_dmap):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image_id = Path(image_id)

    pil_image = F.to_pil_image(image.cpu()).convert("RGB")
    pil_image.save(debug_dir / image_id)

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, 100)

    def _annotate_density_map(density_map, prefix):
        count = density_map.sum()
        density_map = normalize_map(density_map)
        density_map = (255 * density_map).astype(np.uint8)

        pil_density_map = Image.fromarray(density_map)
        draw = ImageDraw.Draw(pil_density_map)
        text = f"{prefix} Num of Nets: {count}"
        draw.text((75, 75), text=text, font=font, fill=191)
        return pil_density_map

    pil_pred_dmap = _annotate_density_map(pred_dmap, 'Det')
    pil_pred_dmap.save(debug_dir / f'{image_id.stem}_pred_dmap.png')

    pil_gt_dmap = _annotate_density_map(gt_dmap, 'GT')
    pil_gt_dmap.save(debug_dir / f'{image_id.stem}_gt_dmap.png')
