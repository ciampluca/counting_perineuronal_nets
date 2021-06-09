# -*- coding: utf-8 -*-
import copy
import itertools
import logging
import os

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
from prefetch_generator import BackgroundGenerator
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
import torchvision.transforms.functional as F

from omegaconf import DictConfig
import hydra

from datasets.perineural_nets_dmap_dataset import PerineuralNetsDMapDataset
from density.metrics import counting
from utils import seed_everything, update_dict

tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)

# Creating logger
log = logging.getLogger(__name__)


def save_image_and_density_maps(image, image_id, pred_dmap, gt_dmap):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image_id = Path(image_id)

    pil_image = F.to_pil_image(image.cpu()).convert("RGB")
    pil_image.save(debug_dir / image_id)

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, 100)

    def _annotate_density_map(density_map, prefix):
        count = density_map.sum()
        dmin, dmax = density_map.min(), density_map.max()
        density_map = (density_map - dmin) / (dmax - dmin)
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


def train_one_epoch(model, dataloader, optimizer, criterion, device, cfg, writer, epoch):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target = sample[0]
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, gt_dmaps = input_and_target.split(1, dim=1)
        # Expanding images to 3 channels
        images = images.expand(-1, 3, -1, -1)

        # Computing pred dmaps
        pred_dmaps = model(images)
        if cfg.model.name == "UNet":
            pred_dmaps /= 1000

        # Computing loss and backwarding it
        loss = criterion(pred_dmaps, gt_dmaps)
        loss.backward()

        batch_metrics = {
            'loss': loss.item()
        }

        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.train.batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.train.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, cfg, epoch):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.train.val_device

    @torch.no_grad()
    def _predict(batch):
        input_and_target, patch_hw, start_yx, image_hw, image_id = batch
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, gt_dmaps = input_and_target.split(1, dim=1)
        # Expanding images to 3 channels
        images = images.expand(-1, 3, -1, -1)
        # Padding image to mitigate problems when reconstructing
        pad = cfg.train.border_pad
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

    def _unbatch(batches):
        for batch in batches:
            yield from zip(*batch)

    processed_batches = map(_predict, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=7500)  # prefetch batches using threading
    processed_samples = _unbatch(processed_batches)

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

        if cfg.train.debug and epoch % cfg.train.debug == 0:
            save_image_and_density_maps(full_image, image_id, full_pred_dmap, full_gt_dmap)

        # Cleaning
        del normalization_map
        del full_gt_dmap
        del full_pred_dmap
        del full_image

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    return metrics


@hydra.main(config_path="conf/density_based", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    log.info(f"Run path: {Path.cwd()}")

    cfg = copy.deepcopy(hydra_cfg.technique)
    for _, v in hydra_cfg.items():
        update_dict(cfg, v)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    model_cache_dir = Path(hydra.utils.get_original_cwd()) / cfg.model.cache_folder
    torch.hub.set_dir(model_cache_dir)
    best_models_folder = Path('best_models')
    best_models_folder.mkdir(parents=True, exist_ok=True)

    # No possible to set checkpoint and pre-trained model at the same time
    assert not (cfg.model.resume and cfg.model.pretrained), "Only one between 'pretrained' and 'resume' can be specified."

    # Reproducibility
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # create tensorboard writer
    writer = SummaryWriter()

    # Creating training dataset and dataloader
    log.info(f"Loading training data of dataset {cfg.dataset.train.name}")
    train_dataset_params = cfg.dataset.train.params
    valid_dataset_params = cfg.dataset.validation.params
    log.info("Train input size: {0}x{0}".format(train_dataset_params.patch_size))

    train_transform = Compose([ToTensor(), RandomHorizontalFlip()])
    train_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.train.name}")

    train_dataset = train_dataset(transforms=train_transform, **train_dataset_params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Found {len(train_dataset)} samples in training dataset")

    # create validation dataset and dataloader
    log.info(f"Loading validation data")
    log.info("Validation input size: {0}x{0}".format(valid_dataset_params.patch_size))

    valid_batch_size = cfg.train.val_batch_size if cfg.train.val_batch_size else cfg.train.batch_size
    valid_transform = ToTensor()
    valid_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.validation.name}")
    valid_dataset = valid_dataset(transforms=valid_transform, **valid_dataset_params)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Found {len(valid_dataset)} samples in validation dataset")

    # Creating model
    log.info(f"Creating model")
    model = hydra.utils.get_class(f"models.{cfg.model.name}")
    skip_weights_loading = cfg.model.resume or cfg.model.pretrained
    model = model(skip_weights_loading=skip_weights_loading, **cfg.model.params)

    # Putting model to device
    model.to(device)

    # Constructing an optimizer
    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optimizer.name}")
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optimizer.params)
    scheduler = None
    if cfg.optimizer.scheduler is not None:
        scheduler = hydra.utils.get_class(f"torch.optim.lr_scheduler.{cfg.optimizer.scheduler.name}")
        scheduler = scheduler(optimizer, **cfg.optimizer.scheduler.params)

    # Setting criterion
    criterion = hydra.utils.get_class(f"torch.nn.{cfg.optimizer.loss.name}")()

    # Eventually resuming a pre-trained model
    if cfg.model.pretrained:
        log.info(f"Resuming pre-trained model")
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location=device, model_dir=cfg.model.cache_folder)
        else:
            pre_trained_model = torch.load(cfg.model.pretrained, map_location=device)
        model.load_state_dict(pre_trained_model['model'])

    start_epoch = 0
    best_validation_metrics = {}
    best_metrics_epoch = {}
    best_thresholds = {}

    train_log_path = 'train_log.csv'
    valid_log_path = 'valid_log.csv'

    train_log = pd.DataFrame()
    valid_log = pd.DataFrame()

    # optionally resume from a saved checkpoint
    # if cfg.model.resume:
    if Path('last.pth').exists():
        log.info(f"Resuming training from last checkpoint.")
        checkpoint = torch.load('last.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_validation_metrics = checkpoint['best_validation_metrics']
        best_metrics_epoch = checkpoint['best_metrics_epoch']
        if 'best_thresholds' in checkpoint:
            best_thresholds = checkpoint['best_thresholds']

        train_log = pd.read_csv(train_log_path)
        valid_log = pd.read_csv(valid_log_path)

    # Train loop
    log.info(f"Start training")
    progress = trange(start_epoch, cfg.train.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = \
            train_one_epoch(model, train_loader, optimizer, criterion, device, cfg, writer, epoch)
        scheduler.step()    # update lr scheduler

        train_metrics['epoch'] = epoch
        train_log = train_log.append(train_metrics, ignore_index=True)
        train_log.to_csv(train_log_path, index=False)

        # validation
        if (epoch + 1) % cfg.train.val_freq == 0:
            valid_metrics = validate(model, valid_loader, criterion, device, cfg, epoch)
            for metric, value in valid_metrics.items():
                writer.add_scalar(f'valid/{metric}', value, epoch)  # log to tensorboard

            for metric, value in valid_metrics.items():
                cur_best = best_validation_metrics.get(metric, None)
                if 'count/err' in metric:
                    continue  # ignore metric
                elif 'ssim' in metric:
                    is_new_best = cur_best is None or value > cur_best  # higher is better
                else:
                    is_new_best = cur_best is None or value < cur_best  # lower is better

                if is_new_best:  # save if new best metric is achieved
                    best_validation_metrics[metric] = value
                    best_metrics_epoch[metric] = epoch
                    best_thresholds[metric] = valid_metrics.get(f'{metric}_thr', None)

                    ckpt_path = best_models_folder / f"best_model_{cfg.dataset.validation.name}_metric_{metric.replace('/', '-')}.pth"
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': valid_metrics
                    }, ckpt_path)

            # save last checkpoint for resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_validation_metrics': best_validation_metrics,
                'best_metrics_epoch': best_metrics_epoch,
                'best_thresholds': best_thresholds
            }, 'last.pth')

            valid_metrics['epoch'] = epoch
            valid_log = valid_log.append(valid_metrics, ignore_index=True)
            valid_log.to_csv(valid_log_path, index=False)

    log.info("Training ended. Exiting....")


if __name__ == "__main__":
    main()
