# -*- coding: utf-8 -*-
import os
import copy
import itertools
import logging

from functools import partial
from pathlib import Path
from segmentation.utils import segm_map_to_points

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm, trange

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

from omegaconf import DictConfig
import hydra

from points.metrics import detection_and_counting
from points.match import match
from segmentation.metrics import dice_jaccard
from segmentation.utils import segm_map_to_points
from utils import seed_everything, update_dict

tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)

# Creating logger
log = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, device, cfg, writer, epoch):
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
        images, targets, weights = input_and_target.split(1, dim=1)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weights)
        predictions = torch.sigmoid(logits)
        soft_dice, soft_jaccard = dice_jaccard(targets, predictions)

        loss.backward()

        batch_metrics = {
            'loss': loss.item(),
            'soft_dice': soft_dice.item(),
            'soft_jaccard': soft_jaccard.item()
        }

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

            if cfg.optim.debug and (i + 1) % cfg.optim.debug_freq == 0:
                writer.add_images('train/inputs', images, n_iter)
                writer.add_images('train/targets', targets, n_iter)
                writer.add_images('train/predictions', predictions, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    return metrics


@torch.no_grad()
def validate(model, dataloader, device, cfg, epoch):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    @torch.no_grad()
    def _predict(batch):
        input_and_target, patch_hw, start_yx, image_hw, image_id = batch
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, targets, weights = input_and_target.split(1, dim=1)
        logits = model(images)
        predictions = torch.sigmoid(logits)
        # remove single channel dimension, move to chosen device
        images, targets, weights, predictions = map(lambda x: x[:, 0].to(validation_device), (images, targets, weights, predictions))
        processed_batch = (image_id, image_hw, images, targets, weights, predictions, patch_hw, start_yx)
        return processed_batch

    def _unbatch(batches):
        for batch in batches:
            yield from zip(*batch)

    processed_batches = map(_predict, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=7500)  # prefetch batches using threading
    processed_samples = _unbatch(processed_batches)

    metrics = []
    thr_metrics = []
    
    # group by image_id (and image_hw for convenience) --> iterate over full images
    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)

    n_images = len(dataloader.dataset)
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        n_images = len(dataloader.dataset.datasets)

    progress = tqdm(groups, total=n_images, desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in progress:
        # full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        full_segm_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_target_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_weight_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        # build full maps from patches
        progress.set_description('EVAL (patches)')
        for _, _, patch, target, weights, prediction, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            # full_image[y:y+h, x:x+w] = patch[:h, :w]
            full_segm_map[y:y+h, x:x+w] += prediction[:h, :w]
            full_target_map[y:y+h, x:x+w] += target[:h, :w]
            full_weight_map[y:y+h, x:x+w] += weights[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        # full_image /= normalization_map
        full_segm_map /= normalization_map
        full_target_map /= normalization_map
        full_weight_map /= normalization_map

        full_segm_map = torch.clamp(full_segm_map, 0, 1)  # XXX to fix, sporadically something goes off limits
        del normalization_map

        # compute metrics
        progress.set_description('EVAL (metrics)')

        ## threshold-free metrics
        weighted_bce_loss = F.binary_cross_entropy(full_segm_map, full_target_map, full_weight_map)
        soft_dice, soft_jaccard = dice_jaccard(full_target_map, full_segm_map)
        
        ## threshold-dependent metrics
        thrs = torch.linspace(0, 1, 21).tolist() + [2,]

        image_thr_metrics = []
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        for thr in progress_thrs:
            full_bin_segm_map = full_segm_map >= thr

            # segmentation metrics
            progress_thrs.set_description(f'thr={thr:.2f} (segm)')
            dice, jaccard = dice_jaccard(full_target_map, full_bin_segm_map)

            segm_metrics = {
                'segm/dice': dice.item(),
                'segm/jaccard': jaccard.item()
            }
        
            # counting metrics
            progress_thrs.set_description(f'thr={thr:.2f} (count)')
            localizations = segm_map_to_points(full_segm_map.cpu().numpy(), thr=thr)
            groundtruth = dataloader.dataset.annot.loc[image_id]

            tolerance = 1.25 * cfg.dataset.validation.params.target_params.radius  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            count_pdet_metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)

            image_thr_metrics.append({
                'image_id': image_id,
                'thr': thr,
                **segm_metrics,
                **count_pdet_metrics
            })
        
        pr = pd.DataFrame(image_thr_metrics).sort_values('pdet/recall', ascending=False)
        recalls = pr['pdet/recall'].values
        precisions = pr['pdet/precision'].values
        average_precision = - np.sum(np.diff(recalls) * precisions[:-1])  # sklearn's ap

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'segm/weighted_bce_loss': weighted_bce_loss.item(),
            'segm/soft_dice': soft_dice.item(),
            'segm/soft_jaccard': soft_jaccard.item(),
            'pdet/average_precision': average_precision.item(),
        })

        thr_metrics.extend(image_thr_metrics)

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()

    # pick best threshold metrics
    thr_metrics = pd.DataFrame(thr_metrics).set_index(['image_id', 'thr'])
    mean_thr_metrics = thr_metrics.pivot_table(index='thr', values=thr_metrics.columns, aggfunc='mean')
    
    best_thr_metrics = mean_thr_metrics.aggregate({
        'segm/dice': max,
        'segm/jaccard': max,
        'pdet/precision': max,
        'pdet/recall': max,
        'pdet/f1_score': max,
        'count/err': lambda x: min(x, key=abs),
        'count/mae': min,
        'count/mse': min,
        'count/mare': min,
        **{f'count/game-{l}': min for l in range(6)}
    }).rename(lambda i: i + '_best').to_dict()

    best_thrs = mean_thr_metrics.aggregate({
        'segm/dice': 'idxmax',
        'segm/jaccard': 'idxmax',
        'pdet/precision': 'idxmax',
        'pdet/recall': 'idxmax',
        'pdet/f1_score': 'idxmax',
        'count/err': lambda x: x.abs().idxmin(),
        'count/mae': 'idxmin',
        'count/mse': 'idxmin',
        'count/mare': 'idxmin',
        **{f'count/game-{l}': 'idxmin' for l in range(6)}
    }).rename(lambda i: i + '_best_thr').to_dict()

    metrics.update(best_thr_metrics)
    metrics.update(best_thrs)

    return metrics


@hydra.main(config_path="conf/segmentation_based", config_name="config")
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
    assert not(cfg.model.resume and cfg.model.pretrained), "Only one between 'pretrained' and 'resume' can be specified."

    # Reproducibility
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # create tensorboard writer
    writer = SummaryWriter()

    # create train dataset and dataloader
    log.info(f"Loading training data of dataset {cfg.dataset.train.name}")

    train_dataset_params = cfg.dataset.train.params
    valid_dataset_params = cfg.dataset.validation.params
    log.info("Train input size: {0}x{0}".format(train_dataset_params.patch_size))

    train_transform = Compose([ToTensor(), RandomHorizontalFlip()])
    train_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.train.name}")

    train_dataset = train_dataset(transforms=train_transform, **train_dataset_params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers
    )
    log.info(f"Found {len(train_dataset)} samples in training dataset")

    # create validation dataset and dataloader
    log.info(f"Loading validation data")

    log.info("Validation input size: {0}x{0}".format(valid_dataset_params.patch_size))
    valid_batch_size = cfg.optim.val_batch_size if cfg.optim.val_batch_size else cfg.optim.batch_size
    valid_transform = ToTensor()
    valid_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.validation.name}")
    valid_dataset = valid_dataset(transforms=valid_transform, **valid_dataset_params)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=cfg.optim.num_workers
    )
    log.info(f"Found {len(valid_dataset)} samples in validation dataset")

    # create model
    log.info(f"Creating model")
    model = hydra.utils.get_class(f"models.{cfg.model.name}")
    model = model(**cfg.model.params)

    # move model to device
    model.to(device)

    # build the optimizer
    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optim.optimizer.name}")
    optimizer = optimizer(model.parameters(), **cfg.optim.optimizer.params)
    scheduler = None
    if cfg.optim.lr_scheduler is not None:
        scheduler = hydra.utils.get_class(f"torch.optim.lr_scheduler.{cfg.optim.lr_scheduler.name}")
        scheduler = scheduler(optimizer, **cfg.optim.lr_scheduler.params)

    # optionally load pre-trained weights
    if cfg.model.pretrained:
        log.info(f"Loading pre-trained model: {cfg.model.pretrained}")
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
    progress = trange(start_epoch, cfg.optim.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg, writer, epoch)
        scheduler.step() # update lr scheduler

        train_metrics['epoch'] = epoch
        train_log = train_log.append(train_metrics, ignore_index=True)
        train_log.to_csv(train_log_path, index=False)

        # evaluation
        if (epoch + 1) % cfg.optim.val_freq == 0:
            valid_metrics = validate(model, valid_loader, device, cfg, epoch)
            for metric, value in valid_metrics.items():
                writer.add_scalar(f'valid/{metric}', value, epoch)  # log to tensorboard

            for metric, value in valid_metrics.items():
                cur_best = best_validation_metrics.get(metric, None)
                if ('thr' in metric) or ('count/err' in metric):
                    continue  # ignore metric
                elif ('jaccard' in metric) or ('dice' in metric) or ('pdet' in metric):
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

