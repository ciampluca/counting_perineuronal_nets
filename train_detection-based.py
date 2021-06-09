# -*- coding: utf-8 -*-
import os
import copy
import itertools
import logging
import math
import random

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import boxes as box_ops
from torchvision.transforms.functional import to_pil_image

from omegaconf import DictConfig
import hydra

from detection.metrics import dice_jaccard
from detection.transforms import Compose, RandomHorizontalFlip, ToTensor
from detection.utils import check_empty_images, collate_fn, build_coco_compliant_batch
from models import faster_rcnn
from points.metrics import detection_and_counting
from points.match import match
from utils import reduce_dict, seed_everything, update_dict

tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)

# Creating logger
log = logging.getLogger(__name__)


def save_image_with_boxes(image, image_id, det_boxes, gt_boxes, cfg):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    pil_image = to_pil_image(image.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for box in gt_boxes:
        draw.rectangle(box, outline='red', width=cfg.misc.bb_outline_width)
    
    for box in det_boxes:
        draw.rectangle(box, outline='green', width=cfg.misc.bb_outline_width)

    # Add text to image
    text = f"Det Num of Cells: {len(det_boxes)}, GT Num of Cells: {len(gt_boxes)}"

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, cfg.misc.font_size)

    text_pos = cfg.misc.text_pos
    draw.text((text_pos, text_pos), text=text, font=font, fill=(0, 191, 255))
    pil_image.save(debug_dir / image_id)


def train_one_epoch(model, dataloader, optimizer, device, cfg, writer, epoch):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target = sample[0]
        # splits input and target building them to be coco compliant
        images, targets = build_coco_compliant_batch(input_and_target)
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In case of empty images (i.e, without bbs), we handle them as negative images
        # (i.e., images with only background and no object), creating a fake object that represent the background
        # class and does not affect training
        # https://discuss.pytorch.org/t/torchvision-faster-rcnn-empty-training-images/46935/12
        targets = check_empty_images(targets)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            log.error(f"Loss is {loss_value}, stopping training")
            exit(1)

        losses.backward()

        batch_metrics = {
            'loss': loss_value
        }

        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.optim.batch_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.optim.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    return metrics


@torch.no_grad()
def validate(model, dataloader, device, cfg, epoch):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    @torch.no_grad()
    def _predict(batch):
        input_and_target, *patch_info = batch
        # splits input and target building them to be coco compliant
        images, targets = build_coco_compliant_batch(input_and_target)
        images = [i.to(device) for i in images]

        predictions = model(images)

        # prepare data for validation
        images = torch.stack(images)
        images = images.squeeze(dim=1).to(validation_device)
        targets_bbs = [t['boxes'].to(validation_device) for t in targets]
        predictions_bbs = [p['boxes'].to(validation_device) for p in predictions]
        predictions_scores = [p['scores'].to(validation_device) for p in predictions]

        patch_hw, start_yx, image_hw, image_id = patch_info
        processed_batch = (image_id, image_hw, images, targets_bbs, predictions_bbs, predictions_scores, patch_hw, start_yx)

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
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        boxes = []
        scores = []

        # build full image with preds from patches
        progress.set_description('EVAL (patches)')
        for _, _, patch, _, patch_boxes, patch_scores, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y+h, x:x+w] = patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0
            if patch_boxes.nelement() != 0:
                patch_boxes += torch.as_tensor([x, y, x, y])
                boxes.append(patch_boxes)
                scores.append(patch_scores)

        boxes = torch.cat(boxes) if len(boxes) else torch.empty(0, 4, dtype=torch.float32)
        scores = torch.cat(scores) if len(scores) else torch.empty(0, dtype=torch.float32)

        progress.set_description('EVAL (cleaning)')
        # remove boxes with center outside the image     
        image_wh = torch.tensor(image_hw[::-1])
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        boxes_center = boxes_center.round().long()
        keep = (boxes_center < image_wh).all(axis=1)

        boxes = boxes[keep]
        scores = scores[keep]
        boxes_center = boxes_center[keep]  # we need those later

        # clip boxes to image limits
        ih, iw = image_hw
        l = torch.tensor([[0, 0, 0, 0]])
        u = torch.tensor([[iw, ih, iw, ih]])   
        boxes = torch.max(l, torch.min(boxes, u))

        # filter boxes in the overlapped areas using nms
        xc, yc = boxes_center.T
        in_overlap_zone = normalization_map[yc, xc] != 1.0

        boxes_in_overlap = boxes[in_overlap_zone]
        scores_in_overlap = scores[in_overlap_zone]
        keep = box_ops.nms(boxes_in_overlap, scores_in_overlap, iou_threshold=cfg.model.params.nms)

        boxes = torch.cat((boxes[~in_overlap_zone], boxes_in_overlap[keep]))
        scores = torch.cat((scores[~in_overlap_zone], scores_in_overlap[keep]))

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        # cleaning
        del normalization_map

        # compute metrics
        progress.set_description('EVAL (metrics)')

        # threshold-dependent metrics
        image_thr_metrics = []

        groundtruth = dataloader.dataset.annot.loc[image_id]
        gt_points = groundtruth[['X', 'Y']].values

        half_box = cfg.dataset.validation.params.target_params.side / 2
        gt_boxes = np.hstack((gt_points - half_box, gt_points + half_box))

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            save_image_with_boxes(full_image, image_id, boxes, gt_boxes, cfg)

        thrs = torch.linspace(0, 1, 21).tolist() + [2, ]
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        for thr in progress_thrs:
            progress_thrs.set_description(f'thr={thr:.2f} (det)')

            keep = scores >= thr
            thr_boxes = boxes[keep]
            thr_scores = scores[keep]

            # segmentation metrics
            dice, jaccard = dice_jaccard(gt_boxes, thr_boxes, thr_scores, image_hw, thr=thr)

            segm_metrics = {
                'segm/dice': dice.item(),
                'segm/jaccard': jaccard.item()
            }

            # counting metrics
            localizations = (thr_boxes[:, :2] + thr_boxes[:, 2:]) / 2
            localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
            localizations['score'] = thr_scores

            tolerance = 1.25 * half_box  # min distance to match points
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


def compute_dataset_splits(cfg):
    image_paths = os.listdir(cfg.dataset.train.params.root)
    image_paths = [i for i in image_paths if i.endswith("cell.png")]
    
    n_train_samples = cfg.dataset.train.params.num_samples
    n_val_samples = cfg.dataset.validation.params.num_samples

    # the dataset contains 200 images; 100 should be for test
    if n_train_samples + n_val_samples > 100:
        log.error(f"Splits train+val can contain a maximum of 100 images")

    random.shuffle(image_paths)
    
    train_images = image_paths[:n_train_samples]
    valid_images = image_paths[n_train_samples:n_train_samples + n_val_samples]
    test_images = image_paths[n_train_samples + n_val_samples:]
    
    train_split = pd.DataFrame({'name': train_images})
    valid_split = pd.DataFrame({'name': valid_images})
    test_split = pd.DataFrame({'name': test_images})

    train_split['split'] = 'train'
    valid_split['split'] = 'validation'
    test_split['split'] = 'test'

    splits = pd.concat((train_split, valid_split, test_split), ignore_index=True)
    splits.to_csv('dataset_splits.csv', index=False)  # saving indexes for testing

    return train_images, valid_images


@hydra.main(config_path="conf/detection_based", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    log.info(f"Run path: {Path.cwd()}")

    cfg = copy.deepcopy(hydra_cfg.technique)
    for _, v in hydra_cfg.items():
        update_dict(cfg, v)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    torch.hub.set_dir(cfg.model.params.cache_folder)
    best_models_folder = Path('best_models')
    best_models_folder.mkdir(parents=True, exist_ok=True)

    # Cannot set checkpoint and pre-trained model at the same time
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

    if cfg.dataset.train.name == "VGGCellsDataset":
        train_images, valid_images = compute_dataset_splits(cfg)
        train_dataset_params['image_names'] = train_images
        valid_dataset_params['image_names'] = valid_images

    train_transform = Compose([RandomHorizontalFlip(), ToTensor()])
    train_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.train.name}")

    train_dataset = train_dataset(transforms=train_transform, **train_dataset_params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers,
        collate_fn=collate_fn,
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
        num_workers=cfg.optim.num_workers,
        collate_fn=collate_fn,
    )
    log.info(f"Found {len(valid_dataset)} samples in validation dataset")

    # Creating model
    log.info(f"Creating model")
    model = hydra.utils.get_class(f"models.{cfg.model.name}")
    skip_weights_loading = cfg.model.resume or cfg.model.pretrained
    model = model(skip_weights_loading=skip_weights_loading, **cfg.model.params)

    # move model to device
    model.to(device)

    # build the optimizer
    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optim.optimizer.name}")
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optim.optimizer.params)

    scheduler = None
    if cfg.optim.lr_scheduler is not None:
        scheduler = hydra.utils.get_class(f"torch.optim.lr_scheduler.{cfg.optim.lr_scheduler.name}")
        scheduler = scheduler(optimizer, **cfg.optim.lr_scheduler.params)

    # optionally load pre-trained weights
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
    progress = trange(start_epoch, cfg.optim.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg, writer, epoch)
        scheduler.step()  # update lr scheduler

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
