# -*- coding: utf-8 -*-
import itertools
import logging
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

import torch
from torchvision.ops import boxes as box_ops
from torchvision.transforms.functional import to_pil_image

import hydra

from detection.metrics import dice_jaccard
from detection.utils import check_empty_images, build_coco_compliant_batch
from points.metrics import detection_and_counting
from points.match import match
from utils import reduce_dict, unbatch

tqdm = partial(tqdm, dynamic_ncols=True)

# creating logger
log = logging.getLogger(__name__)


def _save_image_with_boxes(image, image_id, det_boxes, gt_boxes, cfg):

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


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        # splits input and target building them to be coco compliant
        images, targets = build_coco_compliant_batch(sample[0])
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
            exit(1)  # XXX this breaks hydra multirun

        losses.backward()

        batch_metrics = {'loss': loss_value}
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
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
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

    processed_batches = map(_predict, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=7500)  # prefetch batches using threading
    processed_samples = unbatch(processed_batches)

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
        keep = box_ops.nms(boxes_in_overlap, scores_in_overlap, iou_threshold=cfg.model.module.nms)

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

        half_box = cfg.data.validation.target_params.side / 2
        gt_boxes = np.hstack((gt_points - half_box, gt_points + half_box))

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_with_boxes(full_image, image_id, boxes, gt_boxes, cfg)

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
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}

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
    }).to_dict()

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
    }).to_dict()

    best_thr_metrics = {k: {'value': v, 'threshold': best_thrs[k]} for k, v in best_thr_metrics.items()}
    metrics.update(best_thr_metrics)

    return metrics


