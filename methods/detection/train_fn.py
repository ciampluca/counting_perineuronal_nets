# -*- coding: utf-8 -*-
from functools import partial
import logging
import math
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import torch
from torchvision.ops import boxes as box_ops
from tqdm import tqdm

from ..points.metrics import detection_and_counting, detection_average_precision
from ..points.match import match
from ..points.utils import draw_groundtruth_and_predictions
from .metrics import dice_jaccard
from .utils import check_empty_images, build_coco_compliant_batch, reduce_dict

tqdm = partial(tqdm, dynamic_ncols=True)

# creating logger
log = logging.getLogger(__name__)


def _save_image_with_boxes(
    image,
    image_id,
    det_boxes,
    det_labels,
    gt_boxes,
    gt_labels,
    cfg,
):

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image = (255 * image.squeeze()).astype(np.uint8)
    pil_image = Image.fromarray(image).convert("RGB")

    font_path = str(Path(hydra.utils.get_original_cwd()) / "font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, cfg.misc.font_size)

    class_labels = np.unique(np.hstack((det_labels, gt_labels)))
    for i in class_labels:
        class_pil_image = pil_image.copy()
        draw = ImageDraw.Draw(class_pil_image)

        for box in gt_boxes[gt_labels == i].tolist():
            draw.rectangle(box, outline='red', width=cfg.misc.bb_outline_width)

        for box in det_boxes[det_labels == i].tolist():
            draw.rectangle(box, outline='green', width=cfg.misc.bb_outline_width)

        # Add text to image
        text = f"Det Num of Cells (Cls#{i}): {len(det_boxes)}, GT Num of Cells (Cls#{i}): {len(gt_boxes)}"
        text_pos = cfg.misc.text_pos
        draw.text((text_pos, text_pos), text=text, font=font, fill=(0, 191, 255))
        class_pil_image.save(debug_dir / f'cls{i}_{image_id}')


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



def _collate_fn(image_info, image_patches, device, cfg):
    image_id, image_hw = image_info
    
    image = None
    normalization_map = None
    boxes = []
    labels = []
    scores = []

    # build full image with preds from patches
    for (patch_hw, start_yx), (patch, patch_boxes, patch_labels, patch_scores) in image_patches:

        if image is None:
            in_channels = patch.shape[-1]
            in_hwc = image_hw + (in_channels,)

            image = torch.empty(in_hwc, dtype=torch.float32, device=device)
            normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=device)

        (y, x), (h, w) = start_yx, patch_hw
        image[y:y+h, x:x+w] = patch[:h, :w]
        normalization_map[y:y+h, x:x+w] += 1.0
        if patch_boxes.nelement() != 0:
            patch_boxes += torch.as_tensor([x, y, x, y], device=device)
            boxes.append(patch_boxes)
            labels.append(patch_labels)
            scores.append(patch_scores)

    boxes = torch.cat(boxes) if len(boxes) else torch.empty(0, 4, dtype=torch.float32, device=device)
    labels = torch.cat(labels) if len(labels) else torch.empty(0, dtype=torch.int64, device=device)
    scores = torch.cat(scores) if len(scores) else torch.empty(0, dtype=torch.float32, device=device)

    # progress.set_description('PRED (cleaning)')
    # remove boxes with center outside the image     
    image_wh = torch.tensor(image_hw[::-1], device=device)
    boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
    boxes_center = boxes_center.round().long()
    keep = (boxes_center < image_wh).all(axis=1)

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    boxes_center = boxes_center[keep]  # we need those later

    # clip boxes to image limits
    ih, iw = image_hw
    l = torch.tensor([[0, 0, 0, 0]], device=device)
    u = torch.tensor([[iw, ih, iw, ih]], device=device)
    boxes = torch.max(l, torch.min(boxes, u))

    # filter boxes in the overlapped areas using nms
    xc, yc = boxes_center.T
    in_overlap_zone = normalization_map[yc, xc] != 1.0

    boxes_in_overlap = boxes[in_overlap_zone]
    labels_in_overlap = labels[in_overlap_zone]
    scores_in_overlap = scores[in_overlap_zone]
    # keep = box_ops.nms(boxes_in_overlap, scores_in_overlap, iou_threshold=cfg.model.module.nms)  # TODO check equivalence of batched_nms() and nms()
    keep = box_ops.batched_nms(boxes_in_overlap, scores_in_overlap, labels_in_overlap, iou_threshold=cfg.model.module.nms)

    boxes = torch.cat((boxes[~in_overlap_zone], boxes_in_overlap[keep]))
    labels = torch.cat((labels[~in_overlap_zone], labels_in_overlap[keep]))
    scores = torch.cat((scores[~in_overlap_zone], scores_in_overlap[keep]))

    image = image.cpu().numpy()
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()
    
    # cleaning
    del normalization_map

    return image_id, image_hw, image, boxes, labels, scores


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    @torch.no_grad()
    def process_fn(batch):
        # splits input and target building them to be coco compliant
        images, targets = build_coco_compliant_batch(batch)
        images = [i.to(device) for i in images]

        predictions = model(images)

        # prepare data for validation
        images = torch.stack(images)
        images = images.movedim(1, -1).to(validation_device)  # channel dim as last
        # targets_bbs = [t['boxes'].to(validation_device) for t in targets]
        # targets_labels = [t['labels'].to(validation_device) for t in targets]
        predictions_bbs = [p['boxes'].to(validation_device) for p in predictions]
        predictions_labels = [p['labels'].to(validation_device) for p in predictions]
        predictions_scores = [p['scores'].to(validation_device) for p in predictions]

        predictions_labels = [(p - 1) for p in predictions_labels]  # remove BG class

        processed_batch = (images, # targets_bbs, targets_labels,
            predictions_bbs, predictions_labels, predictions_scores)
        return processed_batch

    collate_fn = partial(_collate_fn, device=validation_device, cfg=cfg)

    metrics = []
    thr_metrics = []

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image_hw, image, boxes, labels, scores in progress:
        # compute metrics
        progress.set_description('EVAL (metrics)')

        out_hwc = image_hw + (cfg.model.module.out_channels,)

        groundtruth = dataloader.dataset.annot.loc[[image_id]]
        gt_points = groundtruth[['X', 'Y']].values
        gt_labels = groundtruth['class'].values

        half_box = cfg.data.validation.target_params.side / 2
        gt_boxes = np.hstack((gt_points - half_box, gt_points + half_box))

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_with_boxes(image, image_id, boxes, labels, gt_boxes, gt_labels, cfg)

        # threshold-dependent metrics
        image_thr_metrics = []

        thrs = torch.linspace(0, 1, 21).tolist() + [2, ]
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        for thr in progress_thrs:
            progress_thrs.set_description(f'thr={thr:.2f} (det)')

            keep = scores >= thr
            thr_boxes = boxes[keep]
            thr_labels = labels[keep]
            thr_scores = scores[keep]

            # segmentation metrics
            segm_metrics = dice_jaccard(gt_boxes, gt_labels, thr_boxes, thr_labels, thr_scores, out_hwc, thr=thr)

            # counting metrics
            localizations = (thr_boxes[:, :2] + thr_boxes[:, 2:]) / 2
            localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
            localizations['class'] = thr_labels
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

        average_precisions = detection_average_precision(image_thr_metrics)

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            **average_precisions,
        })

        thr_metrics.extend(image_thr_metrics)

        progress.set_description('EVAL (patches)')

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}

    # pick best threshold metrics
    thr_metrics = pd.DataFrame(thr_metrics).set_index(['image_id', 'thr'])
    mean_thr_metrics = thr_metrics.pivot_table(index='thr', values=thr_metrics.columns, aggfunc='mean')

    # TODO factor out common code that follows in evaluate()s
    def _get_agg_func(metric_name, idx=False):
        if metric_name.startswith('count/err'):
            if idx:
                return lambda x: x.abs().idxmin()
            return lambda x: min(x, key=abs)
        
        if metric_name.startswith('count/'):
            return 'idxmin' if idx else min
        
        return 'idxmax' if idx else max

    value_aggfuncs = {k: _get_agg_func(k, idx=False) for k in thr_metrics.columns}
    thr_aggfuncs = {k: _get_agg_func(k, idx=True) for k in thr_metrics.columns}
    
    best_thr_metrics = mean_thr_metrics.aggregate(value_aggfuncs).to_dict()
    best_thrs = mean_thr_metrics.aggregate(thr_aggfuncs).to_dict()

    best_thr_metrics = {k: {'value': v, 'threshold': best_thrs[k]} for k, v in best_thr_metrics.items()}
    metrics.update(best_thr_metrics)

    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(patches):
        inputs = list(i.to(device) for i in patches)
        predictions = model(inputs)
        # prepare data
        patches = patches.movedim(1, -1)  # channels as last dim
        boxes = [p['boxes'].to(device) for p in predictions]
        labels = [p['labels'].to(device) for p in predictions]
        scores = [p['scores'].to(device) for p in predictions]

        labels = [(p - 1) for p in labels]  # remove BG class
        return patches, boxes, labels, scores

    collate_fn = partial(_collate_fn, device=device, cfg=cfg)

    all_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED (patches)', leave=False)
    for image_id, image_hw, image, boxes, labels, scores in progress:
        image = (255 * image).astype(np.uint8)

        # TODO: check when there are no anns in the image
        groundtruth = dataloader.dataset.annot.loc[[image_id]].copy()
        if 'AV' in groundtruth.columns:  # for the PNN dataset only
            groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)

        localizations = (boxes[:, :2] + boxes[:, 2:]) / 2
        localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
        localizations['class'] = labels
        localizations['score'] = scores

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')

            thresholded_localizations = localizations[localizations.score >= thr].reset_index()

            # match groundtruths and predictions
            tolerance = 1.25 * (cfg.data.validation.target_params.side / 2)  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, thresholded_localizations, tolerance)

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
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3/macro'].idxmin()
            gp = pd.concat(image_gt_and_preds, ignore_index=True)
            gp = gp[gp.thr == best_thr]

            radius = cfg.data.validation.target_params.side // 2
            for i, gp_i in gp.groupby('class'):
                image = draw_groundtruth_and_predictions(image, gp_i, radius=radius, marker='square')
                io.imsave(outdir / f'annot_cls{i}_{image_id}', image)

    all_metrics = pd.DataFrame(all_metrics)
    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_gp.to_csv(outdir / 'all_gt_preds.csv.gz')
        all_metrics.to_csv(outdir / 'all_metrics.csv.gz')


@torch.no_grad()
def predict_points(dataloader, model, device, threshold, cfg):
    """ Predict and find points. """
    model.eval()

    @torch.no_grad()
    def process_fn(patches):
        predictions = model([i.to(device) for i in patches])
        boxes = [p['boxes'] for p in predictions]
        labels = [p['labels'] for p in predictions]
        scores = [p['scores'] for p in predictions]

        labels = [(p - 1) for p in labels]  # remove BG class
        return boxes, scores

    # TODO merge with _collate_fn and do something like:
    # collate_fn = partial(_collate_fn, device=device, cfg=cfg, **image=False**)

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=device)
        boxes = []
        labels = []
        scores = []

        # build full image with preds from patches
        for (patch_hw, start_yx), (patch_boxes, patch_labels, patch_scores) in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            normalization_map[y:y+h, x:x+w] += 1.0
            if patch_boxes.nelement() != 0:
                patch_boxes += torch.as_tensor([x, y, x, y], device=device)
                boxes.append(patch_boxes)
                labels.append(patch_labels)
                scores.append(patch_scores)

        boxes = torch.cat(boxes) if len(boxes) else torch.empty(0, 4, dtype=torch.float32, device=device)
        labels = torch.cat(labels) if len(labels) else torch.empty(0, dtype=torch.int64, device=device)
        scores = torch.cat(scores) if len(scores) else torch.empty(0, dtype=torch.float32, device=device)

        # progress.set_description('PRED (cleaning)')
        # remove boxes with center outside the image     
        image_wh = torch.tensor(image_hw[::-1], device=device)
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        boxes_center = boxes_center.round().long()
        keep = (boxes_center < image_wh).all(axis=1)

        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        boxes_center = boxes_center[keep]  # we need those later

        # clip boxes to image limits
        ih, iw = image_hw
        l = torch.tensor([[0, 0, 0, 0]], device=device)
        u = torch.tensor([[iw, ih, iw, ih]], device=device)
        boxes = torch.max(l, torch.min(boxes, u))

        # filter boxes in the overlapped areas using nms
        xc, yc = boxes_center.T
        in_overlap_zone = normalization_map[yc, xc] != 1.0

        boxes_in_overlap = boxes[in_overlap_zone]
        labels_in_overlap = labels[in_overlap_zone]
        scores_in_overlap = scores[in_overlap_zone]
        # keep = box_ops.nms(boxes_in_overlap, scores_in_overlap, iou_threshold=cfg.model.module.nms)  # TODO check equivalence of batched_nms() and nms()
        keep = box_ops.batched_nms(boxes_in_overlap, scores_in_overlap, labels_in_overlap, iou_threshold=cfg.model.module.nms)

        boxes = torch.cat((boxes[~in_overlap_zone], boxes_in_overlap[keep]))
        labels = torch.cat((labels[~in_overlap_zone], labels_in_overlap[keep]))
        scores = torch.cat((scores[~in_overlap_zone], scores_in_overlap[keep]))

        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()

        # cleaning
        del normalization_map

        return image_id, boxes, labels, scores

    all_localizations = []

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, progress=True)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED (patches)', leave=False)
    for image_id, boxes, labels, scores in progress:

        localizations = (boxes[:, :2] + boxes[:, 2:]) / 2
        localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
        localizations['class'] = labels
        localizations['score'] = scores

        localizations = localizations[localizations.score >= threshold].reset_index()
        localizations['imgName'] = image_id
        localizations['thr'] = threshold

        all_localizations.append(localizations)
    
    all_localizations = pd.concat(all_localizations, ignore_index=True)
    return all_localizations
