# -*- coding: utf-8 -*-
import logging
import copy
import os
import math
import sys
import numpy as np
from pathlib import Path
import collections
import pandas as pd
from tqdm import tqdm, trange
import itertools
from PIL import ImageDraw, ImageFont
from functools import partial
import random

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import boxes as box_ops

from omegaconf import DictConfig
import hydra

from prefetch_generator import BackgroundGenerator

from datasets.det_transforms import Compose, RandomHorizontalFlip, ToTensor
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn
from utils.misc import reduce_dict
from utils import points

tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)

# Creating logger
log = logging.getLogger(__name__)


def get_model_detection(num_classes, cfg, load_custom_model=False):
    if cfg.model.backbone not in ["resnet50", "resnet101", "resnet152"]:
        log.error(f"Backbone not supported")
        exit(1)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes += 1    # num classes + background

    if load_custom_model:
        model_pretrained = False
        backbone_pretrained = False
    else:
        model_pretrained = cfg.model.coco_model_pretrained
        backbone_pretrained = cfg.model.backbone_pretrained

    # anchor generator: these are default values, but maybe we have to change them
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    # these are default values, but maybe we can change them
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2)

    # Creating model
    if cfg.model.backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg.model.params.max_dets_per_image,
            box_nms_thresh=cfg.model.params.nms,
            box_score_thresh=cfg.model.params.det_thresh,
            model_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.model.cache_folder),
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg.model.backbone == "resnet101":
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg.model.params.max_dets_per_image,
            box_nms_thresh=cfg.model.params.nms,
            box_score_thresh=cfg.model.params.det_thresh,
            model_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.model.cache_folder),
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg.model.backbone == "resnet152":
        log.error(f"Model with ResNet152 to be implemented")
        exit(1)
    else:
        log.error(f"Not supported backbone")
        exit(1)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def check_empty_images(targets):
    if targets[0]['boxes'].is_cuda:
        device = targets[0]['boxes'].get_device()
    else:
        device = torch.device("cpu")

    for target in targets:
        if target['boxes'].nelement() == 0:
            target['boxes'] = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32, device=device)
            target['labels'] = torch.zeros((1,), dtype=torch.int64, device=device)
            target['iscrowd'] = torch.zeros((1,), dtype=torch.int64, device=device)

    return targets


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dice_jaccard(y_true, y_pred, y_pred_scores, shape, smooth=1, thr=None):
    """ Computes Dice and Jaccard coefficients for image segmentation. """
    gt_seg_map = np.zeros(shape, dtype=np.float32)
    for gt_bb in y_true:
        gt_seg_map[int(gt_bb[1]):int(gt_bb[3]) + 1, int(gt_bb[0]):int(gt_bb[2]) + 1] = 1.0

    det_seg_map = np.zeros_like(gt_seg_map)
    for det_bb, score in zip(y_pred, y_pred_scores):
        if thr is not None:
            if score < thr:
                continue
        det_seg_map[int(det_bb[1]):int(det_bb[3]) + 1, int(det_bb[0]):int(det_bb[2]) + 1] = \
            np.maximum(det_seg_map[int(det_bb[1]):int(det_bb[3]) + 1, int(det_bb[0]):int(det_bb[2]) + 1], score)

    intersection = np.sum(gt_seg_map * det_seg_map)
    sum_ = np.sum(gt_seg_map) + np.sum(det_seg_map)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)

    return dice.mean(), jaccard.mean()


def save_img_with_bbs(img, img_id, det_bbs, gt_bbs):

    def _is_empty(l):
        return all(_is_empty(i) if isinstance(i, list) else False for i in l)

    debug_dir = os.path.join(os.getcwd(), 'output_debug')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    pil_image = to_pil_image(img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    if not _is_empty(gt_bbs):
        for bb in gt_bbs:
            draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='red', width=3)
        img_gt_num = len(gt_bbs)
    if not _is_empty(det_bbs):
        for bb in det_bbs:
            draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='green', width=3)
        img_det_num = len(det_bbs)
    # Add text to image
    text = f"Det Num of Nets: {img_det_num}, GT Num of Nets: {img_gt_num}"
    font_path = os.path.join(hydra.utils.get_original_cwd(), "./font/LEMONMILK-RegularItalic.otf")
    font = ImageFont.truetype(font_path, 100)
    draw.text((75, 75), text=text, font=font, fill=(0, 191, 255))
    pil_image.save(os.path.join(debug_dir, img_id))


def train_one_epoch(model, dataloader, optimizer, device, cfg, writer, epoch):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target, patch_hw, start_yx, image_hw, image_id = sample
        # splits input and target building them to be coco compliant
        images, targets = dataloader.dataset.build_coco_compliant_batch(input_and_target)
        images = list(img.to(device) for img in images)
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
            sys.exit(1)

        losses.backward()

        batch_metrics = {
            'loss': loss_value
        }

        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

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
def validate(model, dataloader, device, cfg, epoch):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.train.val_device

    @torch.no_grad()
    def _predict(batch):
        input_and_target, patch_hw, start_yx, image_hw, image_id = batch
        # splits input and target building them to be coco compliant
        images, targets = dataloader.dataset.build_coco_compliant_batch(input_and_target)
        images = list(img.to(device) for img in images)

        predictions = model(images)

        # prepare data for validation
        images = torch.stack(images)
        images = images.squeeze(dim=1).to(validation_device)
        targets_bbs = [t['boxes'].to(validation_device) for t in targets]
        predictions_bbs = [p['boxes'].to(validation_device) for p in predictions]
        predictions_scores = [p['scores'].to(validation_device) for p in predictions]

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
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        num_imgs = len(dataloader.dataset.datasets)
    else:
        num_imgs = len(dataloader.dataset)
    progress = tqdm(groups, total=num_imgs, desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in progress:
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_image_det_bbs = torch.empty(0, 4, dtype=torch.float32)
        full_image_det_scores = torch.empty(0, dtype=torch.float32)

        # build full image with preds from patches
        progress.set_description('EVAL (patches)')
        for _, _, patch, _, prediction_bbs, prediction_scores, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y+h, x:x+w] = patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0
            if prediction_bbs.nelement() != 0:
                prediction_bbs[:, 0:1] += x
                prediction_bbs[:, 2:3] += x
                prediction_bbs[:, 1:2] += y
                prediction_bbs[:, 3:4] += y
                full_image_det_bbs = torch.cat((full_image_det_bbs, prediction_bbs))
                full_image_det_scores = torch.cat((full_image_det_scores, prediction_scores))

        # Removing bbs outside image and clipping
        full_image_filtered_det_bbs = torch.empty(0, 4, dtype=torch.float32)
        full_image_filtered_det_scores = torch.empty(0, dtype=torch.float32)
        l = torch.tensor([[0.0, 0.0, 0.0, 0.0]])      # Setting the lower and upper bound per column
        u = torch.tensor([[image_hw[1], image_hw[0], image_hw[1], image_hw[0]]])
        for bb, score in zip(full_image_det_bbs, full_image_det_scores):
            bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
            x_c, y_c = int(bb[0] + (bb_w / 2)), int(bb[1] + (bb_h / 2))
            if x_c > image_hw[1] or y_c > image_hw[0]:
                continue
            bb = torch.max(torch.min(bb, u), l)
            full_image_filtered_det_bbs = torch.cat((full_image_filtered_det_bbs, bb))
            full_image_filtered_det_scores = torch.cat((full_image_filtered_det_scores, torch.Tensor([score.item()])))

        # Performing filtering of the bbs in the overlapped areas using nms
        in_overlap_areas_indices = []
        in_overlap_areas_det_bbs, full_image_final_det_bbs = \
            torch.empty(0, 4, dtype=torch.float32), torch.empty(0, 4, dtype=torch.float32)
        in_overlap_areas_det_scores, full_image_final_det_scores = \
            torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
        for i, (det_bb, det_score) in enumerate(zip(full_image_filtered_det_bbs, full_image_filtered_det_scores)):
            bb_w, bb_h = det_bb[2] - det_bb[0], det_bb[3] - det_bb[1]
            x_c, y_c = int(det_bb[0] + (bb_w / 2)), int(det_bb[1] + (bb_h / 2))
            if normalization_map[y_c, x_c] != 1.0:
                in_overlap_areas_indices.append(i)
                in_overlap_areas_det_bbs = torch.cat((in_overlap_areas_det_bbs, torch.Tensor([det_bb.cpu().numpy()])))
                in_overlap_areas_det_scores = torch.cat((in_overlap_areas_det_scores, torch.Tensor([det_score.item()])))
            else:
                full_image_final_det_bbs = torch.cat((full_image_final_det_bbs, torch.Tensor([det_bb.cpu().numpy()])))
                full_image_final_det_scores = torch.cat((full_image_final_det_scores, torch.Tensor([det_score.item()])))

        # non-maximum suppression
        if in_overlap_areas_indices:
            keep_in_overlap_areas_indices = box_ops.nms(
                in_overlap_areas_det_bbs,
                in_overlap_areas_det_scores,
                iou_threshold=cfg.model.params.nms
            )

        if in_overlap_areas_indices:
            for i in keep_in_overlap_areas_indices:
                full_image_final_det_bbs = torch.cat((full_image_final_det_bbs, torch.Tensor([in_overlap_areas_det_bbs[i].cpu().numpy()])))
                full_image_final_det_scores = torch.cat((full_image_final_det_scores, torch.Tensor([in_overlap_areas_det_scores[i].item()])))

        # Cleaning
        del full_image_det_bbs
        del full_image_det_scores
        del full_image_filtered_det_bbs
        del full_image_filtered_det_scores
        del in_overlap_areas_det_bbs
        del in_overlap_areas_det_scores
        del normalization_map

        # compute metrics
        progress.set_description('EVAL (metrics)')

        # threshold-dependent metrics
        thrs = torch.linspace(0, 1, 21).tolist() + [2, ]

        image_thr_metrics = []
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        full_image_final_det_bbs = full_image_final_det_bbs.cpu().tolist()
        full_image_final_det_scores = full_image_final_det_scores.cpu().tolist()
        groundtruth = dataloader.dataset.annot.loc[image_id]
        full_image_target_points = groundtruth.values.tolist()
        half_bb_side = dataloader.dataset.gt_params['side'] / 2
        full_image_target_bbs = \
            [[center[0] - half_bb_side, center[1] - half_bb_side, center[0] + half_bb_side, center[1] + half_bb_side]
             for center in full_image_target_points]
        if cfg.train.debug and epoch % cfg.train.debug == 0:
            save_img_with_bbs(full_image, image_id, full_image_final_det_bbs, full_image_target_bbs)
        for thr in progress_thrs:
            full_image_final_det_bbs_thr = [bb for bb, score in
                                            zip(full_image_final_det_bbs, full_image_final_det_scores) if
                                            score > thr]
            full_image_final_det_scores_thr = [score for score in full_image_final_det_scores if score > thr]

            # segmentation metrics
            progress_thrs.set_description(f'thr={thr:.2f} (segm)')
            dice, jaccard = dice_jaccard(
                full_image_target_bbs, full_image_final_det_bbs_thr, full_image_final_det_scores_thr, image_hw, thr=thr)

            segm_metrics = {
                'segm/dice': dice.item(),
                'segm/jaccard': jaccard.item()
            }

            # counting metrics
            localizations = [[bb[1] + ((bb[3]-bb[1])/2), bb[0] + ((bb[2]-bb[0])/2)] for bb in full_image_final_det_bbs_thr]

            localizations = pd.DataFrame(localizations, columns=['Y', 'X'])
            localizations['score'] = full_image_final_det_scores_thr

            tolerance = 1.25 * (cfg.dataset.validation.params.gt_params.side / 2)  # min distance to match points
            groundtruth_and_predictions = points.match(groundtruth, localizations, tolerance)
            count_pdet_metrics = points.compute_metrics(groundtruth_and_predictions, image_hw=image_hw)

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
    img_names = [img_name for img_name in os.listdir(cfg.dataset.train.params.root) if img_name.endswith("cell.png")]
    num_train_sample = cfg.dataset.train.params.num_sample
    num_val_sample = cfg.dataset.validation.params.num_sample
    # the dataset contains 200 images; 100 should be for test
    if num_train_sample + num_val_sample > 100:
        log.error(f"Splits train+val can contain a maximum of 100 images")
    indexes = list(range(0, len(img_names)))
    random.shuffle(indexes)
    train_indexes = indexes[0:num_train_sample]
    train_img_names = [img_names[i] for i in train_indexes]
    val_indexes = indexes[num_train_sample:num_train_sample+num_val_sample]
    val_img_names = [img_names[i] for i in val_indexes]
    test_img_names = list(set(img_names) - set(train_img_names + val_img_names))
    df = pd.DataFrame({"name": test_img_names})  # saving indexes for testing
    df.to_csv('test_img_names.csv')

    return train_img_names, val_img_names


@hydra.main(config_path="conf/detection_based", config_name="config")
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
    params = cfg.dataset.train.params
    log.info("Train input size: {0}x{0}".format(params.patch_size))
    if cfg.dataset.train.name == "VGGCellsDataset":
        train_img_names, val_img_names = compute_dataset_splits(cfg)
    train_transform = Compose([RandomHorizontalFlip(), ToTensor()])
    train_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.train.name}.{cfg.dataset.train.name}")
    train_dataset = train_dataset(
        transforms=train_transform,
        image_names=train_img_names,
        **params
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=train_dataset.custom_collate_fn,
    )
    log.info(f"Found {len(train_dataset)} samples in training dataset")

    # create validation dataset and dataloader
    log.info(f"Loading validation data")
    params = cfg.dataset.validation.params
    log.info("Validation input size: {0}x{0}".format(params.patch_size))
    valid_batch_size = cfg.train.val_batch_size if cfg.train.val_batch_size else cfg.train.batch_size
    valid_transform = ToTensor()
    valid_dataset = hydra.utils.get_class(f"datasets.{cfg.dataset.validation.name}.{cfg.dataset.validation.name}")
    valid_dataset = valid_dataset(
        transforms=valid_transform,
        image_names=val_img_names,
        **params,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=valid_dataset.custom_collate_fn,
    )
    log.info(f"Found {len(valid_dataset)} samples in validation dataset")

    # Creating model
    log.info(f"Creating model")
    load_custom_model = False
    if cfg.model.resume or cfg.model.pretrained:
        load_custom_model = True
    model = get_model_detection(num_classes=1, cfg=cfg, load_custom_model=load_custom_model)

    # Putting model to device
    model.to(device)

    # Constructing an optimizer
    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optimizer.name}")
    optimizer = optimizer(filter(lambda p: p.requires_grad,
                                 model.parameters()), **cfg.optimizer.params)
    scheduler = None
    if cfg.optimizer.scheduler is not None:
        scheduler = hydra.utils.get_class(
            f"torch.optim.lr_scheduler.{cfg.optimizer.scheduler.name}")
        scheduler = scheduler(
            **
            {**{"optimizer": optimizer},
             **cfg.optimizer.scheduler.params})

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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg, writer, epoch)
        scheduler.step()    # update lr scheduler

        train_metrics['epoch'] = epoch
        train_log = train_log.append(train_metrics, ignore_index=True)
        train_log.to_csv(train_log_path, index=False)

        # validation
        if (epoch + 1) % cfg.train.val_freq == 0:
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
