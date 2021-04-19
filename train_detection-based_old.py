# -*- coding: utf-8 -*-
import os
import sys
import math
import copy
from PIL import ImageDraw, ImageFont
import tqdm
import timeit
import numpy as np
import logging
from omegaconf import DictConfig
import hydra

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops

from datasets.perineural_nets_bbox_dataset import PerineuralNetsBBoxDataset
import utils.misc as utils
from utils.misc import update_dict, random_seed, get_bbox_transforms, save_checkpoint, check_empty_images, coco_evaluate, compute_map, compute_dice_and_jaccard
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn
from utils.transforms_bbs import CropToFixedSize, PadToSize

# Creating logger
log = logging.getLogger("Counting Nets")


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
            box_detections_per_img=cfg.model.max_dets_per_image,
            box_nms_thresh=cfg.model.nms,
            box_score_thresh=cfg.model.det_thresh,
            model_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.model.cache_folder),
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg.model.backbone == "resnet101":
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg.model.max_dets_per_image,
            box_nms_thresh=cfg.model.nms,
            box_score_thresh=cfg.model.det_thresh,
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, cfg, tensorboard_writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for train_iteration, (images, targets) in enumerate(metric_logger.log_every(data_loader, cfg.training.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In case of empty images (i.e, without bbs), we handle them as negative images
        # (i.e., images with only background and no object), creating a fake object that represent the background
        # class and does not affect training
        # https://discuss.pytorch.org/t/torchvision-faster-rcnn-empty-training-images/46935/12
        targets = check_empty_images(targets)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            log.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if train_iteration % cfg.training.log_loss == 0:
            tensorboard_writer.add_scalar('Training/Detection-based Learning Rate', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + train_iteration)
            tensorboard_writer.add_scalar('Training/Detection-based Reduced Sum Losses', losses_reduced, epoch * len(data_loader) + train_iteration)
            tensorboard_writer.add_scalars('Training/Detection-based All Losses', loss_dict, epoch * len(data_loader) + train_iteration)


@torch.no_grad()
def validate(model, val_dataloader, device, cfg, epoch):
    # Validation
    model.eval()
    log.info(f"Start validation of the epoch {epoch}")
    start = timeit.default_timer()

    epoch_mae, epoch_mse, epoch_are = 0.0, 0.0, 0.0
    epoch_dets_for_coco_eval = []
    epoch_dets_for_map_eval = {}

    crop_width, crop_height = cfg.dataset.validation.params.input_size
    stride_w = crop_width - cfg.dataset.validation.params.patches_overlap
    stride_h = crop_height - cfg.dataset.validation.params.patches_overlap

    for images, targets in tqdm.tqdm(val_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            img_gt_num = len(target['boxes'])
            img_id = target['image_id'].item()
            img_name = val_dataloader.dataset.image_files[img_id]

            # Image is divided in patches
            img_w, img_h = image.shape[2], image.shape[1]
            img_det_num = 0
            img_det_bbs, img_det_scores, img_det_labels = [], [], []

            num_h_patches, num_v_patches = math.ceil(img_w / stride_w), math.ceil(img_h / stride_h)
            img_w_padded = stride_w * num_h_patches + (crop_width - stride_w)
            img_h_padded = stride_h * num_v_patches + (crop_height - stride_h)

            padded_image, padded_target = PadToSize()(
                image=image,
                min_width=img_w_padded,
                min_height=img_h_padded,
                target=copy.deepcopy(target)
            )

            normalization_map = torch.zeros_like(padded_image)
            reconstructed_image = torch.zeros_like(padded_image)

            for i in range(0, img_h, stride_h):
                for j in range(0, img_w, stride_w):
                    image_patch, target_patch = CropToFixedSize()(
                        padded_image,
                        x_min=j,
                        y_min=i,
                        x_max=j + crop_width,
                        y_max=i + crop_height,
                        min_visibility=cfg.training.bbox_discard_min_vis,
                        target=copy.deepcopy(padded_target),
                    )

                    reconstructed_image[:, i:i + crop_height, j:j + crop_width] += image_patch
                    normalization_map[:, i:i + crop_height, j:j + crop_width] += 1.0

                    det_outputs = model([image_patch.to(device)])

                    bbs, scores, labels = det_outputs[0]['boxes'].data.cpu().tolist(), \
                                          det_outputs[0]['scores'].data.cpu().tolist(), \
                                          det_outputs[0]['labels'].data.cpu().tolist()

                    img_det_bbs.extend([[bb[0] + j, bb[1] + i, bb[2] + j, bb[3] + i] for bb in bbs])
                    img_det_labels.extend(labels)
                    img_det_scores.extend(scores)

            reconstructed_image /= normalization_map
            overlapping_map = np.where(normalization_map[0].cpu().numpy() != 1.0, 1, 0)

            # Performing filtering of the bbs in the overlapped areas using nms
            keep_overlap = []
            for i, det_bb in enumerate(img_det_bbs):
                bb_w, bb_h = det_bb[2] - det_bb[0], det_bb[3] - det_bb[1]
                x_c, y_c = int(det_bb[0] + (bb_w/2)), int(det_bb[1] + (bb_h/2))
                if overlapping_map[y_c, x_c] == 1:
                    keep_overlap.append(i)

            bbs_in_overlapped_areas = [img_det_bbs[i] for i in keep_overlap]
            scores_in_overlapped_areas = [img_det_scores[i] for i in keep_overlap]
            labels_in_overlapped_areas = [img_det_labels[i] for i in keep_overlap]
            final_bbs = [bb for i, bb in enumerate(img_det_bbs) if i not in keep_overlap]
            final_scores = [score for i, score in enumerate(img_det_scores) if i not in keep_overlap]
            final_labels = [label for i, label in enumerate(img_det_labels) if i not in keep_overlap]

            # non-maximum suppression
            if keep_overlap:
                keep_overlap = box_ops.nms(
                    torch.as_tensor(bbs_in_overlapped_areas, dtype=torch.float32),
                    torch.as_tensor(scores_in_overlapped_areas, dtype=torch.float32),
                    iou_threshold=cfg.model.nms
                )

            bbs_in_overlapped_areas = [bbs_in_overlapped_areas[i] for i in keep_overlap]
            final_bbs.extend(bbs_in_overlapped_areas)
            scores_in_overlapped_areas = [scores_in_overlapped_areas[i] for i in keep_overlap]
            final_scores.extend(scores_in_overlapped_areas)
            labels_in_overlapped_areas = [labels_in_overlapped_areas[i] for i in keep_overlap]
            final_labels.extend(labels_in_overlapped_areas)

            pad_w_to_remove, pad_h_to_remove = int((img_w_padded - img_w)) / 2, int((img_h_padded - img_h) / 2)
            final_bbs = [
                [bb[0] - pad_w_to_remove, bb[1] - pad_h_to_remove, bb[2] - pad_w_to_remove, bb[3] - pad_h_to_remove] for
                bb in final_bbs]

            for det_bb, det_score in zip(final_bbs, final_scores):
                if float(det_score) > cfg.training.det_thresh_for_counting:
                    img_det_num += 1

            if cfg.training.debug and epoch % cfg.training.debug_freq == 0:
                debug_dir = os.path.join(os.getcwd(), 'output_debug')
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                # Removing pad from image
                h_pad_top = int((img_h_padded - img_h) / 2.0)
                h_pad_bottom = img_h_padded - img_h - h_pad_top
                w_pad_left = int((img_w_padded - img_w) / 2.0)
                w_pad_right = img_w_padded - img_w - w_pad_left
                # Drawing det bbs
                reconstructed_image = reconstructed_image[:, h_pad_top:img_h_padded - h_pad_bottom,
                                      w_pad_left:img_w_padded - w_pad_right]
                # Drawing det bbs
                pil_reconstructed_image = to_pil_image(reconstructed_image)
                draw = ImageDraw.Draw(pil_reconstructed_image)
                for bb in final_bbs:
                    draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='red', width=3)
                gt_bboxes = target['boxes'].data.cpu().tolist()
                for bb in gt_bboxes:
                    draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='green', width=3)
                # Add text to image
                text = f"Det Num of Nets: {img_det_num}, GT Num of Nets: {img_gt_num}"
                font_path = os.path.join(hydra.utils.get_original_cwd(), "./font/LEMONMILK-RegularItalic.otf")
                font = ImageFont.truetype(font_path, 100)
                draw.text((75, 75), text=text, font=font, fill=(0, 191, 255))
                pil_reconstructed_image.save(
                    os.path.join(debug_dir, "reconstructed_{}_with_bbs_epoch_{}.png".format(img_name.rsplit(".", 1)[0], epoch)))

            # Updating errors
            img_mae = abs(img_det_num - img_gt_num)
            img_mse = (img_det_num - img_gt_num) ** 2
            img_are = abs(img_det_num - img_gt_num) / np.clip(img_gt_num, 1, a_max=None)
            epoch_mae += img_mae
            epoch_mse += img_mse
            epoch_are += img_are

            # Updating list of dets for coco eval
            epoch_dets_for_coco_eval.append({
                'boxes': torch.as_tensor(final_bbs, dtype=torch.float32),
                'scores': torch.as_tensor(final_scores, dtype=torch.float32),
                'labels': torch.as_tensor(final_labels, dtype=torch.int64),
            })

            # Updating dets for map evaluation
            epoch_dets_for_map_eval[img_id] = {
                'pred_bbs': final_bbs,
                'scores': final_scores,
                'labels': final_labels,
                'gt_bbs': target['boxes'].data.cpu().tolist(),
                'img_dim': (img_w, img_h),
            }

    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    log.info(f"Validation epoch {epoch} ended. Total running time: {hours}:{mins}:{secs}.")

    # Computing mean of the errors
    epoch_mae /= len(val_dataloader.dataset)
    epoch_mse /= len(val_dataloader.dataset)
    epoch_are /= len(val_dataloader.dataset)

    # Computing COCO mAP
    coco_det_map = coco_evaluate(val_dataloader, epoch_dets_for_coco_eval, max_dets=cfg.training.coco_max_dets, folder_to_save=os.getcwd())

    # Computing map
    det_map = compute_map(epoch_dets_for_map_eval)

    # Computing dice score and jaccard index
    dice_score, jaccard_index = compute_dice_and_jaccard(epoch_dets_for_map_eval)

    return epoch_mae, epoch_mse, epoch_are, coco_det_map, det_map, dice_score, jaccard_index


@hydra.main(config_path="conf/detection_based", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = copy.deepcopy(hydra_cfg.technique)
    for _, v in hydra_cfg.items():
        update_dict(cfg, v)

    experiment_name = f"{cfg.model.name}_{cfg.model.backbone}_coco_pretrained-{cfg.model.coco_model_pretrained}" \
                      f"_{cfg.dataset.training.name}_specular_split-{cfg.training.specular_split}" \
                      f"_input_size-{cfg.dataset.training.params.input_size}_nms-{cfg.model.nms}" \
                      f"_val_patches_overlap-${cfg.dataset.validation.params.patches_overlap}" \
                      f"_det_thresh_for_counting-{cfg.training.det_thresh_for_counting}_batch_size-${cfg.training.batch_size}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    torch.hub.set_dir(os.path.join(hydra.utils.get_original_cwd(), cfg.model.cache_folder))
    best_models_folder = os.path.join(os.getcwd(), 'best_models')
    if not os.path.exists(best_models_folder):
        os.makedirs(best_models_folder)

    # No possible to set checkpoint and pre-trained model at the same time
    if cfg.model.resume and cfg.model.pretrained:
        log.error(f"You can't set checkpoint and pretrained-model at the same time")
        exit(1)

    # Reproducibility
    seed = cfg.seed
    if device.type == "cuda":
        random_seed(seed, True)
    elif device.type == "cpu":
        random_seed(seed, False)

    # Creating tensorboard writer
    writer = SummaryWriter(comment="_" + experiment_name)

    # Creating training dataset and dataloader
    log.info(f"Loading training data")
    training_crop_width, training_crop_height = cfg.dataset.training.params.input_size
    if training_crop_width != training_crop_height:
        logging.error(f"Crops must be squares")
        exit(1)
    list_frames = cfg.dataset.training.params.all_frames
    list_train_frames = cfg.dataset.training.params.train_frames
    if cfg.training.specular_split:
        list_train_frames = list_frames

    train_dataset = PerineuralNetsBBoxDataset(
        data_root=cfg.dataset.training.root,
        dataset_name=cfg.dataset.training.name,
        transforms=get_bbox_transforms(train=True, crop_width=training_crop_width, crop_height=training_crop_height, min_visibility=cfg.training.bbox_discard_min_vis),
        list_frames=list_train_frames,
        min_visibility=cfg.training.bbox_discard_min_vis,
        load_in_memory=cfg.dataset.training.params.load_in_memory,
        with_patches=cfg.training.precomputed_patches,
        specular_split=cfg.training.specular_split,
        percentage=cfg.dataset.training.params.percentage,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=train_dataset.standard_collate_fn,
    )

    log.info(f"Found {len(train_dataset)} samples in training dataset")

    # Creating validation dataset and dataloader
    log.info(f"Loading validation data")
    val_crop_width, val_crop_height = cfg.dataset.validation.params.input_size
    if val_crop_width != val_crop_height or val_crop_width % 32 != 0 or val_crop_height % 32 != 0:
        logging.error(f"Crops must be squares and in validation mode crop dim must be multiple of 32")
        exit(1)
    list_frames = cfg.dataset.validation.params.all_frames
    list_val_frames = cfg.dataset.validation.params.val_frames
    if cfg.training.specular_split:
        list_val_frames = list_frames

    val_dataset = PerineuralNetsBBoxDataset(
        data_root=cfg.dataset.validation.root,
        dataset_name=cfg.dataset.validation.name,
        transforms=get_bbox_transforms(train=False),
        list_frames=list_val_frames,
        load_in_memory=False,
        min_visibility=cfg.training.bbox_discard_min_vis,
        with_patches=False,
        specular_split=cfg.training.specular_split,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=val_dataset.standard_collate_fn,
    )

    log.info(f"Found {len(val_dataset)} samples in validation dataset")

    # Initializing validation metrics
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    best_validation_are = float(sys.maxsize)
    best_validation_map, best_validation_coco_map, best_validation_dice, best_validation_jaccard = 0.0, 0.0, 0.0, 0.0
    min_mae_epoch, min_mse_epoch, min_are_epoch, best_map_epoch, best_coco_map_epoch, best_dice_epoch, best_jaccard_epoch = -1, -1, -1, -1, -1, -1, -1

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

    start_epoch = 0
    # Eventually resuming a pre-trained model
    if cfg.model.pretrained:
        log.info(f"Resuming pre-trained model")
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location=device, model_dir=cfg.model.cache_folder)
        else:
            pre_trained_model = torch.load(cfg.model.pretrained, map_location=device)
        model.load_state_dict(pre_trained_model['model'])

    # Eventually resuming from a saved checkpoint
    if cfg.model.resume:
        log.info(f"Resuming from a checkpoint")
        checkpoint = torch.load(cfg.model.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_validation_mae = checkpoint['best_mae']
        best_validation_mse = checkpoint['best_mse']
        best_validation_are = checkpoint['best_are']
        best_validation_map = checkpoint['best_map']
        best_validation_coco_map = checkpoint['best_coco_map']
        best_validation_dice = checkpoint['best_dice']
        best_validation_jaccard = checkpoint['best_jaccard']
        min_mae_epoch = checkpoint['min_mae_epoch']
        min_mse_epoch = checkpoint['min_mse_epoch']
        min_are_epoch = checkpoint['min_are_epoch']
        best_map_epoch = checkpoint['best_map_epoch']
        best_coco_map_epoch = checkpoint['best_coco_map_epoch']
        best_dice_epoch = checkpoint['best_dice_epoch']
        best_jaccard_epoch = checkpoint['best_jaccard_epoch']

    ################
    ################
    # Training
    log.info(f"Start training")
    for epoch in range(start_epoch, cfg.training.epochs):
        # Training for one epoch
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, cfg, writer)

        # Updating lr scheduler
        scheduler.step()

        # Validating
        if epoch % cfg.training.val_freq == 0:
            epoch_mae, epoch_mse, epoch_are, epoch_coco_evaluator, epoch_det_map, epoch_dice, epoch_jaccard = \
                validate(model, val_dataloader, device, cfg, epoch)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(cfg.dataset.validation.name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(cfg.dataset.validation.name), epoch_mse, epoch)
            writer.add_scalar('Validation on {}/ARE'.format(cfg.dataset.validation.name), epoch_are, epoch)
            epoch_coco_map_05 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            writer.add_scalar('Validation on {}/COCO mAP'.format(cfg.dataset.validation.name), epoch_coco_map_05, epoch)
            writer.add_scalar('Validation on {}/Det mAP'.format(cfg.dataset.validation.name), epoch_det_map, epoch)
            writer.add_scalar('Validation on {}/Dice Score'.format(cfg.dataset.validation.name), epoch_dice, epoch)
            writer.add_scalar('Validation on {}/Jaccard Index'.format(cfg.dataset.validation.name), epoch_jaccard, epoch)

            # Eventually saving best models
            if epoch_mae < best_validation_mae:
                best_validation_mae = epoch_mae
                min_mae_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mae': epoch_mae,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_mae")
            if epoch_mse < best_validation_mse:
                best_validation_mse = epoch_mse
                min_mse_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mse': epoch_mse,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_mse")
            if epoch_are < best_validation_are:
                best_validation_are = epoch_are
                min_are_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mse': epoch_are,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_are")
            if epoch_det_map >= best_validation_map:
                best_validation_map = epoch_det_map
                best_map_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_map': epoch_det_map,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_map")
            if epoch_coco_map_05 >= best_validation_coco_map:
                best_validation_coco_map = epoch_coco_map_05
                best_coco_map_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_coco_map': epoch_coco_map_05,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_coco_map_05")
            if epoch_dice >= best_validation_dice:
                best_validation_dice = epoch_dice
                best_dice_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': epoch_dice,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_dice")
            if epoch_jaccard >= best_validation_jaccard:
                best_validation_jaccard = epoch_jaccard
                best_jaccard_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_jaccard': epoch_jaccard,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_jaccard")
            nl = '\n'
            log.info(f"Epoch: {epoch}, Dataset: {cfg.dataset.validation.name}, MAE: {epoch_mae}, MSE: {epoch_mse}, "
                     f"ARE: {epoch_are}, {nl} mAP: {epoch_det_map}, COCO mAP 0.5: {epoch_coco_map_05},"
                     f"Dice Score: {epoch_dice}, Jaccard Coefficient: {epoch_jaccard}, {nl} "
                     f"Min MAE: {best_validation_mae}, Min MAE Epoch: {min_mae_epoch}, {nl} "
                     f"Min MSE: {best_validation_mse}, Min MSE Epoch: {min_mse_epoch}, {nl} "
                     f"Min ARE: {best_validation_are}, Min ARE Epoch: {min_are_epoch}, {nl} "
                     f"Best mAP: {best_validation_map}, Best mAP Epoch: {best_map_epoch}, {nl} "
                     f"Best COCO mAP: {best_validation_coco_map}, Best COCO mAP Epoch: {best_coco_map_epoch}, {nl} "
                     f"Best Dice: {best_validation_dice}, Best Dice Epoch: {best_dice_epoch}, {nl} "
                     f"Best Jaccard: {best_validation_jaccard}, Best Jaccard Epoch: {best_jaccard_epoch}")

            # Saving last model
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_validation_mae,
                'best_mse': best_validation_mse,
                'best_are': best_validation_are,
                'best_map': best_validation_map,
                'best_coco_map': best_validation_coco_map,
                'best_dice': best_validation_dice,
                'best_jaccard': best_validation_jaccard,
                'min_mae_epoch': min_mae_epoch,
                'min_mse_epoch': min_mse_epoch,
                'min_are_epoch': min_are_epoch,
                'best_map_epoch': best_map_epoch,
                'best_coco_map_epoch': best_coco_map_epoch,
                'best_dice_epoch': best_dice_epoch,
                'best_jaccard_epoch': best_jaccard_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, os.getcwd())


if __name__ == "__main__":
    main()
