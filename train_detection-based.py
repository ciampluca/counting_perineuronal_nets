# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import sys
import math
import copy
from PIL import ImageDraw
import tqdm
import timeit

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.detection.rpn import AnchorGenerator

from datasets.perineural_nets_bbox_dataset import PerineuralNetsBBoxDataset
import utils.misc as utils
from utils.misc import random_seed, get_bbox_transforms, save_checkpoint, check_empty_images, coco_evaluate, compute_map, compute_dice_and_jaccard
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn
from utils.transforms_bbs import CropToFixedSize


def get_model_detection(num_classes, cfg, load_custom_model=False):
    assert cfg['backbone'] == "resnet50" or cfg['backbone'] == "resnet101" or cfg['backbone'] == "resnet152", \
        "Backbone not supported"

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes += 1    # num classes + background

    if load_custom_model:
        model_pretrained = False
        backbone_pretrained = False
    else:
        model_pretrained = cfg['coco_model_pretrained']
        backbone_pretrained = cfg['backbone_pretrained']

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
    if cfg['backbone'] == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg['backbone'] == "resnet101":
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg['backbone'] == "resnet152":
        print("Model with ResNet152 to be implemented")
        exit(1)
    else:
        print("Not supported backbone")
        exit(1)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, tensorboard_writer, train_cfg):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for train_iteration, (images, targets) in enumerate(metric_logger.log_every(data_loader, train_cfg['print_freq'], header)):
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
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if (train_iteration % train_cfg['log_loss'] == 0):
            tensorboard_writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + train_iteration)
            tensorboard_writer.add_scalar('Training/Reduced Sum Losses', losses_reduced, epoch * len(data_loader) + train_iteration)
            tensorboard_writer.add_scalars('Training/All Losses', loss_dict, epoch * len(data_loader) + train_iteration)


@torch.no_grad()
def validate(model, val_dataloader, device, train_cfg, data_cfg, model_cfg, tensorboard_writer, epoch):
    # Validation
    model.eval()
    print("Validation")
    start = timeit.default_timer()

    epoch_mae, epoch_mse = 0.0, 0.0
    epoch_dets_for_coco_eval = []
    epoch_dets_for_map_eval = {}

    for images, targets in tqdm.tqdm(val_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Image is divided in patches
        img_output_patches_with_bbs = []
        for image, target in zip(images, targets):
            img_gt_num = len(target['boxes'])
            img_id = target['image_id'].item()
            img_name = val_dataloader.dataset.image_files[img_id]
            img_w, img_h = image.shape[2], image.shape[1]
            counter_patches = 0
            img_det_num = 0
            img_det_bbs, img_det_scores, img_det_labels = [], [], []

            for i in range(0, img_h, data_cfg['crop_height']):
                for j in range(0, img_w, data_cfg['crop_width']):
                    counter_patches += 1
                    image_patch, target_patch = CropToFixedSize()(
                        image,
                        x_min=j,
                        y_min=i,
                        x_max=j + data_cfg['crop_width'],
                        y_max=i + data_cfg['crop_height'],
                        min_visibility=data_cfg['bbox_discard_min_vis'],
                        target=copy.deepcopy(target),
                    )

                    det_outputs = model([image_patch.to(device)])

                    bbs, scores, labels = det_outputs[0]['boxes'].data.cpu().tolist(), \
                                          det_outputs[0]['scores'].data.cpu().tolist(), \
                                          det_outputs[0]['labels'].data.cpu().tolist()

                    for det_bb, det_score in zip(bbs, scores):
                        if float(det_score) > model_cfg['det_thresh_for_counting']:
                            img_det_num += 1

                    img_det_bbs.extend([[bb[0] + j, bb[1] + i, bb[2] + j, bb[3] + i] for bb in bbs])
                    img_det_labels.extend(labels)
                    img_det_scores.extend(scores)

                    if train_cfg['debug'] and epoch % train_cfg['debug_freq'] == 0:
                        # Drawing bbs on the patch and store it
                        pil_image = to_pil_image(image_patch)
                        draw = ImageDraw.Draw(pil_image)
                        for gt_bb in target_patch['boxes']:  # gt bbs
                            draw.rectangle(
                                [gt_bb[0].cpu().item(), gt_bb[1].cpu().item(), gt_bb[2].cpu().item(),
                                 gt_bb[3].cpu().item()],
                                outline='red',
                                width=3,
                            )
                        for det_bb, det_score in zip(bbs, scores):
                            if float(det_score) > model_cfg['det_thresh_for_counting']:
                                draw.rectangle(
                                    [det_bb[0], det_bb[1], det_bb[2], det_bb[3]],
                                    outline='green',
                                    width=3,
                                )
                        img_output_patches_with_bbs.append(to_tensor(pil_image))

            if train_cfg['debug'] and epoch % train_cfg['debug_freq'] == 0:
                debug_dir = os.path.join(tensorboard_writer.get_logdir(), 'output_debug')
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                img_output_patches_with_bbs = torch.stack(img_output_patches_with_bbs)
                rec_image_with_bbs = img_output_patches_with_bbs.view(
                    int(img_h / data_cfg['crop_height']),
                    int(img_w / data_cfg['crop_width']),
                    *img_output_patches_with_bbs.size()[-3:]
                )
                permuted_rec_image_with_bbs = rec_image_with_bbs.permute(2, 0, 3, 1, 4).contiguous()
                permuted_rec_image_with_bbs = permuted_rec_image_with_bbs.view(
                     permuted_rec_image_with_bbs.shape[0],
                     permuted_rec_image_with_bbs.shape[1] *
                     permuted_rec_image_with_bbs.shape[2],
                     permuted_rec_image_with_bbs.shape[3] *
                     permuted_rec_image_with_bbs.shape[4]
                )
                to_pil_image(permuted_rec_image_with_bbs).save(
                    os.path.join(debug_dir, "reconstructed_{}_with_bbs_epoch_{}.png".format(img_name.rsplit(".", 1)[0], epoch)))

            # Updating errors
            img_mae = abs(img_det_num - img_gt_num)
            img_mse = (img_det_num - img_gt_num) ** 2
            epoch_mae += img_mae
            epoch_mse += img_mse

            # Updating list of dets for coco eval
            epoch_dets_for_coco_eval.append({
                'boxes': torch.as_tensor(img_det_bbs, dtype=torch.float32),
                'scores': torch.as_tensor(img_det_scores, dtype=torch.float32),
                'labels': torch.as_tensor(img_det_labels, dtype=torch.int64),
            })

            # Updating dets for map evaluation
            epoch_dets_for_map_eval[img_id] = {
                'pred_bbs': img_det_bbs,
                'scores': img_det_scores,
                'labels': img_det_labels,
                'gt_bbs': target['boxes'].data.cpu().tolist(),
                'img_dim': (img_w, img_h),
            }

    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Validation ended. Total running time: %d:%d:%d.\n" % (hours, mins, secs))
    print("Starting eval metrics computation...")

    # Computing mean of the errors
    epoch_mae /= len(val_dataloader.dataset)
    epoch_mse /= len(val_dataloader.dataset)

    # Computing COCO mAP
    coco_det_map = coco_evaluate(val_dataloader, epoch_dets_for_coco_eval, max_dets=train_cfg['coco_max_dets'], folder_to_save=tensorboard_writer.get_logdir())

    # Computing map
    det_map = compute_map(epoch_dets_for_map_eval)

    # Computing dice score and jaccard index
    dice_score, jaccard_index = compute_dice_and_jaccard(epoch_dets_for_map_eval)

    return epoch_mae, epoch_mse, coco_det_map, det_map, dice_score, jaccard_index


def main(args):
    print(args)

    # Opening YAML cfg config file
    with open(args.cfg_file, 'r') as stream:
        try:
            cfg_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Retrieving cfg
    train_cfg = cfg_file['training']
    model_cfg = cfg_file['model']
    data_cfg = cfg_file['dataset']

    # Setting device
    device = torch.device(model_cfg['device'])

    # Setting cache directory
    torch.hub.set_dir(model_cfg['cache_folder'])

    # No possible to set checkpoint and pre-trained model at the same time
    if train_cfg['checkpoint'] and train_cfg['pretrained_model']:
        print("You can't set checkpoint and pretrained-model at the same time")
        exit(1)

    # Reproducibility
    seed = train_cfg['seed']
    if device == "cuda":
        random_seed(seed, True)
    elif device == "cpu":
        random_seed(seed, False)

    # Creating tensorboard writer
    if train_cfg['checkpoint']:
        checkpoint = torch.load(train_cfg['checkpoint'])
        writer = SummaryWriter(log_dir=checkpoint['tensorboard_working_dir'])
    else:
        writer = SummaryWriter(comment="_" + train_cfg['tensorboard_filename'])

    # Saving cfg file in the same folder
    copyfile(args.cfg_file, os.path.join(writer.get_logdir(), os.path.basename(args.cfg_file)))

    #####################################
    # Creating datasets and dataloaders
    #####################################
    data_root = data_cfg['root']
    dataset_name = data_cfg['name']
    crop_width, crop_height = data_cfg['crop_width'], data_cfg['crop_height']
    assert crop_width == crop_height, "Crops must be squares"
    list_frames = data_cfg['all_frames']
    list_train_frames, list_val_frames = data_cfg['train_frames'], data_cfg['val_frames']
    if data_cfg['specular_split']:
        list_train_frames = list_val_frames = list_frames

    ################################
    # Creating training datasets and dataloader
    print("Loading training data")

    train_dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        transforms=get_bbox_transforms(train=True, crop_width=crop_width, crop_height=crop_height, min_visibility=data_cfg['bbox_discard_min_vis']),
        list_frames=list_train_frames,
        load_in_memory=data_cfg['load_in_memory'],
        min_visibility=data_cfg['bbox_discard_min_vis'],
        with_patches=data_cfg['train_with_precomputed_patches'],
        percentage=data_cfg['percentage'],
        specular_split=data_cfg['specular_split'],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        collate_fn=train_dataset.standard_collate_fn,
    )

    ################################
    # Creating training datasets and dataloaders
    print("Loading validation data")

    val_dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        transforms=get_bbox_transforms(train=False, resize_factor=crop_width),
        list_frames=list_val_frames,
        load_in_memory=False,
        min_visibility=data_cfg['bbox_discard_min_vis'],
        with_patches=False,
        specular_split=data_cfg['specular_split'],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=val_dataset.standard_collate_fn,
    )

    # Initializing validation metrics
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    best_validation_map, best_validation_coco_map, best_validation_dice, best_validation_jaccard = 0.0, 0.0, 0.0, 0.0
    min_mae_epoch, min_mse_epoch, best_map_epoch, best_coco_map_epoch, best_dice_epoch, best_jaccard_epoch = -1, -1, -1, -1, -1, -1

    #######################
    # Creating model
    #######################
    print("Creating model")
    load_custom_model = False
    if train_cfg['checkpoint'] or train_cfg['pretrained_model']:
        load_custom_model = True
    model = get_model_detection(num_classes=1, cfg=model_cfg, load_custom_model=load_custom_model)

    # Putting model to device
    model.to(device)

    #######################################
    # Defining optimizer and LR scheduler
    #######################################
    ##########################
    # Constructing an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=train_cfg['lr'],
                                momentum=train_cfg['momentum'],
                                weight_decay=train_cfg['weight_decay'],
                                )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg['lr_step_size'],
        gamma=train_cfg['lr_gamma']
    )

    #############################
    # Resuming a model
    #############################
    start_epoch = 0
    # Eventually resuming a pre-trained model
    if train_cfg['pretrained_model']:
        print("Resuming pre-trained model")
        if train_cfg['pretrained_model'].startswith('http://') or train_cfg['pretrained_model'].startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                train_cfg['pretrained_model'], map_location='cpu', model_dir=model_cfg["cache_folder"])
        else:
            pre_trained_model = torch.load(train_cfg['pretrained_model'], map_location='cpu')
        model.load_state_dict(pre_trained_model['model'])

    # Eventually resuming from a saved checkpoint
    if train_cfg['checkpoint']:
        print("Resuming from a checkpoint")
        checkpoint = torch.load(train_cfg['checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_validation_mae = checkpoint['best_mae']
        best_validation_mse = checkpoint['best_mse']
        best_validation_map = checkpoint['best_map']
        best_validation_coco_map = checkpoint['best_coco_map']
        best_validation_dice = checkpoint['best_dice']
        best_validation_jaccard = checkpoint['best_jaccard']
        min_mae_epoch = checkpoint['min_mae_epoch']
        min_mse_epoch = checkpoint['min_mse_epoch']
        best_map_epoch = checkpoint['best_map_epoch']
        best_coco_map_epoch = checkpoint['best_coco_map_epoch']
        best_dice_epoch = checkpoint['best_dice_epoch']
        best_jaccard_epoch = checkpoint['best_jaccard_epoch']

    ################
    ################
    # Training
    print("Start training")
    for epoch in range(start_epoch, train_cfg['epochs']):
        # Training for one epoch
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, writer, train_cfg)

        # Updating lr scheduler
        lr_scheduler.step()

        # Validating
        if (epoch % train_cfg['val_freq'] == 0):
            epoch_mae, epoch_mse, epoch_coco_evaluator, epoch_det_map, epoch_dice, epoch_jaccard = \
                validate(model, val_dataloader, device, train_cfg, data_cfg, model_cfg, writer, epoch)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(dataset_name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(dataset_name), epoch_mse, epoch)
            epoch_coco_map_05 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            writer.add_scalar('Validation on {}/COCO mAP'.format(dataset_name), epoch_coco_map_05, epoch)
            writer.add_scalar('Validation on {}/Det mAP'.format(dataset_name), epoch_det_map, epoch)
            writer.add_scalar('Validation on {}/Dice Score'.format(dataset_name), epoch_dice, epoch)
            writer.add_scalar('Validation on {}/Jaccard Index'.format(dataset_name), epoch_jaccard, epoch)

            # Eventually saving best models
            if epoch_mae < best_validation_mae:
                best_validation_mae = epoch_mae
                min_mae_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mae': epoch_mae,
                }, writer.get_logdir(), best_model=dataset_name + "_mae")
            if epoch_mse < best_validation_mse:
                best_validation_mse = epoch_mse
                min_mse_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mse': epoch_mse,
                }, writer.get_logdir(), best_model=dataset_name + "_mse")
            if epoch_det_map >= best_validation_map:
                best_validation_map = epoch_det_map
                best_map_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_map': epoch_det_map,
                }, writer.get_logdir(), best_model=dataset_name + "_map")
            if epoch_coco_map_05 >= best_validation_coco_map:
                best_validation_coco_map = epoch_coco_map_05
                best_coco_map_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_coco_map': epoch_coco_map_05,
                }, writer.get_logdir(), best_model=dataset_name + "_coco_map_05")
            if epoch_dice >= best_validation_dice:
                best_validation_dice = epoch_dice
                best_dice_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': epoch_dice,
                }, writer.get_logdir(), best_model=dataset_name + "_dice")
            if epoch_jaccard >= best_validation_jaccard:
                best_validation_jaccard = epoch_jaccard
                best_jaccard_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_jaccard': epoch_jaccard,
                }, writer.get_logdir(), best_model=dataset_name + "_jaccard")

            print('Epoch: ', epoch, ' Dataset: ', dataset_name,
                  ' MAE: ', epoch_mae, ' MSE ', epoch_mse, ' mAP ', epoch_det_map, ' COCO mAP 0.5 ', epoch_coco_map_05,
                  ' Dice Score: ', epoch_dice, ' Jaccard Coefficient: ', epoch_jaccard,
                  ' Min MAE: ', best_validation_mae, ' Min MAE Epoch: ', min_mae_epoch,
                  ' Min MSE: ', best_validation_mse, ' Min MSE Epoch ', min_mse_epoch,
                  ' Best mAP: ', best_validation_map, ' Best mAP Epoch: ', best_map_epoch,
                  ' Best COCO mAP: ', best_validation_coco_map, ' Best COCO mAP Epoch: ', best_coco_map_epoch,
                  ' Best Dice: ', best_validation_dice, ' Best Dice Epoch: ', best_dice_epoch,
                  ' Best Jaccard: ', best_validation_jaccard, ' Best Jaccard Epoch: ', best_jaccard_epoch
                  )

            # Saving last model
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_validation_mae,
                'best_mse': best_validation_mse,
                'best_map': best_validation_map,
                'best_coco_map': best_validation_coco_map,
                'best_dice': best_validation_dice,
                'best_jaccard': best_validation_jaccard,
                'min_mae_epoch': min_mae_epoch,
                'best_map_epoch': best_map_epoch,
                'best_coco_map_epoch': best_coco_map_epoch,
                'best_dice_epoch': best_dice_epoch,
                'best_jaccard_epoch': best_jaccard_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, writer.get_logdir())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)
