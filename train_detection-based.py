# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import sys
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from datasets.perineural_nets_bbox_dataset import PerineuralNetsBBoxDataset
import utils.misc as utils
from utils.misc import random_seed, get_bbox_transforms
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

    # Creating model
    if cfg['backbone'] == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
        )
    elif cfg['backbone'] == "resnet101":
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
        )
    elif cfg['backbone'] == "resnet152":
        print("Model with ResNet152 to be implemented")
        exit(1)
    else:
        print("Not supported backbone")
        exit(1)

    detection_model = model
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = model.backbone

    return detection_model, backbone


def check_empty_images(targets):
    for target in targets:
        if target['boxes'].nelement() == 0:
            target['boxes'] = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32, device=target['boxes'].get_device())
            target['area'] = torch.as_tensor([1], dtype=torch.float32, device=target['boxes'].get_device())
            target['labels'] = torch.zeros((1,), dtype=torch.int64, device=target['boxes'].get_device())
            target['iscrowd'] = torch.zeros((1,), dtype=torch.int64, device=target['boxes'].get_device())

    return targets


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
    crop_width, crop_height = train_cfg['crop_width'], train_cfg['crop_height']

    ################################
    # Creating training datasets and dataloader
    print("Loading training data")

    train_dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        transforms=get_bbox_transforms(train=True, crop_width=crop_width, crop_height=crop_height),
        list_frames=data_cfg['train_frames']
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
        transforms=get_bbox_transforms(train=False, crop_width=crop_width, crop_height=crop_height),
        list_frames=data_cfg['val_frames']
    )

    val_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=val_dataset.standard_collate_fn,
    )

    # Initializing best validation ap value
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    best_validation_are = float(sys.maxsize)
    min_mae_epoch = -1

    #######################
    # Creating model
    #######################
    print("Creating model")
    load_custom_model = False
    if train_cfg['checkpoint'] or train_cfg['pretrained_model']:
        load_custom_model = True
    model, backbone = get_model_detection(num_classes=1, cfg=model_cfg, load_custom_model=load_custom_model)

    # Putting model to device and setting train mode
    model.to(device)
    model.train()

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
        best_validation_are = checkpoint['best_are']
        min_mae_epoch = checkpoint['min_mae_epoch']

    ################
    ################
    # Training
    print("Start training")
    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(train_dataloader, print_freq=train_cfg['print_freq'], header=header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In case of empty images (i.e, without bbs), we handle them as negative images
            # (i.e., images with only background and no object), creating a fake object that represent the backgound
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
                for target in targets:
                    image_id = target['image_id'].item()
                    print(train_dataset.imgs[image_id])
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            # clip norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if epoch % train_cfg['log_loss'] == 0:
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar('Training/Reduced Sum Losses', losses_reduced, epoch)
                writer.add_scalars('Training/All Losses', loss_dict, epoch)

            if (epoch % train_cfg['save_freq'] == 0 and epoch != 0):
                # Validation
                model.eval()
                with torch.no_grad():
                    print("Validation")
                    for images, targets in val_dataloader:
                        images = list(image.to(device) for image in images)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                        for image, target in zip(images, targets):
                        # Image is divided in patches
                            img_w, img_h = image.shape[2], image.shape[1]
                            for i in range(0, img_h, crop_height):
                                for j in range(0, img_w, crop_width):
                                    image, target = CropToFixedSize()(
                                        image,
                                        x_min=j,
                                        y_min=i,
                                        x_max=j + crop_width,
                                        y_max=i + crop_height,
                                        target=target
                                    )

            # Updating lr scheduler
            lr_scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)
