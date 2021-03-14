# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import sys
import math
import copy
from PIL import ImageDraw

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.detection.rpn import AnchorGenerator

from datasets.perineural_nets_bbox_dataset import PerineuralNetsBBoxDataset
import utils.misc as utils
from utils.misc import random_seed, get_bbox_transforms, save_checkpoint, check_empty_images
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

    train_iteration = 0
    for images, targets in metric_logger.log_every(data_loader, train_cfg['print_freq'], header):
        train_iteration += 1
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
            tensorboard_writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]["lr"], (epoch+1)*train_iteration)
            tensorboard_writer.add_scalar('Training/Reduced Sum Losses', losses_reduced, (epoch+1)*train_iteration)
            tensorboard_writer.add_scalars('Training/All Losses', loss_dict, (epoch+1)*train_iteration)


def validate(model, val_dataloader, device, train_cfg, data_cfg, tensorboard_writer, epoch):
    # Validation
    model.eval()
    with torch.no_grad():
        print("Validation")
        epoch_mae, epoch_mse = 0.0, 0.0
        for images, targets in tqdm.tqdm(val_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Image is divided in patches
            img_output_patches_with_bbs = []
            for image, target in zip(images, targets):
                gt_num = len(target['boxes'])
                img_id = target['image_id'].item()
                img_name = val_dataloader.dataset.image_files[img_id]
                img_w, img_h = image.shape[2], image.shape[1]
                counter_patches = 0
                det_num = 0
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

                        det_outputs = model(image.unsqueeze(dim=0).to(device))

                        det_outputs = [{k: v for k, v in t.items()} for t in det_outputs]

                        bbs, scores, labels = det_outputs[0]['boxes'].data.cpu().numpy(), \
                                              det_outputs[0]['scores'].data.cpu().numpy(), \
                                              det_outputs[0]['labels'].data.cpu().numpy()

                        det_num += len(bbs)

                        if train_cfg['debug']:
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
                            for det_bb in bbs:
                                draw.rectangle(
                                    [det_bb[0].item(), det_bb[1].item(), det_bb[2].item(), det_bb[3].item()],
                                    outline='green',
                                    width=3,
                                )
                            img_output_patches_with_bbs.append(to_tensor(pil_image))

                if train_cfg['debug']:
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
                img_mae = abs(det_num - gt_num)
                img_mse = (det_num - gt_num) ** 2
                epoch_mae += img_mae
                epoch_mse += img_mse

            # Computing mean of the errors
            epoch_mae /= len(val_dataloader.dataset)
            epoch_mse /= len(val_dataloader.dataset)

        return epoch_mae, epoch_mse


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

    ################################
    # Creating training datasets and dataloader
    print("Loading training data")

    train_dataset = PerineuralNetsBBoxDataset(
        data_root=data_root,
        transforms=get_bbox_transforms(train=True, crop_width=crop_width, crop_height=crop_height, min_visibility=data_cfg['bbox_discard_min_vis']),
        list_frames=data_cfg['train_frames'],
        load_in_memory=data_cfg['load_in_memory'],
        min_visibility=data_cfg['bbox_discard_min_vis'],
        with_patches=data_cfg['train_with_precomputed_patches'],
        percentage=data_cfg['percentage']
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
        transforms=get_bbox_transforms(train=False, resize_factor=crop_width),
        list_frames=data_cfg['val_frames'],
        load_in_memory=False,
        min_visibility=data_cfg['bbox_discard_min_vis'],
        with_patches=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=val_dataset.standard_collate_fn,
    )

    # Initializing best validation ap value
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    min_mae_epoch = -1

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
        min_mae_epoch = checkpoint['min_mae_epoch']

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
            epoch_mae, epoch_mse = validate(model, val_dataloader, device, train_cfg, data_cfg, writer, epoch)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(dataset_name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(dataset_name), epoch_mse, epoch)

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
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_mse': epoch_mse,
                }, writer.get_logdir(), best_model=dataset_name + "_mse")

            print('Epoch: ', epoch, ' Dataset: ', dataset_name, ' MAE: ', epoch_mae,
                  ' Min MAE : ', best_validation_mae, ' Min MAE Epoch: ', min_mae_epoch,
                  ' MSE: ', epoch_mse)

            # Saving last model
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_validation_mae,
                'best_mse': best_validation_mse,
                'min_mae_epoch': min_mae_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, writer.get_logdir())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)
