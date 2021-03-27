# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import tqdm
import sys
from PIL import Image, ImageFont, ImageDraw
import timeit
import ssim
import math
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.CSRNet import CSRNet
from models.UNet_nobias import UNet
from utils.misc import random_seed, get_dmap_transforms, save_checkpoint, normalize
from datasets.perineural_nets_dmap_dataset import PerineuralNetsDmapDataset
from utils.transforms_dmaps import PadToSize, CropToFixedSize

available_models = {
    'CSRNet': CSRNet,
    'UNet': UNet,
}


@torch.no_grad()
def validate(model, val_dataloader, device, train_cfg, data_cfg, model_cfg, epoch, tensorboard_writer):
    # Validation
    model.eval()
    print("Validation")
    start = timeit.default_timer()

    stride_w, stride_h = data_cfg['crop_width'], data_cfg['crop_height']
    if data_cfg['overlap_val_patches']:
        stride_w, stride_h = data_cfg['crop_width'] - data_cfg['overlap_val_patches'], data_cfg['crop_height'] - \
                             data_cfg['overlap_val_patches']

    epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim = 0.0, 0.0, 0.0, 0.0, 0.0
    for images, targets in tqdm.tqdm(val_dataloader):
        images = images.to(device)
        gt_dmaps = targets['dmap'].to(device)
        img_name = val_dataloader.dataset.image_files[targets['img_id']]

        # Batch size in val mode is always 1
        image = images.squeeze(dim=0)
        gt_dmap = gt_dmaps.squeeze(dim=0)

        # Image and dmap are divided in patches
        img_w, img_h = image.shape[2], image.shape[1]
        num_h_patches, num_v_patches = math.ceil(img_w / stride_w), math.ceil(img_h / stride_h)
        img_w_padded = (num_h_patches - math.floor(img_w / stride_w)) * (
                    stride_w * num_h_patches + (data_cfg['crop_width'] - stride_w))
        img_h_padded = (num_v_patches - math.floor(img_h / stride_h)) * (
                    stride_h * num_v_patches + (data_cfg['crop_height'] - stride_h))
        padded_image, padded_gt_dmap = PadToSize()(
            image=image,
            min_width=img_w_padded,
            min_height=img_h_padded,
            dmap=gt_dmap
        )

        normalization_map = torch.zeros_like(padded_image)
        reconstructed_image = torch.zeros_like(padded_image)
        reconstructed_dmap = torch.zeros_like(padded_gt_dmap)

        for i in range(0, img_h, stride_h):
            for j in range(0, img_w, stride_w):
                image_patch, gt_dmap_patch = CropToFixedSize()(
                    padded_image,
                    x_min=j,
                    y_min=i,
                    x_max=j + data_cfg['crop_width'],
                    y_max=i + data_cfg['crop_height'],
                    dmap=copy.deepcopy(padded_gt_dmap)
                )

                reconstructed_image[:, i:i + data_cfg['crop_height'], j:j + data_cfg['crop_width']] += image_patch
                normalization_map[:, i:i + data_cfg['crop_height'], j:j + data_cfg['crop_width']] += 1.0

                # Computing dmap for the patch
                pred_dmap_patch = model(image_patch.unsqueeze(dim=0).to(device))
                if model_cfg['name'] == "UNet":
                    pred_dmap_patch /= 1000

                reconstructed_dmap[:, i:i + data_cfg['crop_height'], j:j + data_cfg['crop_width']] += pred_dmap_patch.squeeze(dim=0)

        reconstructed_image /= normalization_map
        reconstructed_dmap /= normalization_map[0].unsqueeze(dim=0)

        # Updating metrics
        loss = torch.nn.MSELoss()(reconstructed_dmap.unsqueeze(dim=0), padded_gt_dmap.unsqueeze(dim=0))
        img_loss = loss.item()
        img_mae = abs(reconstructed_dmap.sum() - padded_gt_dmap.sum()).item()
        img_mse = ((reconstructed_dmap.sum() - padded_gt_dmap.sum()) ** 2).item()
        img_are = (abs(reconstructed_dmap.sum() - padded_gt_dmap.sum()) / torch.clamp(padded_gt_dmap.sum(), min=1)).item()
        img_ssim = ssim.ssim(padded_gt_dmap.unsqueeze(dim=0), reconstructed_dmap.unsqueeze(dim=0)).item()

        if train_cfg['debug'] and epoch % train_cfg['debug_freq'] == 0:
            debug_dir = os.path.join(tensorboard_writer.get_logdir(), 'output_debug')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            num_nets = torch.sum(reconstructed_dmap)
            pil_reconstructed_dmap = Image.fromarray(
                normalize(reconstructed_dmap.squeeze(dim=0).cpu().numpy()).astype('uint8'))
            draw = ImageDraw.Draw(pil_reconstructed_dmap)
            # Add text to image
            text = "Num of Nets: {}".format(num_nets)
            font_path = "./font/LEMONMILK-RegularItalic.otf"
            font = ImageFont.truetype(font_path, 100)
            draw.text((75, 75), text=text, font=font, fill=191)
            pil_reconstructed_dmap.save(
                os.path.join(debug_dir, "reconstructed_{}_dmap_epoch_{}.png".format(img_name.rsplit(".", 1)[0], epoch))
            )

        # Updating errors
        epoch_loss += img_loss
        epoch_mae += img_mae
        epoch_mse += img_mse
        epoch_are += img_are
        epoch_ssim += img_ssim

    # Computing mean of the errors
    epoch_mae /= len(val_dataloader.dataset)
    epoch_mse /= len(val_dataloader.dataset)
    epoch_are /= len(val_dataloader.dataset)
    epoch_loss /= len(val_dataloader.dataset)
    epoch_ssim /= len(val_dataloader.dataset)

    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Validation ended. Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    return epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim


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

    train_dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=get_dmap_transforms(train=True, crop_width=crop_width, crop_height=crop_height),
        list_frames=list_train_frames,
        load_in_memory=data_cfg['load_in_memory'],
        with_patches=data_cfg['train_with_precomputed_patches'],
        specular_split=data_cfg['specular_split'],
        percentage=data_cfg['percentage'],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        collate_fn=train_dataset.custom_collate_fn,
    )

    ################################
    # Creating training datasets and dataloaders
    print("Loading validation data")
    assert crop_width % 32 == 0 and crop_height % 32 == 0, "In validation mode, crop dim must be multiple of 32"

    val_dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=get_dmap_transforms(train=False, crop_width=crop_width, crop_height=crop_height),
        list_frames=list_val_frames,
        load_in_memory=False,
        with_patches=False,
        specular_split=data_cfg['specular_split'],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=val_dataset.custom_collate_fn,
    )

    # Initializing validation metrics
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    best_validation_are = float(sys.maxsize)
    best_validation_ssim = 0.0
    min_mae_epoch, min_mse_epoch, min_are_epoch, best_ssim_epoch = -1, -1, -1, -1

    #######################
    # Creating model
    #######################
    print("Creating model")
    model_name = model_cfg['name']
    assert model_name in available_models, "Not implemented model"
    if model_name == "UNet":
        model = available_models.get(model_cfg['name'])(in_channels=3, n_classes=1, padding=True, batch_norm=True)
    elif model_name == "CSRNet":
        load_initial_weights = True if train_cfg['pretrained_model'] or train_cfg['checkpoint'] else False
        model = available_models.get(model_cfg['name'])(load_weights=load_initial_weights)

    # Putting model to device and setting train mode
    model.to(device)
    model.train()

    ##################################################
    # Defining optimizer, LR scheduler and criterion
    ##################################################
    # Constructing an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if train_cfg['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=train_cfg['lr'])
    elif train_cfg['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(params,
                                    lr=train_cfg['lr'],
                                    momentum=train_cfg['momentum'],
                                    weight_decay=train_cfg['weights_decay'])
    else:
        print("Not implemented optimizer")
        exit(1)

    # Setting criterion
    criterion = None
    if train_cfg['loss'] == "MeanSquaredError":
        criterion = torch.nn.MSELoss()
    else:
        print("Not implemented loss")
        exit(1)
    aux_criterion = None
    if train_cfg['aux_loss'] == "SSIM":
        aux_criterion = ssim.SSIM(window_size=11)

    # Constructing lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
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
        best_validation_ssim = checkpoint['best_ssim']
        min_mae_epoch = checkpoint['min_mae_epoch']
        min_mse_epoch = checkpoint['min_mse_epoch']
        min_are_epoch = checkpoint['min_are_epoch']
        best_ssim_epoch = checkpoint['best_ssim_epoch']

    ################
    ################
    # Training
    print("Start training")
    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        # Training for one epoch
        for train_iteration, (images, targets) in enumerate(tqdm.tqdm(train_dataloader)):
            # Retrieving input images and associated gt
            images = images.to(device)
            gt_dmaps = targets['dmap'].to(device)

            # Computing pred dmaps
            pred_dmaps = model(images)
            if model_name == "UNet":
                pred_dmaps /= 1000

            # Computing loss and backwarding it
            if train_cfg['loss'] == "MeanSquaredError":
                loss = criterion(pred_dmaps, gt_dmaps)
            else:
                print("Loss to be implemented")
                exit(1)
            if train_cfg['aux_loss'] == "SSIM":
                aux_loss = -aux_criterion(gt_dmaps, pred_dmaps)
                ssim_value = - aux_loss.item()
                print("SSIM value: {}".format(ssim_value))
                loss += train_cfg['lambda_aux_loss'] * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating loss
            epoch_loss += loss.item()

            if train_iteration % train_cfg['log_loss'] == 0 and train_iteration != 0:
                writer.add_scalar('Training/Loss Total', epoch_loss / train_iteration, epoch * len(train_dataloader) + train_iteration)
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'],  epoch * len(train_dataloader) + train_iteration)

        # Updating lr scheduler
        lr_scheduler.step()

        # Validating
        if (epoch % train_cfg['val_freq'] == 0):
            epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim = \
                validate(model, val_dataloader, device, train_cfg, data_cfg, model_cfg, epoch, writer)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(dataset_name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(dataset_name), epoch_mse, epoch)
            writer.add_scalar('Validation on {}/ARE'.format(dataset_name), epoch_are, epoch)
            writer.add_scalar('Validation on {}/Loss'.format(dataset_name), epoch_loss, epoch)
            writer.add_scalar('Validation on {}/SSIM'.format(dataset_name), epoch_ssim, epoch)

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
            if epoch_are < best_validation_are:
                best_validation_are = epoch_are
                min_are_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_are': epoch_are,
                }, writer.get_logdir(), best_model=dataset_name + "_are")
            if epoch_ssim > best_validation_ssim:
                best_validation_ssim = epoch_ssim
                best_ssim_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_ssim': epoch_ssim,
                }, writer.get_logdir(), best_model=dataset_name + "_ssim")

            print('Epoch: ', epoch, ' Dataset: ', dataset_name, ' MAE: ', epoch_mae, ' MSE: ', epoch_mse, ' ARE: ', epoch_are, ' SSIM: ', epoch_ssim,
                  ' Min MAE: ', best_validation_mae, ' Min MAE Epoch: ', min_mae_epoch,
                  ' Min MSE: ', best_validation_mse, ' Min MSE Epoch: ', min_mse_epoch,
                  ' Min ARE: ', best_validation_are, ' Min ARE Epoch: ', min_are_epoch,
                  ' Best SSIM: ', best_validation_ssim, ' Best SSIM Epoch: ', best_ssim_epoch
                  )

            # Saving last model
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_validation_mae,
                'best_mse': best_validation_mse,
                'best_are': best_validation_are,
                'best_ssim': best_validation_ssim,
                'min_mae_epoch': min_mae_epoch,
                'min_mse_epoch': min_mse_epoch,
                'min_are_epoch': min_are_epoch,
                'best_ssim_epoch': best_ssim_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, writer.get_logdir())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)

