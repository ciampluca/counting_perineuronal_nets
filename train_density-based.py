# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import tqdm
import sys
from PIL import Image
import timeit

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.CSRNet import CSRNet
from utils.misc import random_seed, get_dmap_transforms, save_checkpoint, normalize
from datasets.perineural_nets_dmap_dataset import PerineuralNetsDmapDataset


@torch.no_grad()
def validate(model, val_dataloader, device, train_cfg, data_cfg, epoch, tensorboard_writer):
    # Validation
    model.eval()
    print("Validation")
    start = timeit.default_timer()

    epoch_mae, epoch_mse, epoch_are, epoch_loss = 0.0, 0.0, 0.0, 0.0
    for images, targets in tqdm.tqdm(val_dataloader):
        images = images.to(device)
        gt_dmaps = targets['dmap'].to(device)
        img_name = val_dataloader.dataset.image_files[targets['img_id']]

        # Image and gt dmap are divided in patches
        dmap_output_patches = []
        img_patches = images.data.unfold(1, 3, 3).unfold(2, data_cfg['crop_width'], data_cfg['crop_height']).\
            unfold(3, data_cfg['crop_width'], data_cfg['crop_height'])
        gt_dmap_patches = gt_dmaps.data.unfold(1, 1, 1).unfold(2, data_cfg['crop_width'], data_cfg['crop_height']).\
            unfold(3, data_cfg['crop_width'], data_cfg['crop_height'])
        counter_patches = 1

        img_loss, img_mae, img_mse, img_are = 0.0, 0.0, 0.0, 0.0
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                for r in range(img_patches.shape[2]):
                    for c in range(img_patches.shape[3]):
                        img_patch = img_patches[i, j, r, c, ...]
                        gt_dmap_patch = gt_dmap_patches[i, j, r, c, ...]

                        # Computing dmap for the patch
                        pred_dmap_patch = model(img_patch.unsqueeze(0).to(device))
                        dmap_output_patches.append(pred_dmap_patch.squeeze(dim=1))

                        # Computing loss and updating errors for the patch
                        patch_loss = torch.nn.MSELoss()(pred_dmap_patch, gt_dmap_patch.unsqueeze(1))
                        img_loss += patch_loss.item()
                        patch_mae = abs(pred_dmap_patch.sum() - gt_dmap_patch.sum())
                        patch_mse = (pred_dmap_patch.sum() - gt_dmap_patch.sum()) ** 2
                        patch_are = abs(pred_dmap_patch.sum() - gt_dmap_patch.sum()) / \
                                    torch.clamp(gt_dmap_patch.sum(), min=1)
                        img_mae += patch_mae.item()
                        img_mse += patch_mse.item()
                        img_are += patch_are.item()

                        counter_patches += 1

        if train_cfg['debug'] and epoch % train_cfg['debug_freq'] == 0:
            debug_dir = os.path.join(tensorboard_writer.get_logdir(), 'output_debug')
            # Reconstructing predicted dmap
            dmap_output_patches = torch.stack(dmap_output_patches)
            rec_pred_dmap = dmap_output_patches.view(
                gt_dmap_patches.shape[2], gt_dmap_patches.shape[3], *gt_dmap_patches.size()[-3:])
            rec_pred_dmap = rec_pred_dmap.permute(2, 0, 3, 1, 4).contiguous()
            rec_pred_dmap = rec_pred_dmap.view(
                rec_pred_dmap.shape[0], rec_pred_dmap.shape[1] * rec_pred_dmap.shape[2],
                rec_pred_dmap.shape[3] * rec_pred_dmap.shape[4])
            Image.fromarray(normalize(rec_pred_dmap[0].cpu().numpy()).astype('uint8')).save(
                os.path.join(debug_dir, "reconstructed_{}_dmap_epoch_{}.png".format(img_name.rsplit(".", 1)[0], epoch)))

        # Updating errors
        epoch_loss += img_loss
        epoch_mae += img_mae
        epoch_mse += img_mse
        epoch_are += img_are

    # Computing mean of the errors
    epoch_mae /= len(val_dataloader.dataset)
    epoch_mse /= len(val_dataloader.dataset)
    epoch_are /= len(val_dataloader.dataset)
    epoch_loss /= len(val_dataloader.dataset)

    return epoch_mae, epoch_mse, epoch_are, epoch_loss


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
    min_mae_epoch, min_mse_epoch, min_are_epoch = -1, -1, -1

    #######################
    # Creating model
    #######################
    print("Creating model")
    load_initial_weights = True if train_cfg['pretrained_model'] or train_cfg['checkpoint'] else False
    model = CSRNet(load_weights=load_initial_weights)

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
        min_mae_epoch = checkpoint['min_mae_epoch']
        min_mse_epoch = checkpoint['min_mse_epoch']
        min_are_epoch = checkpoint['min_are_epoch']

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

            # Computing loss and backwarding it
            if train_cfg['loss'] == "MeanSquaredError":
                loss = criterion(pred_dmaps, gt_dmaps)
            else:
                print("Loss to be implemented")
                exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating loss
            epoch_loss += loss.item()

            if (train_iteration % train_cfg['log_loss'] == 0):
                writer.add_scalar('Train/Loss Total', epoch_loss / train_iteration, epoch * len(train_dataloader) + train_iteration)
                writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'],  epoch * len(train_dataloader) + train_iteration)

        # writer.add_scalar('Train/Loss Total', epoch_loss / len(train_dataset), epoch)
        # writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'],  epoch)

        # Updating lr scheduler
        lr_scheduler.step()

        # Validating
        if (epoch % train_cfg['val_freq'] == 0):
            epoch_mae, epoch_mse, epoch_are, epoch_loss = validate(model, val_dataloader, device, train_cfg, data_cfg, epoch, writer)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(dataset_name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(dataset_name), epoch_mse, epoch)
            writer.add_scalar('Validation on {}/ARE'.format(dataset_name), epoch_are, epoch)
            writer.add_scalar('Validation on {}/Loss'.format(dataset_name), epoch_loss, epoch)

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

            print('Epoch: ', epoch, ' Dataset: ', dataset_name, ' MAE: ', epoch_mae, ' MSE: ', epoch_mse, ' ARE: ', epoch_are,
                  ' Min MAE: ', best_validation_mae, ' Min MAE Epoch: ', min_mae_epoch,
                  ' Min MSE: ', best_validation_mse, ' Min MSE Epoch: ', min_mse_epoch,
                  ' Min ARE: ', best_validation_are, ' Min ARE Epoch: ', min_are_epoch
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
                'min_mae_epoch': min_mae_epoch,
                'min_mse_epoch': min_mse_epoch,
                'min_are_epoch': min_are_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, writer.get_logdir())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)

