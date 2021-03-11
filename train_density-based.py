# -*- coding: utf-8 -*-
import yaml
import os
from shutil import copyfile
import tqdm
import sys
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from models.CSRNet import CSRNet
from utils.misc import random_seed, get_dmap_transforms, save_checkpoint, normalize
from datasets.perineural_nets_dmap_dataset import PerineuralNetsDmapDataset


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

    train_dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=get_dmap_transforms(train=True, crop_width=crop_width, crop_height=crop_height),
        list_frames=data_cfg['train_frames']
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

    val_dataset = PerineuralNetsDmapDataset(
        data_root=data_root,
        transforms=get_dmap_transforms(train=False, crop_width=crop_width, crop_height=crop_height),
        list_frames=data_cfg['val_frames']
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        collate_fn=val_dataset.custom_collate_fn,
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

    ################
    ################
    # Training
    print("Start training")
    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        # Training for one epoch
        for images, targets in tqdm.tqdm(train_dataloader):
            # Retrieving input images and associated gt
            images = images.to(device)
            gt_dmaps = targets['dmap'].unsqueeze(1).to(device)

            # Computing pred dmaps
            pred_dmaps = model(images)

            # Computing loss and backwarding it
            if train_cfg['loss'] == "MeanSquaredError":
                loss = criterion(pred_dmaps, gt_dmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating loss
            epoch_loss += loss.item()

        writer.add_scalar('Train/Loss Total', epoch_loss / len(train_dataset), epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'],  epoch)

        # Validating the epoch
        model.eval()
        with torch.no_grad():
            print("Validation")
            epoch_mae, epoch_mse, epoch_are, epoch_loss = 0.0, 0.0, 0.0, 0.0
            for images, targets in tqdm.tqdm(val_dataloader):
                images = images.to(device)
                gt_dmaps = targets['dmap'].unsqueeze(1).to(device)
                img_name = val_dataset.imgs[targets['img_id']]

                # Image and gt dmap are divided in patches
                output_img_patches, output_gt_dmap_patches, output_pred_dmap_patches = [], [], []
                img_patches = images.data.unfold(1, 3, 3).unfold(2, crop_width, crop_height).\
                    unfold(3, crop_width, crop_height)
                gt_dmap_patches = gt_dmaps.data.unfold(1, 1, 1).unfold(2, crop_width, crop_height).\
                    unfold(3, crop_width, crop_height)
                counter_patches = 1

                patches_loss, patches_mae, patches_mse, patches_are = 0.0, 0.0, 0.0, 0.0
                for i in range(img_patches.shape[0]):
                    for j in range(img_patches.shape[1]):
                        for r in range(img_patches.shape[2]):
                            for c in range(img_patches.shape[3]):
                                print("Patches: {}".format(counter_patches))
                                img_patch = img_patches[i, j, r, c, ...]
                                gt_dmap_patch = gt_dmap_patches[i, j, r, c, ...]
                                output_img_patches.append(img_patch)
                                output_gt_dmap_patches.append(gt_dmap_patch)

                                # Computing dmap for the patch
                                pred_dmap_patch = model(img_patch.unsqueeze(0).to(device))
                                output_pred_dmap_patches.append(pred_dmap_patch.squeeze(dim=1))

                                # Computing loss and updating errors for the patch
                                patch_loss = criterion(pred_dmap_patch, gt_dmap_patch.unsqueeze(1))
                                patches_loss += patch_loss.item()
                                patch_mae = abs(pred_dmap_patch.sum() - gt_dmap_patch.sum())
                                patch_mse = (pred_dmap_patch.sum() - gt_dmap_patch.sum()) ** 2
                                patch_are = abs(pred_dmap_patch.sum() - gt_dmap_patch.sum()) / \
                                            torch.clamp(gt_dmap_patch.sum(), min=1)
                                patches_mae += patch_mae.item()
                                patches_mse += patch_mse.item()
                                patches_are += patch_are.item()

                                counter_patches += 1

                # Reconstructing image, gt and predicted dmap
                # Not necessary to reconstruct image and gt, just for checking
                output_img_patches = torch.stack(output_img_patches)
                rec_image = output_img_patches.view(img_patches.shape[2], img_patches.shape[3], *img_patches.size()[-3:])
                rec_image = rec_image.permute(2, 0, 3, 1, 4).contiguous()
                rec_image = rec_image.view(rec_image.shape[0], rec_image.shape[1] * rec_image.shape[2],
                                           rec_image.shape[3] * rec_image.shape[4])
                to_pil_image(rec_image).save("./output/training/reconstructed_{}.png".format(img_name.rsplit(".", 1)[0]))

                output_gt_dmap_patches = torch.stack(output_gt_dmap_patches)
                rec_gt_dmap = output_gt_dmap_patches.view(gt_dmap_patches.shape[2],
                                                          gt_dmap_patches.shape[3], *gt_dmap_patches.size()[-3:])
                rec_gt_dmap = rec_gt_dmap.permute(2, 0, 3, 1, 4).contiguous()
                rec_gt_dmap = rec_gt_dmap.view(rec_gt_dmap.shape[0], rec_gt_dmap.shape[1] * rec_gt_dmap.shape[2],
                                               rec_gt_dmap.shape[3] * rec_gt_dmap.shape[4])
                Image.fromarray(normalize(rec_gt_dmap[0].cpu().numpy()).astype('uint8')).save(
                    os.path.join("./output/training/reconstructed_{}_gt_dmap.png".format(img_name.rsplit(".", 1)[0])))

                output_pred_dmap_patches = torch.stack(output_pred_dmap_patches)
                rec_pred_dmap = output_pred_dmap_patches.view(gt_dmap_patches.shape[2],
                                                          gt_dmap_patches.shape[3], *gt_dmap_patches.size()[-3:])
                rec_pred_dmap = rec_pred_dmap.permute(2, 0, 3, 1, 4).contiguous()
                rec_pred_dmap = rec_pred_dmap.view(rec_pred_dmap.shape[0], rec_pred_dmap.shape[1] * rec_pred_dmap.shape[2],
                                               rec_pred_dmap.shape[3] * rec_pred_dmap.shape[4])
                Image.fromarray(normalize(rec_pred_dmap[0].cpu().numpy()).astype('uint8')).save(
                    os.path.join("./output/training/epoch_{}_reconstructed_{}_pred_dmap.png".
                                 format(epoch, img_name.rsplit(".", 1)[0])))

                # Updating errors
                epoch_loss += patches_loss
                epoch_mae += patches_mae
                epoch_mse += patches_mse
                epoch_are += patches_are

            # Computing mean of the errors
            epoch_mae /= len(val_dataset)
            epoch_mse /= len(val_dataset)
            epoch_are /= len(val_dataset)
            epoch_loss /= len(val_dataset)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(dataset_name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(dataset_name), epoch_mse, epoch)
            writer.add_scalar('Validation on {}/ARE'.format(dataset_name), epoch_are, epoch)
            writer.add_scalar('Validation on {}/Loss'.format(dataset_name), epoch_loss, epoch)

            writer.add_image(
                "Dataset {} - Image {}".format(dataset_name, img_name), rec_image)
            writer.add_image(
                "Epoch {} - Dataset {} - Pred Count: ".format(epoch, dataset_name) + str(
                    '%.2f' % (rec_pred_dmap.cpu().sum())),
                abs(rec_pred_dmap) / torch.max(rec_pred_dmap))
            writer.add_image(
                "Dataset {} - GT count: ".format(dataset_name) + str('%.2f' % (rec_gt_dmap.cpu().sum())),
                rec_gt_dmap / torch.max(rec_gt_dmap))

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
        if epoch_are < best_validation_are:
            best_validation_are = epoch_are
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_are': epoch_are,
            }, writer.get_logdir(), best_model=dataset_name + "_are")

        print('Epoch: ', epoch, ' Dataset: ', dataset_name, ' MAE: ', epoch_mae,
              ' Min MAE : ', best_validation_mae, ' Min MAE Epoch: ', min_mae_epoch,
              ' MSE: ', epoch_mse, ' ARE: ', epoch_are)

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
            'tensorboard_working_dir': writer.get_logdir()
        }, writer.get_logdir())

        # Updating lr scheduler
        lr_scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)

