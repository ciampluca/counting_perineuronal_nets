# -*- coding: utf-8 -*-
import os
import tqdm
import sys
from PIL import Image, ImageFont, ImageDraw
import timeit
import ssim
import math
import copy
import logging
from omegaconf import DictConfig
import hydra

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.misc import random_seed, get_dmap_transforms, save_checkpoint, normalize, update_dict
from datasets.perineural_nets_dmap_dataset import PerineuralNetsDmapDataset
from utils.transforms_dmaps import PadToSize, CropToFixedSize, PadToResizeFactor

# Creating logger
log = logging.getLogger("Counting Nets")


@torch.no_grad()
def validate(model, val_dataloader, device, cfg, epoch):
    # Validation
    model.eval()
    log.info(f"Start validation of the epoch {epoch}")
    start = timeit.default_timer()

    crop_width, crop_height = cfg.dataset.validation.params.input_size
    stride_w = crop_width - cfg.dataset.validation.params.patches_overlap
    stride_h = crop_height - cfg.dataset.validation.params.patches_overlap

    epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_game_metrics = dict()
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
                    stride_w * num_h_patches + (crop_width - stride_w))
        img_h_padded = (num_v_patches - math.floor(img_h / stride_h)) * (
                    stride_h * num_v_patches + (crop_height - stride_h))
        padded_image, padded_gt_dmap = PadToSize()(
            image=image,
            min_width=img_w_padded,
            min_height=img_h_padded,
            dmap=gt_dmap
        )

        h_pad_top = int((img_h_padded - img_h) / 2.0)
        h_pad_bottom = img_h_padded - img_h - h_pad_top
        w_pad_left = int((img_w_padded - img_w) / 2.0)
        w_pad_right = img_w_padded - img_w - w_pad_left

        normalization_map = torch.zeros_like(padded_image)
        reconstructed_dmap = torch.zeros_like(padded_gt_dmap)

        for i in range(0, img_h, stride_h):
            for j in range(0, img_w, stride_w):
                image_patch, gt_dmap_patch = CropToFixedSize()(
                    padded_image,
                    x_min=j,
                    y_min=i,
                    x_max=j + crop_width,
                    y_max=i + crop_height,
                    dmap=copy.deepcopy(padded_gt_dmap)
                )

                normalization_map[:, i:i + crop_height, j:j + crop_width] += 1.0

                # Computing dmap for the patch
                pred_dmap_patch = model(image_patch.unsqueeze(dim=0).to(device))
                if cfg.model.name == "UNet":
                    pred_dmap_patch /= 1000

                reconstructed_dmap[:, i:i + crop_height, j:j + crop_width] += pred_dmap_patch.squeeze(dim=0)

        reconstructed_dmap /= normalization_map[0].unsqueeze(dim=0)
        reconstructed_dmap = reconstructed_dmap[:, h_pad_top:img_h_padded - h_pad_bottom,
                             w_pad_left:img_w_padded - w_pad_right]

        # Updating metrics
        loss = torch.nn.MSELoss()(reconstructed_dmap.unsqueeze(dim=0), padded_gt_dmap.unsqueeze(dim=0))
        img_loss = loss.item()
        img_mae = abs(reconstructed_dmap.sum() - padded_gt_dmap.sum()).item()
        img_mse = ((reconstructed_dmap.sum() - padded_gt_dmap.sum()) ** 2).item()
        img_are = (abs(reconstructed_dmap.sum() - padded_gt_dmap.sum()) / torch.clamp(padded_gt_dmap.sum(), min=1)).item()
        img_ssim = ssim.ssim(padded_gt_dmap.unsqueeze(dim=0), reconstructed_dmap.unsqueeze(dim=0)).item()

        # Updating errors
        epoch_loss += img_loss
        epoch_mae += img_mae
        epoch_mse += img_mse
        epoch_are += img_are
        epoch_ssim += img_ssim

        # Computing GAME metrics for the image
        padded_image_for_game_metrics, padded_gt_dmap_for_game_metrics = PadToResizeFactor(resize_factor=12)(
            image,
            dmap=copy.deepcopy(gt_dmap)
        )

        img_for_game_metrics_w, img_for_game_metrics_h = padded_image_for_game_metrics.shape[2], padded_image_for_game_metrics.shape[1]

        for L in range(1, 4):
            epoch_game_metrics[f"GAME_{L}"] = 0.0
            num_patches = 4*L
            num_h_patches, num_v_patches = img_for_game_metrics_w / (num_patches/2), img_for_game_metrics_h/ (num_patches/2)
            crop_width, crop_height = img_for_game_metrics_w / num_h_patches, img_for_game_metrics_h / num_v_patches

            for i in range(0, img_for_game_metrics_h, crop_height):
                for j in range(0, img_for_game_metrics_w, crop_width):
                    pred_dmap_patch_for_game_metrics, gt_dmap_patch_for_game_metrics = CropToFixedSize()(
                        copy.deepcopy(reconstructed_dmap),
                        x_min=j,
                        y_min=i,
                        x_max=j + crop_width,
                        y_max=i + crop_height,
                        dmap=copy.deepcopy(padded_gt_dmap_for_game_metrics)
                    )

                    epoch_game_metrics[f"GAME_{L}"] += abs(pred_dmap_patch_for_game_metrics.sum() - gt_dmap_patch_for_game_metrics.sum())

        if cfg.training.debug and epoch % cfg.training.debug_freq == 0:
            debug_folder = os.path.join(os.getcwd(), 'output_debug')
            if not os.path.exists(debug_folder):
                os.makedirs(debug_folder)
            num_nets = torch.sum(reconstructed_dmap)
            pil_reconstructed_dmap = Image.fromarray(
                normalize(reconstructed_dmap.squeeze(dim=0).cpu().numpy()).astype('uint8'))
            draw = ImageDraw.Draw(pil_reconstructed_dmap)
            # Add text to image
            text = f"Num of Nets: {num_nets}"
            font_path = "./font/LEMONMILK-RegularItalic.otf"
            font = ImageFont.truetype(font_path, 100)
            draw.text((75, 75), text=text, font=font, fill=191)
            pil_reconstructed_dmap.save(
                os.path.join(debug_folder, "reconstructed_{}_dmap_epoch_{}.png".format(img_name.rsplit(".", 1)[0], epoch))
            )

    # Computing mean of the errors
    epoch_mae /= len(val_dataloader.dataset)
    epoch_mse /= len(val_dataloader.dataset)
    epoch_are /= len(val_dataloader.dataset)
    epoch_loss /= len(val_dataloader.dataset)
    epoch_ssim /= len(val_dataloader.dataset)
    for k, v in epoch_game_metrics.items():
        epoch_game_metrics[k] = v / len(val_dataloader.dataset)

    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    log.info(f"Validation epoch {epoch} ended. Total running time: {hours}:{mins}:{secs}.")

    return epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim, epoch_game_metrics


@hydra.main(config_path="conf/density_based", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = copy.deepcopy(hydra_cfg.technique)
    for _, v in hydra_cfg.items():
        update_dict(cfg, v)
    experiment_name = f"{cfg.model.name}_{cfg.dataset.training.name}_specular_split-{cfg.training.specular_split}" \
               f"_input_size-{cfg.dataset.training.params.input_size}_loss-${cfg.model.loss.name}" \
               f"_aux_loss-${cfg.model.aux_loss.name}_val_patches_overlap-${cfg.dataset.validation.params.patches_overlap}" \
               f"_batch_size-${cfg.training.batch_size}"

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
    if cfg.model.resume:
        checkpoint = torch.load(cfg.model.resume)
        writer = SummaryWriter(log_dir=checkpoint['tensorboard_working_dir'])
    else:
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

    train_dataset = PerineuralNetsDmapDataset(
        data_root=cfg.dataset.training.root,
        transforms=get_dmap_transforms(train=True, crop_width=training_crop_width, crop_height=training_crop_height),
        list_frames=list_train_frames,
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
        collate_fn=train_dataset.custom_collate_fn,
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

    val_dataset = PerineuralNetsDmapDataset(
        data_root=cfg.dataset.validation.root,
        transforms=get_dmap_transforms(train=False),
        list_frames=list_val_frames,
        load_in_memory=False,
        with_patches=False,
        specular_split=cfg.training.specular_split,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=val_dataset.custom_collate_fn,
    )

    log.info(f"Found {len(val_dataset)} samples in validation dataset")

    # Initializing validation metrics
    best_validation_mae = float(sys.maxsize)
    best_validation_mse = float(sys.maxsize)
    best_validation_are = float(sys.maxsize)
    best_validation_game_1, best_validation_game_2, best_validation_game_3 = float(sys.maxsize), float(sys.maxsize), float(sys.maxsize)
    best_validation_ssim = 0.0
    min_mae_epoch, min_mse_epoch, min_are_epoch, best_ssim_epoch = -1, -1, -1, -1
    min_game_1_epoch, min_game_2_epoch, min_game_3_epoch = -1, -1, -1

    # Creating model
    log.info(f"Creating model")
    if cfg.model.resume or cfg.model.pretrained:
        cfg.model.params.load_weights = True
    model = hydra.utils.get_class(f"models.{cfg.model.name}.{cfg.model.name}")
    model = model(**cfg.model.params)

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

    # Setting criterion
    criterion = hydra.utils.get_class(
            f"torch.nn.{cfg.model.loss.name}")()
    aux_criterion = None
    if cfg.model.aux_loss.name:
        aux_criterion = hydra.utils.get_class(
            f"ssim.{cfg.model.aux_loss.name}")
        aux_criterion = aux_criterion(
            **
            {**cfg.model.aux_loss.params})

    start_epoch = 0
    # Eventually resuming a pre-trained model
    if cfg.model.pretrained:
        log.info(f"Resuming pre-trained model")
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location='cpu', model_dir=cfg.model.cache_folder)
        else:
            pre_trained_model = torch.load(cfg.model.pretrained, map_location='cpu')
        model.load_state_dict(pre_trained_model['model'])

    # Eventually resuming from a saved checkpoint
    if cfg.model.resume:
        log.info(f"Resuming from a checkpoint")
        checkpoint = torch.load(cfg.model.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best_validation_mae = checkpoint['best_mae']
        best_validation_mse = checkpoint['best_mse']
        best_validation_are = checkpoint['best_are']
        best_validation_ssim = checkpoint['best_ssim']
        best_validation_game_1 = checkpoint['best_game_1']
        best_validation_game_2 = checkpoint['best_game_1']
        best_validation_game_3 = checkpoint['best_game_1']
        min_mae_epoch = checkpoint['min_mae_epoch']
        min_mse_epoch = checkpoint['min_mse_epoch']
        min_are_epoch = checkpoint['min_are_epoch']
        best_ssim_epoch = checkpoint['best_ssim_epoch']
        min_game_1_epoch = checkpoint['min_game_1_epoch']
        min_game_2_epoch = checkpoint['min_game_2_epoch']
        min_game_3_epoch = checkpoint['min_game_3_epoch']

    ################
    ################
    # Training
    log.info(f"Start training")
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        # Training for one epoch
        for train_iteration, (images, targets) in enumerate(tqdm.tqdm(train_dataloader)):
            # Retrieving input images and associated gt
            images = images.to(device)
            gt_dmaps = targets['dmap'].to(device)

            # Computing pred dmaps
            pred_dmaps = model(images)
            if cfg.model.name == "UNet":
                pred_dmaps /= 1000

            # Computing loss and backwarding it
            loss = criterion(pred_dmaps, gt_dmaps)
            if cfg.model.aux_loss.name == "SSIM":
                aux_loss = -aux_criterion(gt_dmaps, pred_dmaps)
                loss += cfg.model.aux_loss.lambda_multiplier * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating loss
            epoch_loss += loss.item()

            if train_iteration % cfg.training.log_loss == 0 and train_iteration != 0:
                writer.add_scalar('Training/Density-based Loss Total', epoch_loss / train_iteration, epoch * len(train_dataloader) + train_iteration)
                writer.add_scalar('Training/Density-based Learning Rate', optimizer.param_groups[0]['lr'],  epoch * len(train_dataloader) + train_iteration)

        # Updating lr scheduler
        scheduler.step()

        # Validating
        if epoch % cfg.training.val_freq == 0:
            epoch_mae, epoch_mse, epoch_are, epoch_loss, epoch_ssim, epoch_game_metrics = \
                validate(model, val_dataloader, device, cfg, epoch)

            # Updating tensorboard
            writer.add_scalar('Validation on {}/MAE'.format(cfg.dataset.validation.name), epoch_mae, epoch)
            writer.add_scalar('Validation on {}/MSE'.format(cfg.dataset.validation.name), epoch_mse, epoch)
            writer.add_scalar('Validation on {}/ARE'.format(cfg.dataset.validation.name), epoch_are, epoch)
            writer.add_scalar('Validation on {}/Loss'.format(cfg.dataset.validation.name), epoch_loss, epoch)
            writer.add_scalar('Validation on {}/SSIM'.format(cfg.dataset.validation.name), epoch_ssim, epoch)
            for k, v in epoch_game_metrics:
                writer.add_scalar('Validation on {}/{}'.format(cfg.dataset.validation.name, k), v, epoch)

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
                    'best_are': epoch_are,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_are")
            if epoch_ssim > best_validation_ssim:
                best_validation_ssim = epoch_ssim
                best_ssim_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_ssim': epoch_ssim,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_ssim")
            epoch_game_1, epoch_game_2, epoch_game_3 = 0.0, 0.0, 0.0
            for k, v in epoch_game_metrics:
                L = int(k.rsplit("_", 1)[1])
                if L == 1:
                    epoch_game_1 = v
                elif L == 2:
                    epoch_game_2 = v
                elif L == 3:
                    epoch_game_3 = v
            if epoch_game_1 > best_validation_game_1:
                best_validation_game_1 = epoch_game_1
                min_game_1_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_ssim': epoch_game_1,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_game_1")
            if epoch_game_2 > best_validation_game_2:
                best_validation_game_2 = epoch_game_2
                min_game_1_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_ssim': epoch_game_2,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_game_2")
            if epoch_game_3 > best_validation_game_3:
                best_validation_game_3 = epoch_game_3
                min_game_3_epoch = epoch
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_ssim': epoch_game_3,
                }, best_models_folder, best_model=cfg.dataset.validation.name + "_game_3")

            nl = '\n'
            log.info(f"Epoch: {epoch}, Dataset: {cfg.dataset.validation.name}, MAE: {epoch_mae}, MSE: {epoch_mse}, "
                     f"ARE: {epoch_are}, SSIM: {epoch_ssim}, "
                     f"GAME_1: {epoch_game_1}, GAME_2: {epoch_game_2}, GAME_3: {epoch_game_3}, {nl}, "
                     f"Min MAE: {best_validation_mae}, Min MAE Epoch: {min_mae_epoch}, {nl}, "
                     f"Min MSE: {best_validation_mse}, Min MSE Epoch: {min_mse_epoch}, {nl}, "
                     f"Min ARE: {best_validation_are}, Min ARE Epoch: {min_are_epoch}, {nl}, "
                     f"Best SSIM: {best_validation_ssim}, Best SSIM Epoch: {best_ssim_epoch}, "
                     f"Best GAME_1: {best_validation_game_1}, Best GAME_1 Epoch: {min_game_1_epoch}, "
                     f"Best GAME_2: {best_validation_game_2}, Best GAME_2 Epoch: {min_game_2_epoch}, "
                     f"Best GAME_3: {best_validation_game_3}, Best GAME_3 Epoch: {min_game_3_epoch}")

            # Saving last model
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_mae': best_validation_mae,
                'best_mse': best_validation_mse,
                'best_are': best_validation_are,
                'best_ssim': best_validation_ssim,
                'best_game_1': best_validation_game_1,
                'best_game_2': best_validation_game_2,
                'best_game_3': best_validation_game_3,
                'min_mae_epoch': min_mae_epoch,
                'min_mse_epoch': min_mse_epoch,
                'min_are_epoch': min_are_epoch,
                'best_ssim_epoch': best_ssim_epoch,
                'min_game_1_epoch': min_game_1_epoch,
                'min_game_2_epoch': min_game_2_epoch,
                'min_game_3_epoch': min_game_3_epoch,
                'tensorboard_working_dir': writer.get_logdir()
            }, os.getcwd())


if __name__ == "__main__":
    main()

