import os
import logging
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from utils import CheckpointManager, seed_everything

log = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)
trange = partial(trange, dynamic_ncols=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    from omegaconf import OmegaConf; print(OmegaConf.to_yaml(cfg))
    
    log.info(f"Run path: {Path.cwd()}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    # Reproducibility
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # create tensorboard writer
    writer = SummaryWriter()

    # training dataset and dataloader
    try:
        collate_fn = hydra.utils.get_method(f'methods.{cfg.method}.utils.collate_fn')
    except Exception as e:
        collate_fn = None
        log.warning('No collate_fn found, using the default one ...')

    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=cfg.optim.num_workers, collate_fn=collate_fn)
    log.info(f'[TRAIN] {train_dataset}')

    # validation dataset and dataloader
    valid_dataset = hydra.utils.instantiate(cfg.data.validation)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.optim.val_batch_size, num_workers=cfg.optim.num_workers, collate_fn=collate_fn)
    log.info(f'[VALID] {valid_dataset}')

    # create model
    torch.hub.set_dir(cfg.model.cache_folder)
    skip_weights_loading = cfg.optim.resume or cfg.model.pretrained
    model = hydra.utils.instantiate(cfg.model.module, skip_weights_loading=skip_weights_loading)
    # move model to device
    model.to(device)
    model_param_string = ', '.join(f'{k}={v}' for k, v in cfg.model.module.items() if not k.startswith('_'))
    log.info(f"[MODEL] {cfg.model.name}({model_param_string})")

    # build the optimizer
    optimizer = hydra.utils.instantiate(cfg.optim.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.optim.lr_scheduler, optimizer)

    # optionally load pre-trained weights
    if cfg.model.pretrained and cfg.model.resume is not None:
        if cfg.model.pretrained.startswith('http://') or cfg.model.pretrained.startswith('https://'):
            pre_trained_model = torch.hub.load_state_dict_from_url(
                cfg.model.pretrained, map_location=device, model_dir=cfg.model.cache_folder)
        else:
            pre_trained_model = torch.load(cfg.model.pretrained, map_location=device)
        model.load_state_dict(pre_trained_model['model'])
        log.info(f"[PRETRAINED]: {cfg.model.pretrained}")
        
    start_epoch = 0
    best_metrics = {}

    train_log_path = 'train_log.csv'
    valid_log_path = 'valid_log.csv'

    train_log = pd.DataFrame()
    valid_log = pd.DataFrame()

    # optionally resume from a saved checkpoint
    if cfg.optim.resume:
        assert Path('last.pth').exists(), 'Cannot find checkpoint for resuming.'
        checkpoint = torch.load('last.pth', map_location=device)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint['best_metrics']

        train_log = pd.read_csv(train_log_path, index_col=0, header=[0,1])
        valid_log = pd.read_csv(valid_log_path, index_col=0, header=[0,1])
        log.info(f"[RESUME] Resuming from epoch {start_epoch}")

    # get method-specific train and validation loops
    train_one_epoch = hydra.utils.get_method(f'methods.{cfg.method}.train_fn.train_one_epoch')
    validate = hydra.utils.get_method(f'methods.{cfg.method}.train_fn.validate')

    # checkpoint manager
    ckpt_dir = Path('best_models')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = CheckpointManager(ckpt_dir, current_best=best_metrics)

    # Train loop
    log.info(f"Training ...")
    progress = trange(start_epoch, cfg.optim.epochs, initial=start_epoch)
    for epoch in progress:
        # train
        train_metrics = train_one_epoch(train_loader, model, optimizer, device, writer, epoch, cfg)
        scheduler.step()  # update lr scheduler

        # convert for pandas
        train_metrics = {(metric, info): v for metric, infos in train_metrics.items() for info, v in infos.items()}
        train_metrics = pd.DataFrame(train_metrics, index=[epoch]).rename_axis('epoch')
        train_log = train_log.append(train_metrics)
        train_log.to_csv(train_log_path)

        # evaluation
        if (epoch + 1) % cfg.optim.val_freq == 0:
            valid_metrics = validate(valid_loader, model, device, epoch, cfg)

            for metric, info in valid_metrics.items():  # log to tensorboard
                value = info.get('value', None)
                writer.add_scalar(f'valid/{metric}', value, epoch)

                threshold = info.get('threshold', None)
                if threshold is not None:
                    writer.add_scalar(f'valid/{metric}_thr', threshold, epoch)

            
            # save only if best on some metric (via CheckpointManager)
            best_metrics = ckpt_manager.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': valid_metrics
            }, valid_metrics, epoch)

            # save last checkpoint for resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metrics': best_metrics,
            }, 'last.pth')

            # convert for pandas
            valid_metrics = {(metric, info): v for metric, infos in valid_metrics.items() for info, v in infos.items()}
            valid_metrics = pd.DataFrame(valid_metrics, index=[epoch]).rename_axis('epoch')
            valid_log = valid_log.append(valid_metrics)
            valid_log.to_csv(valid_log_path)

    log.info("Training ended. Exiting....")


if __name__ == "__main__":
    main()
