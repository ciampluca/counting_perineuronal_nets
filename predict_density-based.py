# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import copy
from skimage import io, draw
from tqdm import tqdm
import itertools

import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import hydra
from omegaconf import OmegaConf

from prefetch_generator import BackgroundGenerator

from datasets.perineural_nets_dmap_dataset import PerineuralNetsDMapDataset
from utils.misc import normalize


def process_per_patch(dataloader, process_fn, cfg):
    validation_device = cfg.train.val_device

    def _process_batch(batch):
        patch, patch_hw, start_yx, image_hw, image_id = batch

        processed_patch = process_fn(patch)

        patch, processed_patch = map(lambda x: x[:, 0].to(validation_device), (patch, processed_patch))

        processed_batch = (image_id, image_hw, patch, processed_patch, patch_hw, start_yx)

        return processed_batch

    processed_batches = map(_process_batch, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=15000)  # prefetch batches using threading

    def _unbatch(batches):
        for batch in batches:
            yield from zip(*batch)

    processed_samples = _unbatch(processed_batches)

    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)
    groups = tqdm(groups, total=len(dataloader.dataset.datasets), desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in groups:
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        full_pred_dmap = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        for _, _, patch, gt_dmap, pred_dmap, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y+h, x:x+w] = patch[:h, :w]
            full_pred_dmap[y:y+h, x:x+w] += pred_dmap[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        full_pred_dmap /= normalization_map

        # Cleaning
        del normalization_map

        yield image_id, full_image.cpu().numpy(), full_pred_dmap.cpu().numpy()


@torch.no_grad()
def predict(model, dataloader, thr, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        images = copy.deepcopy(inputs).expand(-1, 3, -1, -1)
        pad_value = dataloader.dataset.border_pad
        images = F.pad(images, pad_value)
        pred_dmaps = model(images.to(device))
        pred_dmaps = pred_dmaps[:, :, pad_value:pred_dmaps.shape[2] - pad_value,
                     pad_value:pred_dmaps.shape[3] - pad_value]

        return pred_dmaps

    ids = []
    results = []
    all_gt_and_preds = []
    for image_id, image, dmap in process_per_patch(dataloader, process_fn, cfg):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            io.imsave(outdir / f'dmap_{image_id}', (normalize(dmap)).astype(np.uint8))

        # TODO
        # to be decide how to evaluate


def main(args):
    run_path = Path(args.run)
    cfg_path = run_path / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)['technique']

    device = torch.device(args.device)

    # create test dataset and dataloader
    params = cfg.dataset.validation.params
    params.root = args.data_root
    params.split = 'all'
    # params.overlap = params.patch_size / 2
    test_batch_size = cfg.train.val_batch_size if cfg.train.val_batch_size else cfg.train.batch_size
    test_transform = ToTensor()
    test_dataset = PerineuralNetsDMapDataset(transforms=test_transform, with_targets=False, **params)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    print(f"Found {len(test_dataset)} samples in validation dataset")

    # Creating model
    print(f"Creating model")
    cfg.model.params.load_weights = True
    model = hydra.utils.get_class(f"models.{cfg.model.name}.{cfg.model.name}")
    model = model(**cfg.model.params)

    # Putting model to device
    model.to(device)

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    ckpt_path = max(best_models_folder.glob('*.pth'), key=lambda x: x.stat().st_mtime)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    validation_metrics = checkpoint['metrics']

    thr = validation_metrics['count/game-3_best_thr']
    outdir = (run_path / 'test_predictions') if args.save else None
    predict(model, test_loader, thr, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--data-root', default='data/perineuronal_nets_test', help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
