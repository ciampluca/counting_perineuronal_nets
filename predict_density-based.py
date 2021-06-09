# -*- coding: utf-8 -*-
import argparse
from datasets import PerineuralNetsDataset
import itertools

from math import floor
from pathlib import Path
from points.utils import draw_groundtruth_and_predictions

import cv2
import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
from skimage import io, draw
from skimage.feature import peak_local_max
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.functional as F

import hydra
from omegaconf import OmegaConf

from density.target_builder import DensityTargetBuilder
from density.metrics import counting, counting_yx
from points.match import match
from points.metrics import detection_and_counting

# some colors
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
CYAN = [0, 255, 255]
MAGENTA = [255, 0, 255]


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

    n_images = len(dataloader.dataset)
    if isinstance(dataloader.dataset, ConcatDataset):
        n_images = len(dataloader.dataset.datasets)

    progress = tqdm(groups, total=n_images, desc='PRED', leave=False)
    for (image_id, image_hw), image_patches in progress:
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        full_pred_dmap = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        for _, _, patch, pred_dmap, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y+h, x:x+w] = patch[:h, :w]
            full_pred_dmap[y:y+h, x:x+w] += pred_dmap[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        full_pred_dmap /= normalization_map

        # Cleaning
        del normalization_map

        yield image_id, full_image.cpu().numpy(), full_pred_dmap.cpu().numpy()


@torch.no_grad()
def predict(model, dataloader, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()
    pad = cfg.train.border_pad
    density_map_builder = DensityTargetBuilder(**cfg.dataset.validation.params.target_params)

    @torch.no_grad()
    def process_fn(inputs):
        images = inputs.expand(-1, 3, -1, -1)
        padded_images = F.pad(images, pad)
        pred_dmaps = model(padded_images.to(device))
        h, w = pred_dmaps.shape[2:]
        pred_dmaps = pred_dmaps[:, :, pad:(h - pad), pad:(w - pad)]
        return pred_dmaps

    all_metrics = []
    all_yx_metrics = []
    all_dmap_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist()
    for image_id, image, dmap in process_per_patch(dataloader, process_fn, cfg):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            dmin, dmax = dmap.min(), dmap.max()
            normalized_dmap = (dmap - dmin) / (dmax - dmin)
            normalized_dmap = (255 * normalized_dmap).astype(np.uint8)
            io.imsave(outdir / f'dmap_{image_id}', normalized_dmap)
            del normalized_dmap

        # compute dmap metrics (no thresholding or peak finding)
        gt_points = groundtruth[['Y', 'X']].values
        gt_dmap = density_map_builder.build(image, gt_points)
        dmap_metrics = counting(gt_dmap, dmap)
        dmap_metrics['imgName'] = image_id
        all_dmap_metrics.append(dmap_metrics)

        yx_metrics = counting_yx(gt_points, dmap)
        yx_metrics['imgName'] = image_id
        all_yx_metrics.append(yx_metrics)

        pred_num = np.sum(dmap)

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')

            peak_idx = peak_local_max(
                dmap,
                num_peaks=int(pred_num),
                threshold_abs=0.0,
                min_distance=int(cfg.dataset.validation.params.target_params.sigma),
                exclude_border=int(cfg.dataset.validation.params.target_params.sigma),
                threshold_rel=thr,
            )

            localizations = pd.DataFrame(peak_idx, columns=['Y', 'X'])
            localizations['score'] = dmap[tuple(peak_idx.T)]

            # match groundtruths and predictions
            tolerance = 1.25 * cfg.dataset.validation.params.target_params.sigma  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)

            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr
            image_gt_and_preds.append(groundtruth_and_predictions)

            metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            metrics['imgName'] = image_id
            metrics['thr'] = thr

            image_metrics.append(metrics)
        
        all_metrics.extend(image_metrics)
        all_gt_and_preds.extend(image_gt_and_preds)
        
        if outdir and debug:
            outdir.mkdir(parents=True, exist_ok=True)

            # pick a threshold and draw that prediction set
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3'].idxmin()
            gp = pd.concat(image_gt_and_preds, ignore_index=True)
            gp = gp[gp.thr == best_thr]

            image = draw_groundtruth_and_predictions(image, gp, radius=10, marker='circle')
            io.imsave(outdir / f'annot_{image_id}', image)

    all_metrics = pd.DataFrame(all_metrics)
    all_yx_metrics = pd.DataFrame(all_yx_metrics)
    all_dmap_metrics = pd.DataFrame(all_dmap_metrics)
    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_metrics.to_csv(outdir / 'all_metrics.csv')
        all_yx_metrics.to_csv(outdir / 'yx_metrics.csv')
        all_dmap_metrics.to_csv(outdir / 'dmap_metrics.csv')
        all_gp.to_csv(outdir / 'all_gt_preds.csv')


def main(args):
    run_path = Path(args.run)
    cfg_path = run_path / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)['technique']

    device = torch.device(args.device)

    # create test dataset and dataloader
    params = cfg.dataset.validation.params
    params.root = args.data_root
    params.split = 'all'
    params.target = None
    
    test_batch_size = cfg.train.val_batch_size if cfg.train.val_batch_size else cfg.train.batch_size
    test_transform = ToTensor()
    test_dataset = PerineuralNetsDataset(transforms=test_transform, **params)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    print(f"Found {len(test_dataset)} samples in validation dataset")

    # Creating model
    print(f"Creating model")
    model = hydra.utils.get_class(f"models.{cfg.model.name}.{cfg.model.name}")
    model = model(skip_weights_loading=True, **cfg.model.params)

    # Putting model to device
    model.to(device)

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_PerineuralNetsDataset_metric_{metric_name}.pth'
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    outdir = (run_path / 'test_predictions') if args.save else None
    predict(model, test_loader, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--best-on-metric', default='count/game-3', help='select snapshot that optimizes this metric')
    parser.add_argument('--data-root', default='data/perineuronal_nets_test', help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
