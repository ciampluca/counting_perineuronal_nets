# -*- coding: utf-8 -*-
import argparse

import itertools
from prefetch_generator import BackgroundGenerator
from pathlib import Path
from skimage import io
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import ToTensor

from omegaconf import OmegaConf
import hydra

from datasets import PerineuralNetsDataset
from points.match import match
from points.metrics import detection_and_counting
from points.utils import draw_groundtruth_and_predictions
from segmentation.utils import segm_map_to_points


# some colors
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
CYAN = [0, 255, 255]
MAGENTA = [255, 0, 255]


def process_per_patch(dataloader, process_fn):

    def _process_batch(batch):
        patch, patch_hw, start_yx, image_hw, image_id = batch
        processed_patch = process_fn(patch)
        batch = image_id, image_hw, patch.squeeze(axis=1), processed_patch, patch_hw, start_yx
        batch = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in batch]
        return batch
    
    def _unbatch(batches):
        for batch in batches:
            yield from zip(*batch)

    processed_batches = map(_process_batch, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=15000)  # prefetch batches using threading 
    processed_samples = _unbatch(processed_batches)

    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)

    n_images = len(dataloader.dataset)
    if isinstance(dataloader.dataset, ConcatDataset):
        n_images = len(dataloader.dataset.datasets)

    progress = tqdm(groups, total=n_images, desc='PRED', leave=False)
    for (image_id, image_hw), image_patches in progress:
        full_input = np.empty(image_hw, dtype=np.float32)
        full_output = np.zeros(image_hw, dtype=np.float32)
        normalization_map = np.zeros(image_hw, dtype=np.float32)

        for _, _, patch, processed_patch, patch_hw, start_yx in tqdm(image_patches):
            (y, x), (h, w) = start_yx, patch_hw
            full_input[y:y+h, x:x+w] = patch[:h, :w]
            full_output[y:y+h, x:x+w] += processed_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        full_output /= normalization_map
        del normalization_map

        yield image_id, full_input, full_output


@torch.no_grad()
def predict(model, dataloader, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        logits = model(inputs.to(device))
        return torch.sigmoid(logits).cpu().squeeze(axis=1)

    all_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]
    for image_id, image, segmentation_map in process_per_patch(dataloader, process_fn):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            io.imsave(outdir / f'segm_{image_id}', (255 * segmentation_map).astype(np.uint8))
            # io.imsave(outdir / f'hard_segm_{image_id}', (255 * (segmentation_map >= 0.5)).astype(np.uint8))
            # skimage.measure.find_contours.find_contours(array, level, fully_connected='low', positive_orientation='low')

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')
            # find connected components and centroids
            localizations = segm_map_to_points(segmentation_map, thr=thr)
            
            # match groundtruths and predictions
            tolerance = 1.25 * cfg.dataset.validation.params.target_params.radius  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr
            image_gt_and_preds.append(groundtruth_and_predictions)

            # compute metrics
            metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            metrics['thr'] = thr
            metrics['imgName'] = image_id
        
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
    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_gp.to_csv(outdir / 'all_gt_preds.csv')
        all_metrics.to_csv(outdir / 'all_metrics.csv')

            
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
    
    test_batch_size = cfg.optim.val_batch_size if cfg.optim.val_batch_size else cfg.optim.batch_size
    test_transform = ToTensor()
    test_dataset = PerineuralNetsDataset(transforms=test_transform, **params)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.optim.num_workers)
    print(f"Found {len(test_dataset)} samples in validation dataset")

    # create model
    model = hydra.utils.get_class(f"models.{cfg.model.name}")
    model = model(skip_weights_loading=True, **cfg.model.params)

    # move model to device
    model.to(device)

    # optionally resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_PerineuralNetsDataset_metric_{metric_name}.pth'
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    validation_metrics = checkpoint['metrics']

    validation_thr = validation_metrics[f'{args.best_on_metric}_thr']
    outdir = (run_path / 'test_predictions') if args.save else None
    predict(model, test_loader, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--best-on-metric', default='count/game-3_best', help='select snapshot that optimizes this metric')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--data-root', default='data/perineuronal_nets_test', help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)

