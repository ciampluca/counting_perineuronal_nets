# -*- coding: utf-8 -*-
import os
import argparse

import collections
import itertools
from prefetch_generator import BackgroundGenerator
from pathlib import Path
from skimage import measure, io, draw
from tqdm import tqdm, trange

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from omegaconf import OmegaConf
import hydra

from datasets.perineural_nets_segm_dataset import PerineuralNetsSegmDataset
from utils import points

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
        batch = image_id, image_hw, patch.squeeze(), processed_patch, patch_hw, start_yx
        batch = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in batch]
        return batch
    
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
def predict(model, dataloader, thr, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        logits = model(inputs.to(device))
        return torch.sigmoid(logits).cpu().squeeze(axis=1)

    ids = []
    results = []
    all_gt_and_preds = []
    # thr_metrics = []
    # thrs = np.linspace(0, 1, 21).tolist() + [2]
    for image_id, image, segmentation_map in process_per_patch(dataloader, process_fn):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        # for t in tqdm(thrs):
        hard_segmentation_map = segmentation_map >= thr

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            io.imsave(outdir / f'segm_{image_id}', (255 * segmentation_map).astype(np.uint8))
            io.imsave(outdir / f'hard_segm_{image_id}', 255 * hard_segmentation_map.astype(np.uint8))
            # skimage.measure.find_contours.find_contours(array, level, fully_connected='low', positive_orientation='low')

        # find connected components and centroids
        labeled_map, num_components = measure.label(hard_segmentation_map, return_num=True, connectivity=1)
        localizations = measure.regionprops_table(labeled_map, properties=('centroid',))
        localizations = pd.DataFrame(localizations).rename({'centroid-0':'Y', 'centroid-1':'X'}, axis=1)
        
        # match groundtruths and predictions
        tolerance = 1.25 * cfg.dataset.validation.params.gt_params.radius  # min distance to match points
        groundtruth_and_predictions = points.match(groundtruth, localizations, tolerance)
        groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
        all_gt_and_preds.append(groundtruth_and_predictions)

        # filter by agreement
        selector = groundtruth_and_predictions.agreement.between(4, 7)  # select by agreement
        selector = selector | groundtruth_and_predictions.agreement.isna()  # always keep false positives
        groundtruth_and_predictions = groundtruth_and_predictions[selector]

        # compute metrics
        metrics = points.compute_metrics(groundtruth_and_predictions, image_hw=image_hw)
        metrics['thr'] = thr
        
        # thr_metrics.append(metrics)
        # thr_metrics = pd.DataFrame(thr_metrics)

        ids.append(image_id)
        results.append(metrics)

        if outdir and debug:
            outdir.mkdir(parents=True, exist_ok=True)
            image = np.stack((image, image, image), axis=-1)
            radius = 10

            # iterate gt and predictions
            for c_gt, r_gt, c_p, r_p in groundtruth_and_predictions[['X', 'Y', 'Xp', 'Yp']].values:
                has_gt = not np.isnan(r_gt)
                has_p = not np.isnan(r_p)
                
                if has_gt:  # draw groundtruth
                    r_gt, c_gt = int(r_gt), int(c_gt)
                    rr, cc, val = draw.circle_perimeter_aa(r_gt, c_gt, radius)
                    color = GREEN if has_p else CYAN
                    draw.set_color(image, (rr, cc), color, alpha=val)
                
                if has_p:  # draw prediction
                    r_p, c_p = int(r_p), int(c_p)
                    rr, cc, val = draw.circle_perimeter_aa(r_p, c_p, radius)
                    color = RED if has_gt else MAGENTA
                    draw.set_color(image, (rr, cc), color, alpha=val)
                
                if has_gt and has_p:  # draw line between match
                    rr, cc, val = draw.line_aa(r_gt, c_gt, r_p, c_p)
                    draw.set_color(image, (rr, cc), YELLOW, alpha=val)

            io.imsave(outdir / f'annot_{image_id}', image)

    results = pd.DataFrame(results, index=ids)
    print(results)
    print()
    print(results.describe().loc[['mean', 'std']].transpose())

    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    inA = ~all_gp.X.isna()
    inB = ~all_gp.Xp.isna()
    all_gp['tp'] = inA & inB
    all_gp['fp'] = ~inA & inB
    all_gp['fn'] = inA & ~inB
    all_gp['agreement'] = all_gp.agreement.map('{:g}'.format).replace('nan', 'none')
    by_agree = all_gp.pivot_table(index='agreement', values=['tp','fp','fn'], aggfunc='sum')
    by_agree['support'] = by_agree.sum(axis=1)
    by_agree['micro-tpr'] = by_agree.tp / (by_agree.tp + by_agree.fn)
    by_agree['fdr'] = by_agree.fp / (by_agree.fp + by_agree.tp.sum())

    # eye-candy
    table = by_agree.fillna(0).replace(0,'-')
    with pd.option_context('display.float_format', '{:.1%}'.format):
        print(table)
    
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        results.to_csv(outdir / 'results.csv')
        all_gp.to_csv(outdir / 'all_gt_preds.csv')
        table.to_latex(outdir / 'metrics_by_agreement.csv')

    import pdb; pdb.set_trace()

            
def main(args):
    run_path = Path(args.run)
    cfg_path = run_path / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)['technique']

    device = torch.device(args.device)
    
    # create test dataset and dataloader
    params = cfg.dataset.validation.params
    params.root = 'data/perineuronal_nets_test'
    params.split = 'all'
    params.overlap = params.patch_size / 2
    test_batch_size = cfg.optim.val_batch_size if cfg.optim.val_batch_size else cfg.optim.batch_size
    test_transform = ToTensor()
    test_dataset = PerineuralNetsSegmDataset(transforms=test_transform, with_targets=False, **params)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.optim.num_workers)
    print(f"Found {len(test_dataset)} samples in validation dataset")

    # create model
    model = hydra.utils.get_class(f"models.{cfg.model.name}.{cfg.model.name}")
    model = model(**cfg.model.params)

    # move model to device
    model.to(device)

    # optionally resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    ckpt_path = max(best_models_folder.glob('*.pth'), key=lambda x: x.stat().st_mtime)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    validation_metrics = checkpoint['metrics']

    thr = validation_metrics['segm/dice_best_thr']
    outdir = (run_path / 'test_predictions') if args.save else None
    predict(model, test_loader, thr, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)

