# -*- coding: utf-8 -*-
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
from skimage import io
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.ops import boxes as box_ops
from torchvision.transforms import ToTensor

import hydra
from omegaconf import OmegaConf

from datasets import PerineuralNetsDataset
from points.match import match
from points.metrics import detection_and_counting
from points.utils import draw_groundtruth_and_predictions


def process_per_patch(dataloader, process_fn, cfg, threshold):
    validation_device = cfg.optim.val_device

    def _process_batch(batch):
        patch, patch_hw, start_yx, image_hw, image_id = batch
        predictions = process_fn(patch)

        patch = patch.squeeze(dim=1).to(validation_device)
        predictions_bbs = [p['boxes'].to(validation_device) for p in predictions]
        predictions_scores = [p['scores'].to(validation_device) for p in predictions]

        processed_batch = (image_id, image_hw, patch, predictions_bbs, predictions_scores, patch_hw, start_yx)
        return processed_batch

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
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        boxes = []
        scores = []

        for _, _, patch, patch_boxes, patch_scores, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y + h, x:x + w] = patch[:h, :w]
            normalization_map[y:y + h, x:x + w] += 1.0
            if patch_boxes.nelement() != 0:
                patch_boxes += torch.as_tensor([x, y, x, y])
                boxes.append(patch_boxes)
                scores.append(patch_scores)

        full_image = full_image.cpu().numpy()
        boxes = torch.cat(boxes) if len(boxes) else torch.empty(0, 4, dtype=torch.float32)
        scores = torch.cat(scores) if len(scores) else torch.empty(0, dtype=torch.float32)

        progress.set_description('PRED (cleaning)')
        # remove boxes with center outside the image     
        image_wh = torch.tensor(image_hw[::-1])
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        boxes_center = boxes_center.round().long()
        keep = (boxes_center < image_wh).all(axis=1)

        boxes = boxes[keep]
        scores = scores[keep]
        boxes_center = boxes_center[keep]  # we need those later

        # clip boxes to image limits
        ih, iw = image_hw
        l = torch.tensor([[0, 0, 0, 0]])
        u = torch.tensor([[iw, ih, iw, ih]])   
        boxes = torch.max(l, torch.min(boxes, u))

        # filter boxes in the overlapped areas using nms
        xc, yc = boxes_center.T
        in_overlap_zone = normalization_map[yc, xc] != 1.0

        boxes_in_overlap = boxes[in_overlap_zone]
        scores_in_overlap = scores[in_overlap_zone]
        keep = box_ops.nms(boxes_in_overlap, scores_in_overlap, iou_threshold=cfg.model.params.nms)

        boxes = torch.cat((boxes[~in_overlap_zone], boxes_in_overlap[keep]))
        scores = torch.cat((scores[~in_overlap_zone], scores_in_overlap[keep]))

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        # cleaning
        del normalization_map

        yield image_id, full_image, boxes, scores


@torch.no_grad()
def predict(model, dataloader, min_thr, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        inputs = list(i.to(device) for i in inputs)
        return model(inputs)

    all_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]
    for image_id, image, boxes, scores in process_per_patch(dataloader, process_fn, cfg, min_thr):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)

        localizations = (boxes[:, :2] + boxes[:, 2:]) / 2
        localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
        localizations['score'] = scores

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')

            thresholded_localizations = localizations[localizations.score >= thr].reset_index()

            # match groundtruths and predictions
            tolerance = 1.25 * (cfg.dataset.validation.params.target_params.side / 2)  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, thresholded_localizations, tolerance)

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

            image = draw_groundtruth_and_predictions(image, gp, radius=10, marker='square')
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
    model_params = cfg.model.params
    del model_params['cache_folder']  # inhibits hydra-specific interpolation
    model = hydra.utils.get_class(f"models.{cfg.model.name}")
    model = model(skip_weights_loading=True, **model_params)

    # move model to device
    model.to(device)

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_PerineuralNetsDataset_metric_{metric_name}.pth'
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    minimum_thr = args.minimum_thr
    print(f"Setting minimum threshold: {minimum_thr}")
    outdir = (run_path / f'test_predictions') if args.save else None
    predict(model, test_loader, minimum_thr, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--best-on-metric', default='count/game-3_best', help='select snapshot that optimizes this metric')
    parser.add_argument('--minimum-thr', default=0.05, type=float, help='minimum threshold for bbs filtering')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--data-root', default='data/perineuronal_nets_test', help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
