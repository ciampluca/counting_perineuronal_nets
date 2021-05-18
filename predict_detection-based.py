# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops
from skimage import io, draw

from omegaconf import OmegaConf

from prefetch_generator import BackgroundGenerator

from datasets.PerineuralNetsDetDataset import PerineuralNetsDetDataset
from datasets.det_transforms import ToTensor
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn
from utils import points

# some colors
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
CYAN = [0, 255, 255]
MAGENTA = [255, 0, 255]


def get_model_detection(num_classes, cfg, load_custom_model=False):
    if cfg.model.backbone not in ["resnet50", "resnet101", "resnet152"]:
        print(f"Backbone not supported")
        exit(1)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes += 1    # num classes + background

    if load_custom_model:
        model_pretrained = False
        backbone_pretrained = False
    else:
        model_pretrained = cfg.model.coco_model_pretrained
        backbone_pretrained = cfg.model.backbone_pretrained

    # anchor generator: these are default values, but maybe we have to change them
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    # these are default values, but maybe we can change them
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2)

    # Creating model
    if cfg.model.backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg.model.params.max_dets_per_image,
            box_nms_thresh=cfg.model.params.nms,
            box_score_thresh=cfg.model.params.det_thresh,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg.model.backbone == "resnet101":
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg.model.params.max_dets_per_image,
            box_nms_thresh=cfg.model.params.nms,
            box_score_thresh=cfg.model.params.det_thresh,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )
    elif cfg.model.backbone == "resnet152":
        print(f"Model with ResNet152 to be implemented")
        exit(1)
    else:
        print(f"Not supported backbone")
        exit(1)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def process_per_patch(dataloader, process_fn, cfg, threshold):
    validation_device = cfg.train.val_device

    def _process_batch(batch):
        patch, patch_hw, start_yx, image_hw, image_id = batch

        predictions = process_fn(patch)

        patch = patch.squeeze(dim=1).to(validation_device)
        predictions_bbs = [p['boxes'].to(validation_device) for p in predictions]
        predictions_scores = [p['scores'].to(validation_device) for p in predictions]

        processed_batch = (image_id, image_hw, patch, predictions_bbs, predictions_scores, patch_hw, start_yx)

        return processed_batch

    processed_batches = map(_process_batch, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=5000)  # prefetch batches using threading

    def _unbatch(batches):
        for batch in batches:
            yield from zip(*batch)

    processed_samples = _unbatch(processed_batches)

    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)
    groups = tqdm(groups, total=len(dataloader.dataset.datasets), desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in groups:
        full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_image_det_bbs = torch.empty(0, 4, dtype=torch.float32)
        full_image_det_scores = torch.empty(0, dtype=torch.float32)

        for _, _, patch, prediction_bbs, prediction_scores, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_image[y:y + h, x:x + w] = patch[:h, :w]
            normalization_map[y:y + h, x:x + w] += 1.0
            if prediction_bbs.nelement() != 0:
                prediction_bbs[:, 0:1] += x
                prediction_bbs[:, 2:3] += x
                prediction_bbs[:, 1:2] += y
                prediction_bbs[:, 3:4] += y
                full_image_det_bbs = torch.cat((full_image_det_bbs, prediction_bbs))
                full_image_det_scores = torch.cat((full_image_det_scores, prediction_scores))

        # Removing bbs outside image and clipping; filtering bbs below threshold score
        full_image_filtered_det_bbs = torch.empty(0, 4, dtype=torch.float32)
        full_image_filtered_det_scores = torch.empty(0, dtype=torch.float32)
        l = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # Setting the lower and upper bound per column
        u = torch.tensor([[image_hw[1], image_hw[0], image_hw[1], image_hw[0]]])
        for bb, score in zip(full_image_det_bbs, full_image_det_scores):
            if score < threshold:
                continue
            bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
            x_c, y_c = int(bb[0] + (bb_w / 2)), int(bb[1] + (bb_h / 2))
            if x_c > image_hw[1] or y_c > image_hw[0]:
                continue
            bb = torch.max(torch.min(bb, u), l)
            full_image_filtered_det_bbs = torch.cat((full_image_filtered_det_bbs, bb))
            full_image_filtered_det_scores = torch.cat(
                (full_image_filtered_det_scores, torch.Tensor([score.item()])))

        # Performing filtering of the bbs in the overlapped areas using nms
        in_overlap_areas_indices = []
        in_overlap_areas_det_bbs, full_image_final_det_bbs = \
            torch.empty(0, 4, dtype=torch.float32), torch.empty(0, 4, dtype=torch.float32)
        in_overlap_areas_det_scores, full_image_final_det_scores = \
            torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
        for i, (det_bb, det_score) in enumerate(
                zip(full_image_filtered_det_bbs, full_image_filtered_det_scores)):
            bb_w, bb_h = det_bb[2] - det_bb[0], det_bb[3] - det_bb[1]
            x_c, y_c = int(det_bb[0] + (bb_w / 2)), int(det_bb[1] + (bb_h / 2))
            if normalization_map[y_c, x_c] != 1.0:
                in_overlap_areas_indices.append(i)
                in_overlap_areas_det_bbs = torch.cat(
                    (in_overlap_areas_det_bbs, torch.Tensor([det_bb.cpu().numpy()])))
                in_overlap_areas_det_scores = torch.cat(
                    (in_overlap_areas_det_scores, torch.Tensor([det_score.item()])))
            else:
                full_image_final_det_bbs = torch.cat(
                    (full_image_final_det_bbs, torch.Tensor([det_bb.cpu().numpy()])))
                full_image_final_det_scores = torch.cat(
                    (full_image_final_det_scores, torch.Tensor([det_score.item()])))

        # non-maximum suppression
        if in_overlap_areas_indices:
            keep_in_overlap_areas_indices = box_ops.nms(
                in_overlap_areas_det_bbs,
                in_overlap_areas_det_scores,
                iou_threshold=cfg.model.params.nms
            )

        if in_overlap_areas_indices:
            for i in keep_in_overlap_areas_indices:
                full_image_final_det_bbs = torch.cat(
                    (full_image_final_det_bbs, torch.Tensor([in_overlap_areas_det_bbs[i].cpu().numpy()])))
                full_image_final_det_scores = torch.cat(
                    (full_image_final_det_scores, torch.Tensor([in_overlap_areas_det_scores[i].item()])))

        # Cleaning
        del normalization_map

        yield image_id, full_image.cpu().numpy(), full_image_final_det_bbs.cpu().numpy(), full_image_final_det_scores.cpu().numpy()


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
    for image_id, image, bbs, scores in process_per_patch(dataloader, process_fn, cfg, min_thr):
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)

        localizations = (bbs[:, :2] + bbs[:, 2:]) / 2
        localizations = pd.DataFrame(localizations, columns=['X', 'Y'])
        localizations['score'] = scores

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')
            thresholded_localizations = localizations[localizations.score >= thr].reset_index()

            # match groundtruths and predictions
            tolerance = 1.25 * (cfg.dataset.validation.params.gt_params.side / 2)  # min distance to match points
            groundtruth_and_predictions = points.match(groundtruth, thresholded_localizations, tolerance)

            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr
            image_gt_and_preds.append(groundtruth_and_predictions)

            # compute metrics
            metrics = points.compute_metrics(groundtruth_and_predictions, image_hw=image_hw)
            metrics['imgName'] = image_id
            metrics['thr'] = thr
            image_metrics.append(metrics)

        all_metrics.extend(image_metrics)
        all_gt_and_preds.extend(image_gt_and_preds)

        if outdir and debug:
            outdir.mkdir(parents=True, exist_ok=True)
            image = np.stack((image, image, image), axis=-1)
            bb_half_side = 10

            # pick a threshold and draw that prediction set
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3'].idxmin()
            gp = pd.DataFrame(image_gt_and_preds)
            gp = gp[gp.thr == best_thr]

            # iterate gt and predictions
            for c_gt, r_gt, c_p, r_p, score, agreement in gp[['X', 'Y', 'Xp', 'Yp', 'score', 'agreement']].values:
                has_gt = not np.isnan(r_gt)
                has_p = not np.isnan(r_p)

                if has_gt:  # draw groundtruth
                    rs, cs = int(r_gt - bb_half_side), int(c_gt - bb_half_side)
                    re, ce = int(r_gt + bb_half_side), int(c_gt + bb_half_side)
                    rr, cc = draw.rectangle_perimeter(start=(rs, cs), end=(re, ce), shape=image.shape)
                    color = GREEN if has_p else CYAN
                    image[rr, cc] = color

                if has_p:  # draw prediction
                    rs, cs = int(r_p - bb_half_side), int(c_p - bb_half_side)
                    re, ce = int(r_p + bb_half_side), int(c_p + bb_half_side)
                    rr, cc = draw.rectangle_perimeter(start=(rs, cs), end=(re, ce), shape=image.shape)
                    color = RED if has_gt else MAGENTA
                    image[rr, cc] = color

                if has_gt and has_p:  # draw line between match
                    rr, cc, val = draw.line_aa(r_gt, c_gt, r_p, c_p)
                    draw.set_color(image, (rr, cc), YELLOW, alpha=val)

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
    # params.overlap = params.patch_size / 2
    test_batch_size = cfg.train.val_batch_size if cfg.train.val_batch_size else cfg.train.batch_size
    test_transform = ToTensor()
    test_dataset = PerineuralNetsDetDataset(transforms=test_transform, with_targets=False, **params)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    print(f"Found {len(test_dataset)} samples in validation dataset")

    # create model
    print(f"Creating model")
    model = get_model_detection(num_classes=1, cfg=cfg, load_custom_model=True)

    # move model to device
    model.to(device)

    # resume from a saved checkpoint
    best_models_folder = run_path / 'best_models'
    # ckpt_path = max(best_models_folder.glob('*.pth'), key=lambda x: x.stat().st_mtime)
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_perineural_nets_metric_{metric_name}.pth'
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    validation_metrics = checkpoint['metrics']

    validation_thr = validation_metrics[f'{args.best_on_metric}_thr']
    minimum_thr = args.minimum_thr
    print(f"Setting minimum threshold: {minimum_thr}")
    outdir = (run_path / f'test_predictions') if args.save else None
    predict(model, test_loader, minimum_thr, device, cfg, outdir, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('run', help='Path to run dir')
    parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction')
    parser.add_argument('--no-save', action='store_false', dest='save', help='draw images with predictions')
    parser.add_argument('--debug', action='store_true', default=False, help='draw images with predictions')
    parser.add_argument('--best-on-metric', default='count/game-3_best', help='select snapshot that optimizes this metric')
    parser.add_argument('--minimum-thr', default=0.05, type=float, help='minimum threshold for bbs filtering')
    parser.add_argument('--data-root', default='data/perineuronal_nets_test', help='root of the test subset')
    parser.set_defaults(save=True)

    args = parser.parse_args()
    main(args)
