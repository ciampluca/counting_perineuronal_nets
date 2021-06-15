import itertools

import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
import torch
import torch.nn.functional as F
from tqdm import tqdm

from points.metrics import detection_and_counting
from points.match import match
from segmentation.metrics import dice_jaccard
from segmentation.utils import segmentation_map_to_points

from .metrics import dice_jaccard
from utils import unbatch


def train_one_epoch(dataloader, model, optimizer, device, writer, epoch, cfg):
    """ Trains the model for one epoch. """
    model.train()
    optimizer.zero_grad()

    metrics = []
    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target = sample[0]
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, targets, weights = input_and_target.split(1, dim=1)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weights)
        predictions = torch.sigmoid(logits)
        soft_dice, soft_jaccard = dice_jaccard(targets, predictions)

        loss.backward()

        batch_metrics = {
            'loss': loss.item(),
            'soft_dice': soft_dice.item(),
            'soft_jaccard': soft_jaccard.item()
        }

        metrics.append(batch_metrics)

        postfix = {metric: f'{value:.3f}' for metric, value in batch_metrics.items()}
        progress.set_postfix(postfix)

        if (i + 1) % cfg.optim.batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % cfg.optim.log_every == 0:
            batch_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            n_iter = epoch * n_batches + i
            for metric, value in batch_metrics.items():
                writer.add_scalar(f'train/{metric}', value, n_iter)

            if cfg.optim.debug and (i + 1) % cfg.optim.debug_freq == 0:
                writer.add_images('train/inputs', images, n_iter)
                writer.add_images('train/targets', targets, n_iter)
                writer.add_images('train/predictions', predictions, n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device

    @torch.no_grad()
    def _predict(batch):
        input_and_target, patch_hw, start_yx, image_hw, image_id = batch
        input_and_target = input_and_target.to(device)

        # split channels to get input, target, and loss weights
        images, targets, weights = input_and_target.split(1, dim=1)
        logits = model(images)
        predictions = torch.sigmoid(logits)

        # remove single channel dimension, move to validation device
        images, targets, weights, predictions = map(lambda x: x[:, 0].to(validation_device), (images, targets, weights, predictions))

        processed_batch = (image_id, image_hw, images, targets, weights, predictions, patch_hw, start_yx)
        return processed_batch

    processed_batches = map(_predict, dataloader)
    processed_batches = BackgroundGenerator(processed_batches, max_prefetch=7500)  # prefetch batches using threading
    processed_samples = unbatch(processed_batches)

    metrics = []
    thr_metrics = []
    
    # group by image_id (and image_hw for convenience) --> iterate over full images
    grouper = lambda x: (x[0], x[1].tolist())  # group by (image_id, image_hw)
    groups = itertools.groupby(processed_samples, key=grouper)

    n_images = len(dataloader.dataset)
    if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset):
        n_images = len(dataloader.dataset.datasets)

    progress = tqdm(groups, total=n_images, desc='EVAL', leave=False)
    for (image_id, image_hw), image_patches in progress:
        # full_image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        full_segm_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_target_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        full_weight_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        # build full maps from patches
        progress.set_description('EVAL (patches)')
        for _, _, patch, target, weights, prediction, patch_hw, start_yx in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            # full_image[y:y+h, x:x+w] = patch[:h, :w]
            full_segm_map[y:y+h, x:x+w] += prediction[:h, :w]
            full_target_map[y:y+h, x:x+w] += target[:h, :w]
            full_weight_map[y:y+h, x:x+w] += weights[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        # full_image /= normalization_map
        full_segm_map /= normalization_map
        full_target_map /= normalization_map
        full_weight_map /= normalization_map

        full_segm_map = torch.clamp(full_segm_map, 0, 1)  # XXX to fix, sporadically something goes off limits
        del normalization_map

        # compute metrics
        progress.set_description('EVAL (metrics)')

        ## threshold-free metrics
        weighted_bce_loss = F.binary_cross_entropy(full_segm_map, full_target_map, full_weight_map)
        soft_dice, soft_jaccard = dice_jaccard(full_target_map, full_segm_map)
        
        ## threshold-dependent metrics
        thrs = torch.linspace(0, 1, 21).tolist() + [2,]

        image_thr_metrics = []
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        for thr in progress_thrs:
            full_bin_segm_map = full_segm_map >= thr

            # segmentation metrics
            progress_thrs.set_description(f'thr={thr:.2f} (segm)')
            dice, jaccard = dice_jaccard(full_target_map, full_bin_segm_map)

            segm_metrics = {
                'segm/dice': dice.item(),
                'segm/jaccard': jaccard.item()
            }
        
            # counting metrics
            progress_thrs.set_description(f'thr={thr:.2f} (count)')
            localizations = segmentation_map_to_points(full_segm_map.cpu().numpy(), thr=thr)
            groundtruth = dataloader.dataset.annot.loc[image_id]

            tolerance = 1.25 * cfg.data.validation.target_params.radius  # min distance to match points
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            count_pdet_metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)

            image_thr_metrics.append({
                'image_id': image_id,
                'thr': thr,
                **segm_metrics,
                **count_pdet_metrics
            })
        
        pr = pd.DataFrame(image_thr_metrics).sort_values('pdet/recall', ascending=False)
        recalls = pr['pdet/recall'].values
        precisions = pr['pdet/precision'].values
        average_precision = - np.sum(np.diff(recalls) * precisions[:-1])  # sklearn's ap

        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'segm/weighted_bce_loss': weighted_bce_loss.item(),
            'segm/soft_dice': soft_dice.item(),
            'segm/soft_jaccard': soft_jaccard.item(),
            'pdet/average_precision': average_precision.item(),
        })

        thr_metrics.extend(image_thr_metrics)

    # average among images
    metrics = pd.DataFrame(metrics).set_index('image_id')
    metrics = metrics.mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}

    # pick best threshold metrics
    thr_metrics = pd.DataFrame(thr_metrics).set_index(['image_id', 'thr'])
    mean_thr_metrics = thr_metrics.pivot_table(index='thr', values=thr_metrics.columns, aggfunc='mean')
    
    best_thr_metrics = mean_thr_metrics.aggregate({
        'segm/dice': max,
        'segm/jaccard': max,
        'pdet/precision': max,
        'pdet/recall': max,
        'pdet/f1_score': max,
        'count/err': lambda x: min(x, key=abs),
        'count/mae': min,
        'count/mse': min,
        'count/mare': min,
        **{f'count/game-{l}': min for l in range(6)}
    }).to_dict()

    best_thrs = mean_thr_metrics.aggregate({
        'segm/dice': 'idxmax',
        'segm/jaccard': 'idxmax',
        'pdet/precision': 'idxmax',
        'pdet/recall': 'idxmax',
        'pdet/f1_score': 'idxmax',
        'count/err': lambda x: x.abs().idxmin(),
        'count/mae': 'idxmin',
        'count/mse': 'idxmin',
        'count/mare': 'idxmin',
        **{f'count/game-{l}': 'idxmin' for l in range(6)}
    }).to_dict()

    best_thr_metrics = {k: {'value': v, 'threshold': best_thrs[k]} for k, v in best_thr_metrics.items()}
    metrics.update(best_thr_metrics)

    return metrics

