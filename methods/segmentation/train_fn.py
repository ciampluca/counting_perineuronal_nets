import numpy as np
import pandas as pd
from skimage import io
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..points.metrics import detection_and_counting
from ..points.match import match
from ..points.utils import draw_groundtruth_and_predictions
from .metrics import dice_jaccard
from .utils import segmentation_map_to_points


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
    n_images = dataloader.dataset.num_images()

    @torch.no_grad()
    def process_fn(batch):
        input_and_target = batch.to(device)

        # split channels to get input, target, and loss weights
        images, targets, weights = input_and_target.split(1, dim=1)
        logits = model(images)
        predictions = torch.sigmoid(logits)

        processed_batch = (images, targets, weights, predictions)
        # remove single channel dimension, move to validation device
        processed_batch = [x[:, 0].to(validation_device) for x in processed_batch]
        return processed_batch

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info

        # image = torch.empty(image_hw, dtype=torch.float32, device=validation_device)
        segmentation_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        target_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        weights_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)
        normalization_map = torch.zeros(image_hw, dtype=torch.float32, device=validation_device)

        # build full map from patches
        for (patch_hw, start_yx), (patch, target, weights, prediction) in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            # image[y:y+h, x:x+w] = patch[:h, :w]
            segmentation_map[y:y+h, x:x+w] += prediction[:h, :w]
            target_map[y:y+h, x:x+w] += target[:h, :w]
            weights_map[y:y+h, x:x+w] += weights[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0
            
        # image /= normalization_map
        segmentation_map /= normalization_map
        segmentation_map = torch.clamp(segmentation_map, 0, 1)  # XXX to fix, sporadically something goes off limits

        target_map /= normalization_map
        weights_map /= normalization_map
        del normalization_map

        return image_id, image_hw, segmentation_map, target_map, weights_map

    metrics = []
    thr_metrics = []

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image_hw, segmentation_map, target_map, weights_map in progress:
        # compute metrics
        progress.set_description('EVAL (metrics)')

        ## threshold-free metrics
        weighted_bce_loss = F.binary_cross_entropy(segmentation_map, target_map, weights_map)
        soft_dice, soft_jaccard = dice_jaccard(target_map, segmentation_map)
        
        ## threshold-dependent metrics
        thrs = torch.linspace(0, 1, 21).tolist() + [2,]

        image_thr_metrics = []
        progress_thrs = tqdm(thrs, desc='thr', leave=False)
        for thr in progress_thrs:
            binary_segmentation_map = segmentation_map >= thr

            # segmentation metrics
            progress_thrs.set_description(f'thr={thr:.2f} (segm)')
            dice, jaccard = dice_jaccard(target_map, binary_segmentation_map)

            segm_metrics = {
                'segm/dice': dice.item(),
                'segm/jaccard': jaccard.item()
            }
        
            # counting metrics
            progress_thrs.set_description(f'thr={thr:.2f} (count)')
            localizations = segmentation_map_to_points(segmentation_map.cpu().numpy(), thr=thr)
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
        
        progress.set_description('EVAL (patches)')

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


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=False):
    """ Make predictions on data. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        logits = model(inputs.to(device))
        predictions = torch.sigmoid(logits).squeeze(axis=1)
        predictions = predictions.cpu()
        return inputs.squeeze(axis=1).numpy(), predictions.numpy()

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info
        full_input = np.empty(image_hw, dtype=np.float32)
        full_output = np.zeros(image_hw, dtype=np.float32)
        normalization_map = np.zeros(image_hw, dtype=np.float32)

        for (patch_hw, start_yx), (patch, processed_patch) in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            full_input[y:y+h, x:x+w] = patch[:h, :w]
            full_output[y:y+h, x:x+w] += processed_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        full_output /= normalization_map
        del normalization_map

        return image_id, full_input, full_output

    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)

    all_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]
    for image_id, image, segmentation_map in progress:
        image_hw = image.shape
        image = (255 * image).astype(np.uint8)

        groundtruth = dataloader.dataset.annot.loc[image_id]
        groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            io.imsave(outdir / f'segm_{image_id}', (255 * segmentation_map).astype(np.uint8))
            # io.imsave(outdir / f'hard_segm_{image_id}', (255 * (segmentation_map >= 0.5)).astype(np.uint8))

        image_metrics = []
        image_gt_and_preds = []
        thr_progress = tqdm(thrs, leave=False)
        for thr in thr_progress:
            thr_progress.set_description(f'thr={thr:.2f}')

            # find connected components and centroids
            localizations = segmentation_map_to_points(segmentation_map, thr=thr)

            # match groundtruths and predictions
            tolerance = 1.25 * cfg.data.validation.target_params.radius  # min distance to match points
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


@torch.no_grad()
def predict_points(dataloader, model, device, threshold, cfg):
    """ Predict and find points. """
    model.eval()

    @torch.no_grad()
    def process_fn(inputs):
        logits = model(inputs.to(device))
        predictions = torch.sigmoid(logits).squeeze(axis=1).cpu().numpy()
        return (predictions,)

    def collate_fn(image_info, image_patches):
        image_id, image_hw = image_info
        segmentation_map = np.zeros(image_hw, dtype=np.float32)
        normalization_map = np.zeros(image_hw, dtype=np.float32)

        for (patch_hw, start_yx), (processed_patch,) in image_patches:
            (y, x), (h, w) = start_yx, patch_hw
            segmentation_map[y:y+h, x:x+w] += processed_patch[:h, :w]
            normalization_map[y:y+h, x:x+w] += 1.0

        segmentation_map /= normalization_map
        del normalization_map

        return image_id, segmentation_map

    all_localizations = []
    
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, progress=True)
    n_images = dataloader.dataset.num_images()
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, segmentation_map in progress:
        # find connected components and centroids
        localizations = segmentation_map_to_points(segmentation_map, thr=threshold)
        localizations['imgName'] = image_id
        localizations['thr'] = threshold
        all_localizations.append(localizations)

    all_localizations = pd.concat(all_localizations, ignore_index=True)
    return all_localizations