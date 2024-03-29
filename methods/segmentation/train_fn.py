from functools import partial
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..points.metrics import detection_and_counting, detection_average_precision
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
        n_channels = input_and_target.shape[1]
        x_channels = cfg.model.module.in_channels
        y_channels = (n_channels - x_channels) // 2
        images, targets, weights = input_and_target.split((x_channels, y_channels, y_channels), dim=1)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weights)
        predictions = torch.sigmoid(logits)

        # NCHW -> NHWC
        coefs = dice_jaccard(targets.movedim(1, -1), predictions.movedim(1, -1))

        loss.backward()

        batch_metrics = {
            'loss': loss.item(),
            'soft_dice': coefs['segm/dice/macro'],
            'soft_jaccard': coefs['segm/jaccard/macro'],
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
                for j in range(y_channels):
                    writer.add_images(f'train/targets_cls{j}', targets[:, j:j+1], n_iter)
                    writer.add_images(f'train/predictions_cls{j}', predictions[:, j:j+1], n_iter)

    metrics = pd.DataFrame(metrics).mean(axis=0).to_dict()
    metrics = {k: {'value': v, 'threshold': None} for k, v in metrics.items()}
    return metrics


def _save_image_and_segmentation_maps(image, image_id, segmentation_map, target_map, weights_map, cfg):
    import matplotlib

    debug_dir = Path('output_debug')
    debug_dir.mkdir(exist_ok=True)

    image_id = Path(image_id)

    def _scale_and_save(graymap, path):
        graymap = (255 * graymap.cpu().numpy().squeeze()).astype(np.uint8)
        pil_image = Image.fromarray(graymap).convert("RGB")
        pil_image.save(path)
        
    _scale_and_save(image, debug_dir / image_id)

    n_classes = cfg.model.module.out_channels
    for i in range(n_classes):
        _scale_and_save(segmentation_map[:, :, i], debug_dir / f'{image_id.stem}_segm_cls{i}.png')
        _scale_and_save(target_map[:, :, i], debug_dir / f'{image_id.stem}_target_cls{i}.png')

        wmap = weights_map[:, :, i].cpu().squeeze()
        wmap = matplotlib.cm.viridis(wmap)
        wmap = (255 * wmap).astype(np.uint8)
        pil_image = Image.fromarray(wmap).convert("RGB")
        pil_image.save(debug_dir / f'{image_id.stem}_weights_cls{i}.png')


@torch.no_grad()
def _process_fn(batch, model, device, cfg, has_target=False, return_input=False, return_target=False, return_numpy=False):
    batch = batch.to(device)

    if has_target:
        # split channels to get input, target, and loss weights
        n_channels = batch.shape[1]
        x_channels = cfg.model.module.in_channels
        y_channels = (n_channels - x_channels) // 2
        inputs, targets, weights = batch.split((x_channels, y_channels, y_channels), dim=1)
    else:
        inputs = batch

    logits = model(inputs)
    predictions = torch.sigmoid(logits)

    processed_batch = (predictions,)
    if return_target and has_target:
        processed_batch = (targets, weights) + processed_batch
    if return_input:
        processed_batch = (inputs,) + processed_batch
    
    # move channel dimension
    processed_batch = [x.movedim(1, -1) for x in processed_batch]

    if return_numpy:
        processed_batch = [x.cpu().numpy() for x in processed_batch]
        
    return processed_batch


@torch.no_grad()
def _collate_fn(image_info, image_patches, device, has_input=False, has_target=False):
    image_id, image_hw = image_info

    segmentation_map = None
    normalization_map = None

    # build full map from patches
    for (patch_hw, start_yx), datum in image_patches:

        if has_input and has_target:
            patch, target, weights, prediction = datum
        elif has_input:
            patch, prediction = datum
        elif has_target:
            target, weights, prediction = datum
        else:
            prediction = datum

        if segmentation_map is None:

            if isinstance(prediction, torch.Tensor):
                _zeros = partial(torch.zeros, dtype=torch.float32, device=device)
            else:
                _zeros = partial(np.zeros, dtype=np.float32)

            if has_input:
                in_channels = patch.shape[-1]
                in_hwc = image_hw + (in_channels,)
                image = _zeros(in_hwc)
            
            out_channels = prediction.shape[-1]
            out_hwc = image_hw + (out_channels,)

            if has_target:
                target_map = _zeros(out_hwc)
                weights_map = _zeros(out_hwc)

            segmentation_map = _zeros(out_hwc)
            normalization_map = _zeros(image_hw + (1,))
        
        (y, x), (h, w) = start_yx, patch_hw

        if has_input:
            image[y:y+h, x:x+w] = patch[:h, :w]

        if has_target:
            target_map[y:y+h, x:x+w] += target[:h, :w]
            weights_map[y:y+h, x:x+w] += weights[:h, :w]

        segmentation_map[y:y+h, x:x+w] += prediction[:h, :w]
        normalization_map[y:y+h, x:x+w] += 1.0
    
    if has_input:
        image /= normalization_map
    
    if has_target:
        target_map /= normalization_map
        weights_map /= normalization_map

    segmentation_map /= normalization_map
    # segmentation_map = torch.clamp(segmentation_map, 0, 1)  # XXX to fix, sporadically something goes off limits

    del normalization_map

    result = (segmentation_map,)
    if has_target:
        result = (target_map, weights_map) + result
    if has_input:
        result = (image,) + result
    
    result = (image_id, image_hw) + result
    return result


@torch.no_grad()
def validate(dataloader, model, device, epoch, cfg):
    """ Evaluate model on validation data. """
    model.eval()
    validation_device = cfg.optim.val_device
    n_images = dataloader.dataset.num_images()
    
    process_fn = partial(_process_fn, model=model, device=validation_device, cfg=cfg, has_target=True, return_input=True, return_target=True, return_numpy=False)
    collate_fn = partial(_collate_fn, device=device, has_input=True, has_target=True)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, max_prefetch=1)

    metrics = []
    thr_metrics = []

    progress = tqdm(processed_images, total=n_images, desc='EVAL (patches)', leave=False)
    for image_id, image_hw, image, target_map, weights_map, segmentation_map in progress:
        # compute metrics
        progress.set_description('EVAL (metrics)')

        if cfg.optim.debug and epoch % cfg.optim.debug == 0:
            _save_image_and_segmentation_maps(image, image_id, segmentation_map, target_map, weights_map, cfg)

        ## threshold-free metrics
        weighted_bce_loss = F.binary_cross_entropy(segmentation_map, target_map, weights_map)
        soft_segm_metrics = dice_jaccard(target_map, segmentation_map, prefix='soft_')
        
        ## threshold-dependent metrics
        image_thr_metrics = []

        groundtruth = dataloader.dataset.annot.loc[[image_id]]
        tolerance = 1.25 * cfg.data.validation.target_params.radius  # min distance to match points

        def _compute_thr_metrics(thr, segmentation_map, target_map, groundtruth, tolerance, image_hw):
            binary_segmentation_map = segmentation_map >= thr
            segm_metrics = dice_jaccard(target_map, binary_segmentation_map)
            localizations = segmentation_map_to_points(segmentation_map, thr=thr)
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            count_pdet_metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            
            return {
                'image_id': image_id,
                'thr': thr,
                **segm_metrics,
                **count_pdet_metrics
            }
        
        thrs = torch.linspace(0, 1, 21).tolist() + [2, ]
        job_args = (segmentation_map.cpu().numpy(), target_map.cpu().numpy(), groundtruth, tolerance, image_hw)
        jobs = [delayed(_compute_thr_metrics)(thr, *job_args) for thr in thrs]
        image_thr_metrics = Parallel(n_jobs=-1)(jobs)
        
        average_precisions = detection_average_precision(image_thr_metrics)
        
        # accumulate full image metrics
        metrics.append({
            'image_id': image_id,
            'segm/weighted_bce_loss': weighted_bce_loss.item(),
            **soft_segm_metrics,
            **average_precisions,
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
    
    # TODO factor out common code that follows in evaluate()s
    def _get_agg_func(metric_name, idx=False):
        if metric_name.startswith('count/err'):
            if idx:
                return lambda x: x.abs().idxmin()
            return lambda x: min(x, key=abs)
        
        if metric_name.startswith('count/'):
            return 'idxmin' if idx else min
        
        return 'idxmax' if idx else max

    value_aggfuncs = {k: _get_agg_func(k, idx=False) for k in thr_metrics.columns}
    thr_aggfuncs = {k: _get_agg_func(k, idx=True) for k in thr_metrics.columns}
    
    best_thr_metrics = mean_thr_metrics.aggregate(value_aggfuncs).to_dict()
    best_thrs = mean_thr_metrics.aggregate(thr_aggfuncs).to_dict()

    best_thr_metrics = {k: {'value': v, 'threshold': best_thrs[k]} for k, v in best_thr_metrics.items()}
    metrics.update(best_thr_metrics)

    return metrics


@torch.no_grad()
def predict(dataloader, model, device, cfg, outdir, debug=0):
    """ Make predictions on data. """
    model.eval()
    n_images = dataloader.dataset.num_images()
    
    process_fn = partial(_process_fn, model=model, device=device, cfg=cfg, has_target=False, return_input=True, return_target=False, return_numpy=True)
    collate_fn = partial(_collate_fn, device=device, has_input=True, has_target=False)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, max_prefetch=15000)

    all_metrics = []
    all_gt_and_preds = []
    thrs = np.linspace(0, 1, 201).tolist() + [2]

    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for n_i, (image_id, image_hw, image, segmentation_map) in enumerate(progress):
        image_hw = image_hw[:2]
        image = (255 * image).astype(np.uint8)

        annot = dataloader.dataset.annot
        groundtruth = annot.loc[[image_id]] if image_id in annot.index else pd.DataFrame([], columns=['X','Y'], dtype=np.int)
        if 'AV' in groundtruth.columns:  # for the PNN dataset only
            groundtruth['agreement'] = groundtruth.loc[:, 'AV':'VT'].sum(axis=1)

        if outdir and n_i < debug:  # debug
            outdir.mkdir(parents=True, exist_ok=True)
            io.imsave(outdir / image_id, image)
            n_classes = cfg.model.module.out_channels
            for i in range(n_classes):
                io.imsave(outdir / f'segm_cls{i}_{image_id}', (255 * segmentation_map[:,:,i]).astype(np.uint8))

        image_metrics = []
        image_gt_and_preds = []
        tolerance = 1.25 * cfg.data.validation.target_params.radius  # min distance to match points
        
        def _compute_thr_metrics(thr, segmentation_map, groundtruth, image_id, image_hw):
            # find connected components and centroids
            localizations = segmentation_map_to_points(segmentation_map, thr=thr)

            # match groundtruths and predictions
            groundtruth_and_predictions = match(groundtruth, localizations, tolerance)
            groundtruth_and_predictions['imgName'] = groundtruth_and_predictions.imgName.fillna(image_id)
            groundtruth_and_predictions['thr'] = thr

            # compute metrics
            metrics = detection_and_counting(groundtruth_and_predictions, image_hw=image_hw)
            metrics['thr'] = thr
            metrics['imgName'] = image_id

            return groundtruth_and_predictions, metrics

        job_args = (segmentation_map, groundtruth, image_id, image_hw)
        jobs = [delayed(_compute_thr_metrics)(thr, *job_args) for thr in thrs]
        all_gp_and_metrics = Parallel(n_jobs=-1)(jobs)
        image_gt_and_preds, image_metrics = zip(*all_gp_and_metrics)

        all_metrics.extend(image_metrics)
        all_gt_and_preds.extend(image_gt_and_preds)

        if outdir and n_i < debug:
            outdir.mkdir(parents=True, exist_ok=True)

            # pick a threshold and draw that prediction set
            best_thr = pd.DataFrame(image_metrics).set_index('thr')['count/game-3/macro'].idxmin()
            gp = pd.concat(image_gt_and_preds, ignore_index=True)
            gp = gp[gp.thr == best_thr]

            n_classes = cfg.model.module.out_channels
            radius = cfg.data.validation.target_params.radius
            for i in range(n_classes):
                gp_i = gp[gp['class'] == i]
                image = draw_groundtruth_and_predictions(image, gp_i, radius=radius, marker='circle')
                io.imsave(outdir / f'annot_{image_id}', image)

    all_metrics = pd.DataFrame(all_metrics)
    all_gp = pd.concat(all_gt_and_preds, ignore_index=True)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        all_gp.to_csv(outdir / 'all_gt_preds.csv.gz')
        all_metrics.to_csv(outdir / 'all_metrics.csv.gz')


@torch.no_grad()
def predict_points(dataloader, model, device, threshold, cfg):
    """ Predict and find points. """
    model.eval()
    n_images = dataloader.dataset.num_images()
    
    process_fn = partial(_process_fn, model=model, device=device, cfg=cfg, has_target=False, return_input=False, return_target=False, return_numpy=True)
    collate_fn = partial(_collate_fn, device=device, has_input=False, has_target=False)
    processed_images = dataloader.dataset.process_per_patch(dataloader, process_fn, collate_fn, progress=True, max_prefetch=1)

    all_localizations = []
    
    progress = tqdm(processed_images, total=n_images, desc='PRED', leave=False)
    for image_id, image_hw, segmentation_map in progress:
        # from skimage.transform import rescale
        # io.imsave('debug_' + image_id, rescale(segmentation_map, scale=0.15, anti_aliasing=True))
        # find connected components and centroids
        localizations = segmentation_map_to_points(segmentation_map, thr=threshold)
        localizations['imgName'] = image_id
        localizations['thr'] = threshold
        all_localizations.append(localizations)

    all_localizations = pd.concat(all_localizations, ignore_index=True)
    return all_localizations