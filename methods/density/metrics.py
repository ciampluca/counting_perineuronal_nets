import numpy as np
from skimage.metrics import structural_similarity


def ssim(y_true, y_pred, **kwargs):
    n_classes = y_true.shape[2]
    micro_ssim = structural_similarity(y_true, y_pred, channel_axis=2, **kwargs)
    class_ssim = [structural_similarity(y_true[:, :, i], y_pred[:, :, i], **kwargs) for i in range(n_classes)]
    macro_ssim = np.mean(class_ssim)
    return {
        'density/ssim/micro': micro_ssim,
        'density/ssim/macro': macro_ssim,
        **{f'density/ssim/cls{i}': v for i, v in enumerate(class_ssim)}
    }


def game(gt_dmap, pred_dmap, L):
    """
    Computes Grid Average Mean absolute Error (GAME) from dmaps.

    Args:
        - gt_dmap: (N,M)-array of floats representing gt dmap
        - pred_dmap: (N,M)-array of floats representing pred dmap
        - L: grid level, the image space will be divided in 4 ** L patches
    Returns:
        Value of the GAME-L metric.
    """
    # TODO add support for multiclass
    val = 0.0
    image_h, image_w = gt_dmap.shape[:2]
    patch_h, patch_w = int(np.ceil(image_h / 2**L)), int(np.ceil(image_w / 2**L))
    for r in range(2**L):
        sy, ey = patch_h * r, patch_h * (r + 1)
        for c in range(2**L):
            sx, ex = patch_w * c, patch_w * (c + 1)
            pred_dmap_patch = pred_dmap[sy:ey, sx:ex]
            gt_dmap_patch = gt_dmap[sy:ey, sx:ex]
            n_true = gt_dmap_patch.sum()
            n_pred = pred_dmap_patch.sum()
            val += abs(n_pred - n_true)

    return val


def counting(gt_dmap, pred_dmap):
    """
    Compute counting metrics from the gt and the predicted dmaps

    Args:
        - gt_dmap: (N,M)-array of floats representing gt dmap
        - pred_dmap: (N,M)-array of floats representing pred dmap

    Returns:
        A metric_name -> metric_value dictionary.
    """

    n_classes = gt_dmap.shape[2]
    micro_metrics = _counting(gt_dmap, pred_dmap, 'micro')

    class_metrics = [_counting(gt_dmap[:, :, i], pred_dmap[:, :, i], f'cls{i}') for i in range(n_classes)]

    metrics_keys = [k[:-len('/micro')] for k in micro_metrics.keys()]
    macro_metrics = {f'{k}/macro': np.mean([
        metrics_i[f'{k}/cls{i}'] for i, metrics_i in enumerate(class_metrics)
    ]) for k in metrics_keys}

    results = {
        **micro_metrics,
        **macro_metrics,
    }

    for metrics_i in class_metrics:
        results.update(metrics_i)

    return results
    

def _counting(gt_dmap, pred_dmap, suffix):
    n_predictions = pred_dmap.sum()
    n_groundtruth = gt_dmap.sum()

    counting_error = n_predictions - n_groundtruth
    counting_abs_error = abs(counting_error)
    counting_squared_error = counting_error ** 2
    counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)
    counting_game = {f'count/game-{l}/{suffix}': game(gt_dmap, pred_dmap, L=l) for l in range(6)}

    metrics = {
        f'count/err/{suffix}': counting_error,
        f'count/mae/{suffix}': counting_abs_error,
        f'count/mse/{suffix}': counting_squared_error,
        f'count/mare/{suffix}': counting_abs_relative_error,
        **counting_game
    }

    return metrics


def game_yx(true_yx, pred_dmap, L):
    """
    Computes Grid Average Mean absolute Error (GAME) from dmaps.

    Args:
        - true_yx: (N,2)-array of groundtruth points
        - pred_dmap: (N,M)-array of floats representing pred dmap
        - L: grid level, the image space will be divided in 4 ** L patches
    Returns:
        Value of the GAME-L metric.
    """
    val = 0.0
    image_h, image_w = pred_dmap.shape[:2]
    patch_h, patch_w = int(np.ceil(image_h / 2**L)), int(np.ceil(image_w / 2**L))
    for r in range(2**L):
        sy, ey = patch_h * r, patch_h * (r + 1)
        true_y = true_yx[:, 0]
        true_x_filtered = true_yx[(true_y >= sy) & (true_y < ey), 1]
        for c in range(2**L):
            sx, ex = patch_w * c, patch_w * (c + 1)
            n_pred = pred_dmap[sy:ey, sx:ex].sum()
            n_true = ((true_x_filtered >= sx) & (true_x_filtered < ex)).sum()
            val += abs(n_pred - n_true)

    return val


def counting_yx(gt_df, pred_dmap):
    """
    Compute counting metrics from the gt and the predicted dmaps

    Args:
        - gt_df: pandas dataframe with X,Y,class of groundtruth points
        - pred_dmap: (N,M)-array of floats representing pred dmap

    Returns:
        A metric_name -> metric_value dictionary.
    """
    
    n_classes = pred_dmap.shape[2]
    micro_metrics = _counting_yx_single_class(gt_df[['Y','X']].values, pred_dmap, 'micro')

    class_metrics = [_counting_yx_single_class(gt_df[gt_df['class'] == i][['Y','X']].values, pred_dmap[:, :, i], f'cls{i}') for i in range(n_classes)]

    metrics_keys = [k[:-len('/micro')] for k in micro_metrics.keys()]
    macro_metrics = {f'{k}/macro': np.mean([
        metrics_i[f'{k}/cls{i}'] for i, metrics_i in enumerate(class_metrics)
    ]) for k in metrics_keys}

    results = {
        **micro_metrics,
        **macro_metrics,
    }

    for metrics_i in class_metrics:
        results.update(metrics_i)

    return results


def _counting_yx_single_class(gt_yx, pred_dmap, suffix):
    n_predictions = pred_dmap.sum()
    n_groundtruth = len(gt_yx)

    counting_error = n_predictions - n_groundtruth
    counting_abs_error = abs(counting_error)
    counting_squared_error = counting_error ** 2
    counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)
    counting_game = {f'count/game-{l}/{suffix}': game_yx(gt_yx, pred_dmap, L=l) for l in range(6)}
    
    metrics = {
        f'count/err/{suffix}': counting_error,
        f'count/mae/{suffix}': counting_abs_error,
        f'count/mse/{suffix}': counting_squared_error,
        f'count/mare/{suffix}': counting_abs_relative_error,
        **counting_game
    }

    return metrics