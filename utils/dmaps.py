import numpy as np


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
    val = 0.0
    image_h, image_w = gt_dmap.shape
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


def compute_metrics(gt_dmap, pred_dmap, counting=True):
    """
    Compute counting metrics from the gt and the predicted dmaps

    Args:
        - gt_dmap: (N,M)-array of floats representing gt dmap
        - pred_dmap: (N,M)-array of floats representing pred dmap

    Returns:
        A metric_name -> metric_value dictionary.
    """
    metrics = {}

    if counting:
        n_predictions = pred_dmap.sum()
        n_groundtruth = gt_dmap.sum()
        counting_error = n_predictions - n_groundtruth
        counting_abs_error = abs(counting_error)
        counting_squared_error = counting_error ** 2
        counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)

        metrics['count/err'] = counting_error
        metrics['count/mae'] = counting_abs_error
        metrics['count/mse'] = counting_squared_error
        metrics['count/mare'] = counting_abs_relative_error

        counting_game = {f'count/game-{l}': game(gt_dmap, pred_dmap, L=l) for l in range(6)}
        metrics.update(counting_game)

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
    image_h, image_w = pred_dmap.shape
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


def compute_metrics_yx(gt_yx, pred_dmap, counting=True):
    """
    Compute counting metrics from the gt and the predicted dmaps

    Args:
        - gt_yx: (N,2)-array of groundtruth points
        - pred_dmap: (N,M)-array of floats representing pred dmap

    Returns:
        A metric_name -> metric_value dictionary.
    """
    metrics = {}

    if counting:
        n_predictions = pred_dmap.sum()
        n_groundtruth = len(gt_yx)
        counting_error = n_predictions - n_groundtruth
        counting_abs_error = abs(counting_error)
        counting_squared_error = counting_error ** 2
        counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)

        metrics['count/err'] = counting_error
        metrics['count/mae'] = counting_abs_error
        metrics['count/mse'] = counting_squared_error
        metrics['count/mare'] = counting_abs_relative_error

        counting_game = {f'count/game-{l}': game_yx(gt_yx, pred_dmap, L=l) for l in range(6)}
        metrics.update(counting_game)

    return metrics