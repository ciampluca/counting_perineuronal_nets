import numpy as np


def game(gt_dmap, pred_dmap, L):
    # TODO: it makes sense? eventually, how can we implement game in this case?
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


def counting(gt_cmap, pred_cmap, patch_size):
    """
    Compute counting metrics from the gt and the predicted cmaps

    Args:
        - gt_cmap: (N,M)-array of floats representing gt cmap
        - pred_cmap: (N,M)-array of floats representing pred cmap
        - patch_size (int): int corresponding to target_patch_size

    Returns:
        A metric_name -> metric_value dictionary.
    """

    n_groundtruth = (gt_cmap / (patch_size ** 2.0)).sum()
    n_predictions = (pred_cmap / (patch_size ** 2.0)).sum()

    counting_error = n_predictions - n_groundtruth
    counting_abs_error = abs(counting_error)
    counting_squared_error = counting_error ** 2
    counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)
    # counting_game = {f'count/game-{l}': game(gt_dmap, pred_dmap, L=l) for l in range(6)}

    metrics = {
        'count/err': counting_error,
        'count/mae': counting_abs_error,
        'count/mse': counting_squared_error,
        'count/mare': counting_abs_relative_error,
        # **counting_game
    }

    return metrics
