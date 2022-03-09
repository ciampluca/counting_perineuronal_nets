import numpy as np
import torch


def _dice_jaccard_single_class(y_true, y_pred, smooth, axis):
    intersection = (y_true * y_pred).sum(axis)
    sum_ = y_true.sum(axis) + y_pred.sum(axis)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)
    return dice.mean(), jaccard.mean()


def _atleast_nhwc(x):
    if x.ndim == 2:
        x = x[None, ..., None]
    elif x.ndim == 3:
        x = x[None, ...]
    return x


def dice_jaccard(y_true, y_pred, smooth=1, thr=None, prefix=''):
    """
    Computes Dice and Jaccard coefficients.

    Args:
        y_true (ndarray): (H,W,C)-shaped groundtruth map with binary values (0, 1)
        y_pred (ndarray): (H,W,C)-shaped predicted map with values in [0, 1]
        smooth (int, optional): Smoothing factor to avoid ZeroDivisionError. Defaults to 1.
        thr (float, optional): Threshold to binarize predictions; if None, the soft version of
                               the coefficients are computed. Defaults to None.

    Returns:
        dict: computed metrics organized with the following keys
          - segm/{dice,jaccard}/micro: Micro-averaged Dice and Jaccard coefficients.
          - segm/{dice,jaccard}/macro: Macro-averaged Dice and Jaccard coefficients.
          - segm/{dice,jaccard}/cls0: Dice and Jaccard coefficients for class 0
          - segm/{dice,jaccard}/cls1: Dice and Jaccard coefficients for class 1
          - ...
    """
    y_pred = _atleast_nhwc(y_pred)
    y_true = _atleast_nhwc(y_true)

    y_pred = (y_pred >= thr) if thr is not None else y_pred

    micro_dice, micro_jaccard = _dice_jaccard_single_class(y_true, y_pred, smooth, axis=(1, 2, 3))
    class_dice, class_jaccard = zip(*[
        _dice_jaccard_single_class(y_true[:, :, :, i], y_pred[:, :, :, i], smooth, axis=(1, 2))
        for i in range(y_true.shape[-1])
    ])

    mean_fn = np.mean
    if isinstance(y_pred, torch.Tensor):
        mean_fn = lambda x: torch.mean(torch.stack(x))
    
    macro_dice, macro_jaccard = mean_fn(class_dice), mean_fn(class_jaccard)

    metrics = {
        f'segm/{prefix}dice/micro': micro_dice.item(),
        f'segm/{prefix}dice/macro': macro_dice.item(),
        **{f'segm/{prefix}dice/cls{i}': v.item() for i, v in enumerate(class_dice)},
        f'segm/{prefix}jaccard/micro': micro_jaccard.item(),
        f'segm/{prefix}jaccard/macro': macro_jaccard.item(),
        **{f'segm/{prefix}jaccard/cls{i}': v.item() for i, v in enumerate(class_jaccard)},
    }
    
    return metrics