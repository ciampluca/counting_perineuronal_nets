def dice_jaccard(y_true, y_pred, smooth=1, thr=None):
    """
    Computes Dice and Jaccard coefficients.

    Args:
        y_true (ndarray): (H,W)-shaped groundtruth map with binary values (0, 1)
        y_pred (ndarray): (H,W)-shaped predicted map with values in [0, 1]
        smooth (int, optional): Smoothing factor to avoid ZeroDivisionError. Defaults to 1.
        thr (float, optional): Threshold to binarize predictions; if None, the soft version of
                               the coefficients are computed. Defaults to None.

    Returns:
        tuple: The Dice and Jaccard coefficients.
    """
    axis = (0, 1) if y_true.ndim == 2 else tuple(range(1, y_true.ndim))
    y_pred = (y_pred >= thr) if thr is not None else y_pred

    intersection = (y_true * y_pred).sum(axis)
    sum_ = y_true.sum(axis) + y_pred.sum(axis)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)
    return dice.mean(), jaccard.mean()
