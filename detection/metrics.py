import numpy as np

def dice_jaccard(y_true, y_pred, y_scores, shape, smooth=1, thr=None):
    """
    Computes Dice and Jaccard coefficients.

    Args:
        y_true (ndarray): (N,4)-shaped array of groundtruth bounding boxes coordinates in xyxy format
        y_pred (ndarray): (N,4)-shaped array of predicted bounding boxes coordinates in xyxy format
        y_scores (ndarray): (N,)-shaped array of prediction scores
        shape (tuple): shape of the map, i.e. (h, w)
        smooth (int, optional): Smoothing factor to avoid ZeroDivisionError. Defaults to 1.
        thr (float, optional): Threshold to binarize predictions; if None, the soft version of
                               the coefficients are computed. Defaults to None.

    Returns:
        tuple: The Dice and Jaccard coefficients.
    """
    
    m_true = np.zeros(shape, dtype=np.float32)
    for x0, y0, x1, y1 in y_true.astype(int):
        m_true[y0:y1 + 1, x0: x1 + 1] = 1.

    if thr is not None:
        keep = y_scores >= thr
        y_pred = y_pred[keep]
        y_scores = y_scores[keep]

    m_pred = np.zeros_like(m_true)
    for (x0, y0, x1, y1), score in zip(y_pred.astype(int), y_scores):
        m_pred[y0:y1 + 1, x0: x1 + 1] = np.maximum(m_pred[y0:y1 + 1, x0: x1 + 1], score)

    intersection = np.sum(m_true * m_pred)
    sum_ = np.sum(m_true) + np.sum(m_pred)
    union = sum_ - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    dice = 2. * (intersection + smooth) / (sum_ + smooth)

    return dice.mean(), jaccard.mean()