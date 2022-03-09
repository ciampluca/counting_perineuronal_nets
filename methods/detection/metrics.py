import numpy as np

from methods.segmentation.metrics import dice_jaccard as segmentation_dice_jaccard


def dice_jaccard(
    y_true_boxes,
    y_true_labels,
    y_pred_boxes,
    y_pred_labels,
    y_pred_scores,
    shape,
    smooth=1,
    thr=None,
):
    """
    Computes Dice and Jaccard coefficients.

    Args:
        y_true_boxes (ndarray): (N,4)-shaped array of groundtruth bounding boxes coordinates in xyxy format
        y_true_labels (ndarray): (N,)-shaped array of groundtruth labels
        y_pred_boxes (ndarray): (N,4)-shaped array of predicted bounding boxes coordinates in xyxy format
        y_pred_labels (ndarray): (N,)-shaped array of predicted labels
        y_pred_scores (ndarray): (N,)-shaped array of prediction scores
        shape (tuple): shape of the map, i.e. (h, w, c)
        smooth (int, optional): Smoothing factor to avoid ZeroDivisionError. Defaults to 1.
        thr (float, optional): Threshold to binarize predictions; if None, the soft version of
                               the coefficients are computed. Defaults to None.

    Returns:
        tuple: The Dice and Jaccard coefficients.
    """

    m_true = np.zeros(shape, dtype=np.float32)
    for (x0, y0, x1, y1), label in zip(y_true_boxes.astype(int), y_true_labels):
        m_true[y0:y1 + 1, x0: x1 + 1, label] = 1.

    if thr is not None:
        keep = y_pred_scores >= thr
        y_pred_boxes = y_pred_boxes[keep]
        y_pred_labels = y_pred_labels[keep]
        y_pred_scores = y_pred_scores[keep]

    m_pred = np.zeros_like(m_true)
    for (x0, y0, x1, y1), label, score in zip(y_pred_boxes.astype(int), y_pred_labels, y_pred_scores):
        m_pred[y0:y1 + 1, x0: x1 + 1, label] = np.maximum(m_pred[y0:y1 + 1, x0: x1 + 1, label], score)

    return segmentation_dice_jaccard(m_true, m_pred, smooth=smooth)