import warnings

import numpy as np
import pandas as pd


def game(true_yx, pred_yx, image_hw, L):
    """
    Computes Grid Average Mean absolute Error (GAME) from yx points.

    Args: 
        - true_yx: (N,2)-array of groundtruth points
        - pred_yx: (N,2)-array of predicted points
        - L: grid level, the image space will be divided in 4 ** L patches
    Returns:
        Value of the GAME-L metric.
    """
    val = 0.0
    image_h, image_w = image_hw
    patch_h, patch_w = image_h / 2**L, image_w / 2**L
    for r in range(2**L):
        sy, ey = patch_h * r, patch_h * (r + 1)
        true_y, pred_y = true_yx[:, 0], pred_yx[:, 0]
        true_x_filtered = true_yx[(true_y >= sy) & (true_y < ey), 1]
        pred_x_filtered = pred_yx[(pred_y >= sy) & (pred_y < ey), 1]
        for c in range(2**L):
            sx, ex = patch_w * c, patch_w * (c + 1)
            n_true = ((true_x_filtered >= sx) & (true_x_filtered < ex)).sum()
            n_pred = ((pred_x_filtered >= sx) & (pred_x_filtered < ex)).sum()
            val += abs(n_pred - n_true)
    
    return val


def detection_and_counting(
    groundtruth_and_predictions,
    detection=True,
    counting=True,
    image_hw=None,
    n_classes=None,
):
    """
    Compute counting and/or detection metrics from a pandas DataFrame of
    matched groundtruth and prediction points.

    Args:
        - groundtruth_and_predictions: a pandas DataFrame with at least following four columns:
          X : x-coordinate of the groundtruth point (NaN for non-matches)
          Y : y-coordinate of the groundtruth point (NaN for non-matches)
          Xp: x-coordinate of the corresponding predicted point (NaN for non-matches)
          Yp: y-coordinate of the corresponding predicted point (NaN for non-matches)

        - detection: whether to compute and report detection metrics
        - counting: whether to compute and report counting metrics
        - image_hw: 2-tuple with image size for GAME computation (counting)
        - n_classes: number of classes; if None, it is estimated from data

    Returns:
        A metric_name -> metric_value dictionary.
    """
    if n_classes is None:
        n_classes = groundtruth_and_predictions['class'].max() + 1

    micro_metrics = _detection_and_counting_single_class(groundtruth_and_predictions, detection=detection, counting=counting, image_hw=image_hw)
    micro_metrics = {f'{k}/micro': v for k, v in micro_metrics.items()}

    class_metrics = []
    for i in range(n_classes):
        gp = groundtruth_and_predictions[groundtruth_and_predictions['class'] == i]
        metrics_i = _detection_and_counting_single_class(gp, detection=detection, counting=counting, image_hw=image_hw)
        class_metrics.append(metrics_i)

    macro_metrics = {f'{k}/macro': np.mean([class_metrics[i][k] for i in range(n_classes)]) for k in class_metrics[0].keys()}
    class_metrics = {f'{k}/cls{i}': v for i, metrics_i in enumerate(class_metrics) for k, v in metrics_i.items()}

    metrics = {}
    metrics.update(class_metrics)
    metrics.update(micro_metrics)
    metrics.update(macro_metrics)
    return metrics


def _detection_and_counting_single_class(groundtruth_and_predictions, detection=True, counting=True, image_hw=None):
    metrics = {}

    inA = ~groundtruth_and_predictions.X.isna()
    inB = ~groundtruth_and_predictions.Xp.isna()

    if detection:
        true_positives = (inA & inB).sum()
        false_positives = (~inA & inB).sum()
        false_negatives = (inA & ~inB).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * true_positives / (2 * true_positives + false_negatives + false_positives)

        metrics['pdet/precision'] = precision
        metrics['pdet/recall'] = recall
        metrics['pdet/f1_score'] = f1_score
    
    if counting:
        n_predictions = inB.sum()
        n_groundtruth = inA.sum()
        counting_error = n_predictions - n_groundtruth
        counting_abs_error = abs(counting_error)
        counting_squared_error = counting_error ** 2
        counting_abs_relative_error = abs(counting_error) / max(n_groundtruth, 1)

        metrics['count/err'] = counting_error
        metrics['count/mae'] = counting_abs_error
        metrics['count/mse'] = counting_squared_error
        metrics['count/mare'] = counting_abs_relative_error

        if image_hw is not None:
            true_yx = groundtruth_and_predictions.loc[inA, ['Y' , 'X' ]].values
            pred_yx = groundtruth_and_predictions.loc[inB, ['Yp', 'Xp']].values
            counting_game = {f'count/game-{l}': game(true_yx, pred_yx, image_hw, L=l) for l in range(6)}
            metrics.update(counting_game)
        else:
            warnings.warn("GAME metric not computed. Provide 'image_hw' to compute also this metric.")
    
    return metrics


def _ap_single_class(pr, suffix):
    pr = pr.sort_values(f'pdet/recall/{suffix}')
    recalls = pr[f'pdet/recall/{suffix}'].values
    precisions = pr[f'pdet/precision/{suffix}'].values
    ap = - np.sum(np.diff(recalls) * precisions[:-1])  # sklearn's ap
    return ap


def detection_average_precision(threshold_metrics):
    """ Compute micro and macro averaged precisions.

    Args:
        threshold_metrics (pd.DataFrame): dataframe containing the columns 'thr', 'pdet/recall/<...>',
        and 'pdet/precision/<...>' metrics.
    """
    if not isinstance(threshold_metrics, pd.DataFrame):
        threshold_metrics = pd.DataFrame(threshold_metrics) 

    micro_average_precision = _ap_single_class(threshold_metrics, 'micro')

    classes = [int(c[len('pdet/recall/cls'):]) for c in threshold_metrics.columns if c.startswith('pdet/recall/cls')]
    classes_average_precision = [_ap_single_class(threshold_metrics, f'cls{i}') for i in classes]

    macro_average_precision = np.mean(classes_average_precision)

    return {
        'pdet/average_precision/micro': micro_average_precision.item(),
        'pdet/average_precision/macro': macro_average_precision.item(),
        **{f'pdet/average_precision/cls{i}': v.item() for i, v in enumerate(classes_average_precision)}
    }


