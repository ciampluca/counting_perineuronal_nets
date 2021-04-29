import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# wrapper for scipy's Hungarian algorithm to deal with infs in the cost matrix
# https://github.com/scipy/scipy/issues/6900#issuecomment-451735634
def _linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)


def match(groundtruth, predictions, threshold):
    """
    Matches points between groundtruth and predictions based on distance.

    Args:
        - groundtruth: a pandas DataFrame with 'X' and 'Y' columns (groundtruth points)
        - predictions: a pandas DataFrame with 'X', 'Y', and 'score' columns (predicted points)
        - threshold: the maximum tolerated distance between matching points
    Returns:
        A copy of the groundtruth DataFrame, augmented with Xp and Yp columns
        containing the coordinates of the predicted points that matches the
        groundtruth points. Non-matched groundtruth points have NaNs in Xp and Yp
        columns. Non-matched predictions are added as additional rows
        with NaNs in X, Y columns.
    """
    groundtruth_and_predictions = groundtruth.copy().reset_index()
    groundtruth_and_predictions['Xp'] = np.nan
    groundtruth_and_predictions['Yp'] = np.nan
    groundtruth_and_predictions['score'] = np.nan

    gt_points = groundtruth[['X', 'Y']].values
    pred_points = predictions[['X', 'Y']].values

    distance_matrix = cdist(gt_points, pred_points, 'euclidean')
    matches = distance_matrix < threshold
    distance_matrix[~matches] = np.inf
    matched_pred = matches.any(axis=0)
    matched_gt = matches.any(axis=1)

    non_matched_predictions = predictions  # assume all are non-matched

    if matched_gt.any():
        # run hungarian algo to find best assignment between groundtruth and predictions that matches
        matched_distance_matrix = distance_matrix[matched_gt][:, matched_pred]
        gt_idx, pred_idx = _linear_sum_assignment_with_inf(matched_distance_matrix)
        distances = matched_distance_matrix[gt_idx, pred_idx]

        # the algorithm may assign distant couples, keep only matches with reasonable distance
        real_matches = distances < threshold
        gt_idx = gt_idx[real_matches]
        pred_idx = pred_idx[real_matches]

        # get indices wrt original indexing of gt and predictions
        gt_idx = np.flatnonzero(matched_gt)[gt_idx]
        pred_idx = np.flatnonzero(matched_pred)[pred_idx]

        groundtruth_and_predictions.loc[gt_idx, ['Xp', 'Yp', 'score']] = predictions.loc[pred_idx, ['X', 'Y', 'score']].values
        non_matched_predictions = predictions[~predictions.index.isin(pred_idx)]
    
    non_matched_predictions = non_matched_predictions.rename({'X':'Xp', 'Y':'Yp'}, axis=1)
    groundtruth_and_predictions = groundtruth_and_predictions.append(non_matched_predictions, ignore_index=True)
    return groundtruth_and_predictions


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


def compute_metrics(groundtruth_and_predictions, detection=True, counting=True, image_hw=None):
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

    Returns:
        A metric_name -> metric_value dictionary.
    """
    metrics = {}

    inA = ~groundtruth_and_predictions.X.isna()
    inB = ~groundtruth_and_predictions.Xp.isna()

    if detection:
        true_positives = (inA & inB).sum()
        false_positives = (~inA & inB).sum()
        false_negatives = (inA & ~inB).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.
        recall = true_positives / (true_positives + false_negatives)
        f1_score = true_positives / (true_positives + false_negatives + false_positives)

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