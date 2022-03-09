import numpy as np
import pandas as pd
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

    if 'class' not in groundtruth:
        groundtruth['class'] = 0

    if 'class' not in predictions:
        predictions['class'] = 0
    
    n_classes = max(groundtruth['class'].max(), predictions['class'].max()) + 1

    results = []
    for i in range(n_classes):
        g = groundtruth[groundtruth['class'] == i].reset_index()
        p = predictions[predictions['class'] == i].reset_index()
        gp = _match_single_class(g, p, threshold)
        gp['class'] = i
        results.append(gp)

    results = pd.concat(results, ignore_index=True)
    return results


def _match_single_class(groundtruth, predictions, threshold):
    groundtruth_and_predictions = groundtruth.copy().reset_index()
    groundtruth_and_predictions['Xp'] = np.nan
    groundtruth_and_predictions['Yp'] = np.nan
    groundtruth_and_predictions['score'] = np.nan

    gt_points = np.atleast_2d(groundtruth[['X', 'Y']].values)
    pred_points = np.atleast_2d(predictions[['X', 'Y']].values)

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
    groundtruth_and_predictions = pd.concat([groundtruth_and_predictions, non_matched_predictions], ignore_index=True)
    return groundtruth_and_predictions