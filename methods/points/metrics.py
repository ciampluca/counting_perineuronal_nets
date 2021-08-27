import warnings


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


def detection_and_counting(groundtruth_and_predictions, detection=True, counting=True, image_hw=None):
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