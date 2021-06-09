import pandas as pd

from skimage.feature import peak_local_max


def density_map_to_points(density_map, min_distance, threshold):
    """ Generate a pd.DataFrame of points coordinates with scores
        using local maximum peak detection on a density map.

    Args:
        density_map (ndarray): (H,W)-shaped density map.
        min_distance (int): minimum distance between peaks.
        threshold (float): relative threshold in [0, 1] of the local_max / global_max ratio.

    Returns:
        pd.DataFrame: Detected points with scores (the maximum value of the local peak).
    """
    count = density_map.sum()

    peak_idx = peak_local_max(
        density_map,
        num_peaks=count,
        threshold_abs=0.0,
        min_distance=min_distance,
        exclude_border=min_distance,
        threshold_rel=threshold,
    )

    localizations = pd.DataFrame(peak_idx, columns=['Y', 'X'])
    localizations['score'] = density_map[tuple(peak_idx.T)]

    return localizations