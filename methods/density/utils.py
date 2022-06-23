import pandas as pd

from skimage.feature import peak_local_max


def density_map_to_points(density_map, min_distance, threshold):
    """ Generate a pd.DataFrame of points coordinates with scores
        using local maximum peak detection on a density map.

    Args:
        density_map (ndarray): (H,W,C)-shaped density map.
        min_distance (int): minimum distance between peaks.
        threshold (float): relative threshold in [0, 1] of the local_max / global_max ratio.

    Returns:
        pd.DataFrame: Detected points with scores (the maximum value of the local peak).
    """
    results = []
    for i in range(density_map.shape[2]):
        class_dmap = density_map[:, :, i]
        count = class_dmap.sum().astype(int)

        peak_idx = peak_local_max(
            class_dmap,
            num_peaks=count,
            threshold_abs=0.0,
            min_distance=min_distance,
            exclude_border=min_distance,
            threshold_rel=threshold,
        )

        localizations = pd.DataFrame(peak_idx, columns=['Y', 'X'])
        localizations['score'] = class_dmap[tuple(peak_idx.T)]
        localizations['class'] = i
        results.append(localizations)

    results = pd.concat(results, ignore_index=True)
    return results


def normalize_map(density_map):
    dmin, dmax = density_map.min(), density_map.max()
    if dmin == dmax:
        return density_map
    return (density_map - dmin) / (dmax - dmin)