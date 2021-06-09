import pandas as pd
from skimage import measure

def segm_map_to_points(y_pred, thr=None):
    """ Find connected components of a segmentation map and
        returns a pandas DataFrame with the centroids' coordinates
        and the score (computes as maximum value of the centroid in the map).

    Args:
        y_pred (ndarray): (H,W)-shaped array with values in [0, 1]
        thr (float, optional): Optional threshold used to binarize the map; 
            if None, the map should be already binary. Defaults to None.
    """
    y_pred_hard = y_pred if thr is None else y_pred >= thr

    # find connected components and centroids
    labeled_map, num_components = measure.label(y_pred_hard, return_num=True, connectivity=1)
    localizations = measure.regionprops_table(labeled_map, properties=('centroid', 'bbox'))
    localizations = pd.DataFrame(localizations).rename({
        'centroid-0':'Y',
        'centroid-1':'X',
        'bbox-0':'y0',
        'bbox-1':'x0',
        'bbox-2':'y1',
        'bbox-3':'x1',
    }, axis=1)

    bboxes = localizations[['y0', 'x0', 'y1', 'x1']].values
    localizations['score'] = [y_pred[y0:y1,x0:x1].max() for y0, x0, y1, x1 in bboxes]
    localizations = localizations.drop(columns=['y0', 'x0', 'y1', 'x1'])

    return localizations