import numpy as np
from skimage import draw


# some colors
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
CYAN = [0, 255, 255]
MAGENTA = [255, 0, 255]


DEFAULT_PALETTE = {
    'tp_gt': GREEN,
    'tp_p': RED,
    'tp_match': YELLOW,
    'fn': CYAN,
    'fp': MAGENTA
}


def _square_marker(r, c, radius, shape):
    rs, cs = r - radius, c - radius
    re, ce = r + radius, c + radius
    rr, cc = draw.rectangle_perimeter(start=(rs, cs), end=(re, ce), shape=shape)
    return rr, cc, 1


def _circle_marker(r, c, radius, shape):
    return draw.circle_perimeter_aa(r, c, radius)


def draw_points(image, points_yx, radius=10, marker='circle', color=RED):
    """ Draw points on the image.

    Args:
        image (ndarray): (H,W)-shaped image array.
        points_yx (ndarray): (N,2)-shaped array of points.
        radius (int, optional): Half size of the markers. Defaults to 10.
        marker (str, optional): Type of the marker; can be 'circle' or 'square'. Defaults to 'circle'.
        color ([type], optional): Color of the markers as RGB tuple. Defaults to RED = (255, 0, 0).
    """
    assert marker in ('circle', 'square'), f'Marker type not supported: {marker}'

    if marker == 'circle':
        draw_marker_fn = _circle_marker
    elif marker == 'square':
        draw_marker_fn = _square_marker

    image = np.stack((image, image, image), axis=-1)

    for r, c in points_yx.astype(int):
        rr, cc, val = draw_marker_fn(r, c, radius, image.shape)
        draw.set_color(image, (rr, cc), color, alpha=val)
    
    return image
    

def draw_groundtruth_and_predictions(image, gp, radius=10, marker='circle', palette=None):
    """ Draw groundtruth and predicted points on the image.

    Args:
        image (ndarray, uint8): (H,W)-shaped array in [0, 255].
        gp (pd.DataFrame): dataframe with matched groundtruth and predictions.
        radius (int, optional): Half size of the markers. Defaults to 10.
        marker (str, optional): Type of the marker; can be 'circle' or 'square'. Defaults to 'circle'.
        palette (dict, optional): Color palette for markers as a dict of keys -> color. Keys to be specified are:
            - 'tp_gt': color of groundtruth true positives points,
            - 'tp_p': color of predicted true positives points,
            - 'tp_match': color of the line connecting matching points,
            - 'fn': color of false negative points,
            - 'fp': color of false positive points.
            Colors are triple of integers in [0, 255]. If None, the default palette is used. Defaults to None.
    """
    palette = palette or DEFAULT_PALETTE

    assert marker in ('circle', 'square'), f'Marker type not supported: {marker}'
    if marker == 'circle':
        draw_marker_fn = _circle_marker
    elif marker == 'square':
        draw_marker_fn = _square_marker

    image = np.stack((image, image, image), axis=-1)

    # iterate gt and predictions
    for c_gt, r_gt, c_p, r_p, score, agreement in gp[['X', 'Y', 'Xp', 'Yp', 'score', 'agreement']].values:
        has_gt = not np.isnan(r_gt)
        has_p = not np.isnan(r_p)

        if has_gt:  # draw groundtruth
            color = palette['tp_gt'] if has_p else palette['fn']
            r_gt, c_gt = int(r_gt), int(c_gt)
            rr, cc, val = draw_marker_fn(r_gt, c_gt, radius, image.shape)
            draw.set_color(image, (rr, cc), color, alpha=val)

        if has_p:  # draw prediction
            color = palette['tp_p'] if has_gt else palette['fp']
            r_p, c_p = int(r_p), int(c_p)
            rr, cc, val = draw_marker_fn(r_p, c_p, radius, image.shape)
            draw.set_color(image, (rr, cc), color, alpha=val)

        if has_gt and has_p:  # draw line between match
            rr, cc, val = draw.line_aa(r_gt, c_gt, r_p, c_p)
            draw.set_color(image, (rr, cc), palette['tp_match'], alpha=val)
        
    return image