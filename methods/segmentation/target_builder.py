import logging
import numpy as np

from skimage.draw import disk, line_aa
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

log = logging.getLogger(__name__)
log.addFilter(DuplicateFilter())

    
class SegmentationTargetBuilder:
    """ This builds the segmantation and loss weights maps, as described
        in the 'Methods' section of the paper:

        Falk, Thorsten, et al. "U-Net: deep learning for cell counting,
        detection, and morphometry." Nature methods 16.1 (2019): 67-70.
    """

    def __init__(
        self,
        radius=20,
        radius_ignore=25,
        v_bal=0.1,
        sigma_bal=10,
        sigma_sep=6,
        lambda_sep=50,
        width_sep=1,
        **kwargs
    ):
        """ Constructor.

        Args:
            radius (int, optional): Radius (in px) of the dot placed on a cell in the segmentation map. Defaults to 20.
            radius_ignore (int, optional): Radius (in px) of the 'ignore' zone surrounding the cell. Defaults to 25.
            v_bal (float, optional): Weight of the loss of bg pixels. Defaults to 0.1.
            sigma_bal (int, optional): Gaussian stddev (in px) to blur loss weights of bg pixels near fg pixels. Defaults to 10.
            sigma_sep (int, optional): Gaussian stddev (in px) to blur loss weights of bg pixels near bg ridge pixels. Defaults to 6.
            lambda_sep (int, optional): Multiplier for the separation weights (before being summed to the other loss weights). Defaults to 50.
            width_sep (int, optional):  Width (in px) of bg ridge separating two overlapping foreground cells. Defaults to 1.
        """
        
        assert width_sep == 1, 'Only width_sep=1 is currently supported.'

        self.radius = radius
        self.radius_ignore = radius_ignore
        self.v_bal = v_bal
        self.sigma_bal = sigma_bal
        self.sigma_sep = sigma_sep
        self.lambda_sep = lambda_sep
        self.width_sep = width_sep

    def build(self, shape, points_yx):
        if len(points_yx) < 3:  # fallback to simple case
            return self.build_cliques(shape, points_yx)

        radius = self.radius
        radius_ign = self.radius_ignore
        v_bal = self.v_bal
        s_bal = self.sigma_bal
        s_sep = self.sigma_sep
        lambda_sep = self.lambda_sep

        min_yx, max_yx = np.array((0, 0)), np.array(shape) - 1
        segmentation = np.zeros(shape, dtype=np.float32)
        
        if len(points_yx) == 0:  # empty patch
            weights = np.full_like(segmentation, v_bal, dtype=np.float32)
            return segmentation, weights
        
        weights_balance = np.zeros(shape, dtype=np.float32)
        weights_separation = np.zeros(shape, dtype=np.float32)

        # build segmentation map
        for center in points_yx:  # ignore region
            rr, cc = disk(center, radius_ign, shape=shape)
            segmentation[rr, cc] = -1
        
        dmap1 = np.full(shape, np.inf, dtype=np.float32)
        dmap2 = np.full(shape, np.inf, dtype=np.float32)
        for center in points_yx:  # fg regions (overwrites ignore regions)
            rr, cc = disk(center, radius, shape=shape)
            segmentation[rr, cc] = 1
            
            # build also distance maps needed for w_sep
            distance_map = np.ones(shape, dtype=np.float32)
            distance_map[rr, cc] = 0
            distance_map = distance_transform_edt(distance_map)

            # keep smallest and second smallest per pixel
            dmap1, dmap2 = np.sort(np.stack((dmap1, dmap2, distance_map)), axis=0)[:2]

        # draw ridge as background
        for start, end in self._find_ridges(points_yx, max_yx):
            r0, c0 = np.clip(start, min_yx, max_yx).astype(int)
            r1, c1 = np.clip(end  , min_yx, max_yx).astype(int)
            rr, cc, _ = line_aa(r0, c0, r1, c1)
            segmentation[rr, cc] = 0  # set to bg to create separation
                
        # build w_bal
        is_fg, is_bg, is_ign = segmentation > 0, segmentation == 0, segmentation < 0       
        d1 = distance_transform_edt(is_bg)  # find distance of bg points to nearest fg point
        smooth_v_bal = v_bal + (1 - v_bal) * np.exp(- (d1 ** 2) / (2 * s_bal ** 2))  # smootly increase weight of bg near fg
        weights_balance = np.select([is_fg, is_bg, is_ign], [1, smooth_v_bal, 0])
        
        # build w_sep 
        d1_plus_d2 = dmap1 + dmap2  # sum of distances to nearest and second nearest foreground component
        weights_separation = np.exp(- (d1_plus_d2 ** 2) / (2 * s_sep ** 2))  # smootly increase weight of bg near ridge
        weights_separation[segmentation > 0] = 0  # zero out on foreground regions
        
        # combined weights
        weights = weights_balance + lambda_sep * weights_separation
        weights = weights.astype(np.float32)  # ensure float32

        # set ignore regions as bg in the segmentation map
        segmentation[segmentation < 0] = 0
        return segmentation, weights

    def pack(self, image, target, pad=None):
        segmentation, weights = target
        segmentation = np.pad(segmentation, pad) if pad else segmentation
        weights = np.pad(weights, pad) if pad else weights  # 0 in loss weight = don't care

        # stack in a unique RGB-like tensor, useful for applying data augmentation
        return np.stack((image, segmentation, weights), axis=-1)

    @classmethod
    def _find_ridges(cls, points, limits):
        """ This finds and yields all the segments that separate overlapping points. """
        n_points = len(points)

        if n_points == 1:
            yield from []  # no ridges
        
        elif n_points == 2:  # ridge is perpendicular bisector
            a, b = points
            if not np.allclose(a, b):  # discard duplicate points
                yield cls._find_ridge_between_two(a, b, limits)
            
        else:  # ridge is found using voronoi partitioning
            try:
                vor = Voronoi(points)
                center = vor.points.mean(axis=0)

                for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
                    simplex = np.asarray(simplex)
                    if np.all(simplex >= 0):
                        yield vor.vertices[simplex]
                    else:
                        i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                        t /= np.linalg.norm(t)
                        n = np.array([-t[1], t[0]])  # normal

                        midpoint = vor.points[pointidx].mean(axis=0)
                        direction = np.sign(np.dot(midpoint - center, n)) * n

                        start_point = vor.vertices[i]
                        if np.any(start_point < 0) or np.any(start_point >= limits):
                            log.warn(f'Ignoring Voronoi vertex outside image limits.')
                            continue

                        # check intersection with patch limits:
                        # start_point + t * direction == (0, 0) or (shape - 1)
                        with np.errstate(divide='ignore'):  # division by zero are ok (result is +-inf)
                            t0 = - start_point / direction
                            t1 = (limits - start_point) / direction
                        t = np.hstack((t0, t1))
                        t = np.min(t[t >= 0])
                        far_point = start_point + t * direction

                        yield start_point, far_point

            except QhullError as e:
                # 3+ points that do not span the plane => collinear, we separate them in pairs
                sorted_points = points[np.lexsort(points.T)]
                for a, b in zip(sorted_points[:-1], sorted_points[1:]):
                    if not np.allclose(a, b):  # discard duplicate points
                        yield cls._find_ridge_between_two(a, b, limits)

    
    @staticmethod
    def _find_ridge_between_two(a, b, limits):
        """ helper function for _find_ridges() when only 2 points overlap """
        middle = (a + b) / 2
        ab_dir = b - a
        ridge_dir = np.array((-ab_dir[1], ab_dir[0]))

        # check intersections with patch limits:
        # middle +- t * ridge_dir == (0, 0) or limits
        with np.errstate(divide='ignore'):
            t0 = middle / ridge_dir
            t1 = (middle - limits) / ridge_dir
        t = np.hstack((t0, t1))
        
        h_min = np.max(t[t < 0])
        h_max = np.min(t[t >= 0])
        
        start = middle + h_min * ridge_dir
        end = middle + h_max * ridge_dir

        return start, end
