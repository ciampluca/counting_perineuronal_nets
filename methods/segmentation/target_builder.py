import numpy as np

from skimage.draw import disk, line_aa
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import pdist, squareform

    
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
        
        distance_maps = []
        for center in points_yx:  # fg regions (overwrites ignore regions)
            rr, cc = disk(center, radius, shape=shape)
            segmentation[rr, cc] = 1
            
            # build also distance maps needed for w_sep
            distance_map = np.ones(shape, dtype=np.float32)
            distance_map[rr, cc] = 0
            distance_map = distance_transform_edt(distance_map)
            distance_maps.append(distance_map)
            
        distance_maps = np.stack(distance_maps)
        
        # insert bg ridges to separate overlapping instances
        intersections = squareform(pdist(points_yx, 'sqeuclidean')) <= 4 * radius ** 2
        for group in self._find_cliques(intersections):
            points_idx = np.array(list(group))
            if len(points_idx) < 2:
                continue
                
            points = points_yx[points_idx]
            for ridge_start, ridge_end in self._find_ridges(points, radius_ign):
                r0, c0 = np.clip(ridge_start, min_yx, max_yx).astype(int)
                r1, c1 = np.clip(ridge_end  , min_yx, max_yx).astype(int)
                rr, cc, _ = line_aa(r0, c0, r1, c1)
                segmentation[rr, cc] = 0  # set to bg to create separation
                
        # build w_bal
        is_fg, is_bg, is_ign = segmentation > 0, segmentation == 0, segmentation < 0       
        d1 = distance_transform_edt(is_bg)  # find distance of bg points to nearest fg point
        smooth_v_bal = v_bal + (1 - v_bal) * np.exp(- (d1 ** 2) / (2 * s_bal ** 2))  # smootly increase weight of bg near fg
        weights_balance = np.select([is_fg, is_bg, is_ign], [1, smooth_v_bal, 0])
        
        # build w_sep 
        distance_maps.sort(axis=0)
        d1_plus_d2 = distance_maps[:2].sum(axis=0)  # sum of distances to nearest and second nearest foreground component
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
    
    @staticmethod
    def _find_cliques(adj_matrix):
        """ Finds cliques in graphs, used for building the target segmentation maps.
            From https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
        """
        N = {i: set(np.nonzero(row)[0]) for i, row in enumerate(adj_matrix)}

        def BronKerbosch1(P, R=None, X=None):
            P = set(P)
            R = set() if R is None else R
            X = set() if X is None else X
            if not P and not X:
                yield R
            while P:
                v = P.pop()
                yield from BronKerbosch1(P=P.intersection(N[v]), R=R.union([v]), X=X.intersection(N[v]))
                X.add(v)

        P = N.keys()
        yield from BronKerbosch1(P)

    @classmethod
    def _find_ridges(cls, points, radius):
        """ This finds and yields all the segments that separate overlapping points. """
        
        # work only on the region of the intersecting points to save computation
        round_points = points.astype(int)
        region_start_yx = round_points.min(axis=0) - radius
        region_end_yx = round_points.max(axis=0) + radius
        region_hw = region_end_yx - region_start_yx
        
        points -= region_start_yx
        n_points = len(points)
        center = points.mean(axis=0)
        
        if n_points == 2:  # ridge is perpendicular bisector
            a, b = points
            if not np.allclose(a, b):  # discard duplicate points
                yield cls._find_ridge_between_two(a, b, radius, region_hw, region_start_yx)
            
        else:  # ridge is found using voronoi partitioning
            # this was useful: https://gist.github.com/Sklavit/e05f0b61cb12ac781c93442fbea4fb55
            try:
                v = Voronoi(points)
                for ridge_vertices_idx, ridge_points_idx in zip(v.ridge_vertices, v.ridge_points):
                    end_idx, start_idx = ridge_vertices_idx
                    start = v.vertices[start_idx]
                    if end_idx > 0:
                        end = v.vertices[end_idx]
                    else:
                        a, b = v.points[ridge_points_idx]
                        middle = (a + b) / 2
                        ridge_dir = middle - start
                        ridge_versus = np.sign(np.dot(middle - center, ridge_dir))
                        ridge_dir = ridge_versus * ridge_dir / (np.linalg.norm(ridge_dir) + np.finfo(np.float32).eps)

                        d_sq = np.sum((a - b) ** 2)
                        h = np.sqrt(radius ** 2 - d_sq / 4)
                        
                        # check intersection with patch limits:
                        # middle + t * ridge_dir == (0, 0) or region_hw - 1
                        with np.errstate(divide='ignore'):
                            t0 = - middle / ridge_dir
                            t1 = (region_hw - 1 - middle) / ridge_dir
                        t = np.hstack((t0, t1))
                        h_max = np.min(t[t >= 0])
                        h = np.minimum(h, h_max)
                        
                        end = middle + h * ridge_dir
                    
                    # translate back to patch coordinates
                    start = start + region_start_yx
                    end = end + region_start_yx
                    
                    yield start, end

            except QhullError as e:
                # 3+ points that do not span the plane => collinear, we separate them in pairs
                sorted_points = points[np.lexsort(points.T)]
                for a, b in zip(sorted_points[:-1], sorted_points[1:]):
                    if not np.allclose(a, b):  # discard duplicate points
                        yield cls._find_ridge_between_two(a, b, radius, region_hw, region_start_yx)
    
    @staticmethod
    def _find_ridge_between_two(a, b, radius, region_hw, region_start_yx):
        """ helper function for _find_ridges() when only 2 points overlap """
        middle = (a + b) / 2
        ab_dir = b - a
        
        ridge_dir = np.array((-ab_dir[1], ab_dir[0]))
        ridge_dir = ridge_dir / np.linalg.norm(ridge_dir)
        
        d_sq = np.sum((a - b) ** 2)
        h = np.sqrt(radius ** 2 - d_sq / 4)
        
        # check intersections with patch limits:
        # middle +- t * ridge_dir == (0, 0) or region_hw - 1

        with np.errstate(divide='ignore'):  # division by zero are ok (result is +-inf)
            t0 = middle / ridge_dir
            t1 = (middle - region_hw + 1) / ridge_dir
        t = np.hstack((t0, t1))
        
        h_min = np.max(t[t < 0])
        h_max = np.min(t[t >= 0])
        
        hs = np.maximum(-h, h_min)
        he = np.minimum( h, h_max)
        
        start = middle + hs * ridge_dir
        end = middle + he * ridge_dir

        # translate back to patch coordinates
        start += region_start_yx
        end += region_start_yx

        return start, end
