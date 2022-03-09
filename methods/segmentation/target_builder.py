import numpy as np

from skimage.draw import disk, polygon2mask, polygon_perimeter
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi

from methods.base_target_builder import BaseTargetBuilder
    
class SegmentationTargetBuilder(BaseTargetBuilder):
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

    def build(self, shape, locations, n_classes=None):
        if 'class' not in locations.columns:
            locations['class'] = 0
        
        if n_classes is None:
            n_classes = locations['class'].max() + 1
        
        segmentations = []
        weights = []
        for i in range(n_classes):
            points_i = locations[locations['class'] == i][['Y', 'X']].values
            segmentation_i, weights_i = self._build_single_class(shape, points_i)
            segmentations.append(segmentation_i)
            weights.append(weights_i)
        
        segmentations = np.stack(segmentations, axis=-1)
        weights = np.stack(weights, axis=-1)
        return segmentations, weights

    def _build_single_class(self, shape, points_yx):

        radius = self.radius
        radius_ign = self.radius_ignore
        v_bal = self.v_bal
        s_bal = self.sigma_bal
        s_sep = self.sigma_sep
        lambda_sep = self.lambda_sep

        segmentation = np.zeros(shape, dtype=np.float32)

        if len(points_yx) == 0:  # empty patch
            weights = np.full_like(segmentation, v_bal, dtype=np.float32)
            return segmentation, weights

        weights_balance = np.zeros(shape, dtype=np.float32)
        weights_separation = np.zeros(shape, dtype=np.float32)

        dmap1 = np.full(shape, np.inf, dtype=np.float32)
        dmap2 = np.full(shape, np.inf, dtype=np.float32)
        
        # add fake/outside points to make Voronoi partitioning possible (even when n_points < 4)
        t, l, b, r = 0, 0, shape[0] - 1, shape[1] - 1
        guard_points_yx = np.array([[t - 2*radius_ign, l - 2*radius_ign],
                                    [b + 2*radius_ign, l - 2*radius_ign],
                                    [b + 2*radius_ign, r + 2*radius_ign],
                                    [t - 2*radius_ign, r + 2*radius_ign]], dtype=np.float32)
        points_yx = np.vstack((points_yx, guard_points_yx))

        vor = Voronoi(points_yx)
        centers = vor.points
        regions, vertices = self._voronoi_finite_polygons_2d(vor)

        for seed_yx, region in zip(centers, regions):
            region_vertices = np.array([vertices[v] for v in region])
            region_mask = polygon2mask(shape, region_vertices)
        
            # ignore region
            rr, cc = disk(seed_yx, radius_ign, shape=shape)
            segmentation[rr, cc] = np.where(region_mask[rr, cc], -1, segmentation[rr, cc])

            # foreground region
            rr, cc = disk(seed_yx, radius, shape=shape)
            segmentation[rr, cc] = np.where(region_mask[rr, cc], 1, segmentation[rr, cc])

            if region_mask[rr, cc].size:  # if there are some fg pixels to change
                # build distance maps needed for w_sep
                distance_map = np.ones(shape, dtype=np.float32)
                distance_map[rr, cc] = np.where(region_mask[rr, cc], 0, distance_map[rr, cc])
                distance_map = distance_transform_edt(distance_map)

                # keep smallest and second smallest per pixel
                dmap1, dmap2 = np.sort(np.stack((dmap1, dmap2, distance_map)), axis=0)[:2]
        
            # background ridges
            r, c = region_vertices.T
            rr, cc = polygon_perimeter(r, c, shape=shape)
            segmentation[rr, cc] = 0

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
        segmentations, weights = target

        segmentations = np.pad(segmentations, pad) if pad else segmentations
        weights = np.pad(weights, pad) if pad else weights  # 0 in loss weight = don't care

        # stack in a unique RGB-like tensor, useful for applying data augmentation
        return np.concatenate((image, segmentations, weights), axis=-1)

    @staticmethod
    def _voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
        Thanks to: https://gist.github.com/pv/8036995

        Args:
            vor (Voronoi): Input diagram
            radius (float, optional): Distance to 'points at infinity'.

        Returns:
        regions (list of tuples):
            Indices of vertices in each revised Voronoi regions.
        vertices (list of tuples):
            Coordinates for revised Voronoi vertices. Same as coordinates of input
            vertices, with 'points at infinity' appended to the end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):  # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:  # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)