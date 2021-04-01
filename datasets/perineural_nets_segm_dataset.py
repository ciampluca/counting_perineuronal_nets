import itertools
import numpy as np
import pandas as pd
import h5py

from copy import deepcopy
from pathlib import Path
from skimage.draw import disk, line_aa
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset, ConcatDataset


# from https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
def find_cliques(adj_matrix):
    N = {i: set(np.nonzero(row)[0]) for i, row in enumerate(adj_matrix)}

    def BronKerbosch1(P, R=None, X=None):
        P = set(P)
        R = set() if R is None else R
        X = set() if X is None else X
        if not P and not X:
            yield R
        while P:
            v = P.pop()
            yield from BronKerbosch1(
                P=P.intersection(N[v]), R=R.union([v]), X=X.intersection(N[v]))
            X.add(v)

    P = N.keys()
    yield from BronKerbosch1(P)


class _PerineuralNetsSegmImage(Dataset):
    """ Dataset that provides per-patch iteration of a single big image file. """

    # params for groundtruth segmentation maps generation
    DEFAULT_GT_PARAMS = {
        'radius': 35,         # radius (in px) of the dot placed on a cell in the segmentation map
        'radius_ignore': 40,  # radius (in px) of the 'ignore' zone surrounding the cell
        'v_bal': 0.1,         # weight of the loss of bg pixels
        'sigma_bal': 10,      # gaussian stddev (in px) to blur loss weights of bg pixels near fg pixels
        'sep_width': 1,       # width (in px) of bg ridge separating two overlapping foreground cells
        'sigma_sep': 6,       # gaussian stddev (in px) to blur loss weights of bg pixels near bg ridge pixels
        'lambda_sep': 50,     # multiplier for the separation weights (before being summed to the other loss weights)
    }
    
    def __init__(self, h5_path, annot_path, patch_size=640, stride=None, split='left', max_cache_mem=None, gt_params={}):
        
        self.h5_path = h5_path
        self.annot_path = annot_path
        
        # groundtruth parameters
        self.gt_params = deepcopy(self.DEFAULT_GT_PARAMS)
        self.gt_params.update(gt_params)
        
        assert split in ('left', 'right', 'all'), "split must be one of ('left', 'right', 'all')"
        self.split = split

        # load annotation, keep only the ones of this image
        image_id = Path(h5_path).with_suffix('.tif').name
        self.annot = pd.read_csv(annot_path, index_col=0).loc[image_id]

        # patch size (height and width)
        self.patch_hw = np.array((patch_size, patch_size))

        # windows stride size (height and width)
        self.stride_hw = np.array((stride, stride)) if stride else self.patch_hw

        # hdf5 dataset
        self.data = h5py.File(h5_path, 'r', rdcc_nbytes=max_cache_mem)['data']

        # size of the region from which we take patches
        image_hw = np.array(self.data.shape)
        image_half_hw = image_hw // np.array((1, 2))  # half only width
        self.region_hw = image_hw if split == 'all' else image_half_hw

        # the origin and limits of the region (split) of interest
        self.origin_yx = np.array((0, image_half_hw[1]) if self.split == 'right' else (0, 0))
        self.limits_yx = image_half_hw if self.split == 'left' else image_hw

        # the number of patches in a row and a column
        self.num_patches = np.ceil((self.region_hw - self.patch_hw) / self.stride_hw).astype(int)
        
    def __len__(self):
        # total number of patches
        return self.num_patches.prod().item()
    
    def __getitem__(self, index):
        n_rows, n_cols = self.num_patches
        # row and col indices of the patch
        row_col_idx = np.array((index // n_cols, index % n_cols))

        # patch boundaries
        start_yx = self.origin_yx + self.stride_hw * row_col_idx
        end_yx = np.minimum(start_yx + self.patch_hw, self.limits_yx)
        (sy, sx), (ey, ex) = start_yx, end_yx

        # read patch
        patch = self.data[sy:ey, sx:ex]
        patch_hw = patch.shape  # before padding

        # gather annotations
        selector = self.annot.X.between(sx, ex) & self.annot.Y.between(sy, ey)
        locations = self.annot.loc[selector, ['Y', 'X']].values
        patch_locations = locations - start_yx

        # build target maps
        segmentation, weights = self._build_target_maps(patch, patch_locations)

        # pad patch (in case of patches in last col/rows)
        py, px = - np.array(patch_hw) % self.patch_hw
        pad = ((0, py), (0, px))

        patch = np.pad(patch, pad)  # defaults to zero padding
        segmentation = np.pad(segmentation, pad)
        weights = np.pad(weights, pad)  # 0 in loss weight = don't care

        return patch, segmentation, weights, patch_hw, start_yx

    def _build_target_maps(self, patch, locations):
        """ This builds the segmantation and loss weights maps, as described
            in the 'Methods' section of the paper:

            Falk, Thorsten, et al. "U-Net: deep learning for cell counting,
            detection, and morphometry." Nature methods 16.1 (2019): 67-70.
        """

        radius = self.gt_params['radius']
        radius_ign = self.gt_params['radius_ignore']
        v_bal = self.gt_params['v_bal']
        s_bal = self.gt_params['sigma_bal']
        s_sep = self.gt_params['sigma_sep']
        lambda_sep = self.gt_params['lambda_sep']
        
        shape = patch.shape
        min_yx, max_yx = np.array((0, 0)), np.array(shape) - 1
        segmentation = np.zeros(shape, dtype=np.float32)
        
        if len(locations) == 0:  # empty patch
            weights = np.full_like(segmentation, v_bal)
            return segmentation, weights
        
        weights_balance = np.zeros(shape, dtype=np.float32)
        weights_separation = np.zeros(shape, dtype=np.float32)

        # build segmentation map
        for center in locations:  # ignore region
            rr, cc = disk(center, radius_ign, shape=shape)
            segmentation[rr, cc] = -1
        
        distance_maps = []
        for center in locations:  # fg regions (overwrites ignore regions)
            rr, cc = disk(center, radius, shape=shape)
            segmentation[rr, cc] = 1
            
            # build also distance maps needed for w_sep
            distance_map = np.ones(shape, dtype=np.float32)
            distance_map[rr, cc] = 0
            distance_map = distance_transform_edt(distance_map)
            distance_maps.append(distance_map)
            
        distance_maps = np.stack(distance_maps)
        
        # insert bg ridges to separate overlapping instances
        intersections = squareform(pdist(locations, 'sqeuclidean')) <= 4 * radius ** 2
        for group in find_cliques(intersections):
            points_idx = np.array(list(group))
            if len(points_idx) < 2:
                continue
                
            points = locations[points_idx]
            for ridge_start, ridge_end in self._find_ridges(points):
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
        
        # set ignore regions as bg in the segmentation map
        segmentation[segmentation < 0] = 0
        return segmentation, weights

    def _find_ridges(self, points):
        """ This finds and yields all the segments that separate overlapping points. """
        radius = self.gt_params['radius_ignore']
        
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
            yield self._find_ridge_between_two(a, b, region_hw, region_start_yx)
            
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
                    yield self._find_ridge_between_two(a, b, region_hw, region_start_yx)
    
    def _find_ridge_between_two(self, a, b, region_hw, region_start_yx):
        """ helper function for _find_ridges() when only 2 points overlap """
        radius = self.gt_params['radius_ignore']
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


class PerineuralNetsSegmDataset(ConcatDataset):
    """ Dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    def __init__(self, data_root, transforms=None, patch_size=640, overlap=0, split='all', max_cache_mem=None, gt_params={}):
        self.data_root = Path(data_root)
        self.transforms = transforms
        assert split in ('train', 'validation', 'all'), "split must be one of ('train', 'validation', 'all')"
        self.split = split

        annot_path = self.data_root / 'annotation' / 'annotations.csv'
        image_files = sorted((self.data_root / 'fullFramesH5').glob('*.h5'))
        assert len(image_files) > 0, "No images found"

        if max_cache_mem:
            max_cache_mem /= len(image_files)

        if self.split == 'train':
            splits = ('left', 'right')
        elif self.split == 'validation':
            splits = ('right', 'left')
        else:  # self.split == 'all':
            splits = ('all', )
        splits = itertools.cycle(splits)

        stride = patch_size - overlap
        kwargs = dict(patch_size=patch_size, stride=stride, max_cache_mem=max_cache_mem, gt_params=gt_params)
        datasets = [_PerineuralNetsSegmImage(image_path, annot_path, split=s, **kwargs) for image_path, s in zip(image_files, splits)]
        super(self.__class__, self).__init__(datasets)

    def __getitem__(self, index):
        sample = super(self.__class__, self).__getitem__(index)

        if self.transforms:
            sample = self.transforms(sample)

        return sample       


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_root = 'data/perineuronal_nets'
    dataset = PerineuralNetsSegmDataset(data_root, split='train', patch_size=640, overlap=120, max_cache_mem=8*1024**3)  # bytes = 8 GiB
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    for _ in tqdm(dataloader):
        pass