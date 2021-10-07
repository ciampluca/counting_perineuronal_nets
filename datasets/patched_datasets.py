import itertools
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm


class PatchedMultiImageDataset(ConcatDataset):

    @classmethod
    def from_paths(cls, paths, **kwargs):
        return cls([PatchedImageDataset(p, **kwargs) for p in paths])
    
    def __init__(self, datasets):
        assert all(isinstance(d, PatchedImageDataset) for d in datasets), 'All datasets must be PatchedImageDatasets.'

        self.annot = pd.concat([d.split_annot for d in datasets])
        super().__init__(datasets)   
    
    def num_images(self):
        return len(self.datasets)

    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{len(self.datasets)} image(s), ' \
            f'{len(self)} patches'
        return s

    @staticmethod
    def process_per_patch(dataloader, process_fn, collate_fn, max_prefetch=15000, progress=False):
        """ Process images in batches of patches and reconstruct the entire images.

        Args:
            dataloader (torch.util.data.DataLoader): Loader yielding batches of patches.
            process_fn (callable): batch -> processed_batch.
            collate_fn (callable): (image_info, List[(patch_info, processed_sample)]) -> image_results.
            max_prefetch (int, optional): Max batches to prefetch using threading. Defaults to 15000.
            progress (bool, optional): Whether to show a tqdm progress bar when iterating the dataloader. Defaults to False.
        """
        dataloader = tqdm(dataloader, dynamic_ncols=True, leave=False, disable=not progress)

        def process(batch):
            data, *patch_info = batch
            return patch_info, process_fn(data)
    
        processed_batches = map(process, dataloader)
        processed_batches = BackgroundGenerator(processed_batches, max_prefetch=max_prefetch)  # prefetch batches using threading

        # unbatch into samples
        def unbatch(batches):
            for patch_infos, processed_batch in batches:
                yield from zip(zip(*patch_infos), zip(*processed_batch))

        processed_samples = unbatch(processed_batches)

        # group by image_id (and image_hw for convenience) --> iterate over full images
        def grouper(sample):
            patch_info, _ = sample
            patch_hw, start_yx, image_hw, image_id = patch_info
            return image_id, tuple(image_hw.tolist())

        groups = itertools.groupby(processed_samples, key=grouper)
        
        # remove redundant image infos in groups
        def clean(patch):
            patch_info, processed_sample = patch
            patch_hw, start_yx, image_hw, image_id = patch_info
            return (patch_hw, start_yx), processed_sample

        # call the collate_fn for each image
        def collate(group):
            image_info, patches = group
            patches = map(clean, patches)
            return collate_fn(image_info, patches)
        
        yield from map(collate, groups)    


class PatchedImageDataset(Dataset):
    """ Dataset that provides per-patch iteration of a single big image file
        stored in TIFF or HDF5 format. """
    
    def __init__(
        self,
        path,
        split='all',
        patch_size=640,
        stride=None,
        random_offset=0,
        annotations=None,
        image_id=None,
        target_builder=None,
        transforms=None,
        max_cache_mem=None
    ):
        """ Dataset constructor.

        Args:
            path (str): Path to TIFF or HDF5 file; if HDF5, data must be stored in the hdf5 path '/data'.
            split (str, optional): Data split to return; available choices are 'left', 'right', or 'all'. Defaults to 'all'.
            patch_size (int, optional): Size of patches in which the data will be splitted; if None, the entire image is returned (no patch splitting). Defaults to 640.
            stride (int, optional): Stride for overlapping patches; None stands for non-overlapping patches. Defaults to None.
            random_offset (int, optional): The amount of random offset applied to patch origin; useful for randomizing data in the training phase. Defaults to 0.
            annotations (pd.DataFrame, optional): Dataframe containing points annotations; must be provided if target_builder != None. Defaults to None.
            image_id (str, optional): the ID of this image in the annotations; if None, the image name is used. Defaults to None.
            target_builder (obj, optional): A *TargetBuilder for building and returning also training targets. Defaults to None.
            transforms (callable, optional): A callable for applying transformations on patches. Defaults to None.
            max_cache_mem (int, optional): Cache size in bytes (only for HDF5). Defaults to None.
        """
        assert split in ('left', 'right', 'all'), "split must be one of ('left', 'right', 'all')"
        assert target_builder is None or annotations is not None, 'annotations must be != None if a target_builder is specified'
        
        self.path = Path(path)
        self.random_offset = random_offset
        self.target_builder = target_builder
        self.transforms = transforms
        self.split = split

        # hdf5 dataset
        if self.path.suffix.lower() in ('.h5', '.hdf5'):
            self.data = h5py.File(path, 'r', rdcc_nbytes=max_cache_mem)['data']
        else:  # image format
            image = io.imread(path).astype(np.float32)
            self.data = rgb2gray(image) if image.ndim > 2 else image  # TODO add custom transform params

        # patch size (height and width)
        self.patch_hw = np.array((patch_size, patch_size), dtype=np.int64) if patch_size else np.array(self.data.shape)

        # windows stride size (height and width)
        self.stride_hw = np.array((stride, stride), dtype=np.int64) if stride else self.patch_hw

        # size of the region from which we take patches
        image_hw = np.array(self.data.shape)
        image_half_hw = image_hw // np.array((1, 2))  # half only width
        if split == 'all':
            self.region_hw = image_hw
        elif split == 'left':
            self.region_hw = image_half_hw
        else:  # split == 'right':
            self.region_hw = image_hw - np.array((0, image_half_hw[1]))

        # the origin and limits of the region (split) of interest
        self.origin_yx = np.array((0, image_half_hw[1]) if self.split == 'right' else (0, 0))
        self.limits_yx = image_half_hw if self.split == 'left' else image_hw

        # keep only annotations of this image
        self.image_id = self.path.name if image_id is None else image_id
        self.annot = annotations.loc[self.image_id] if annotations is not None else pd.DataFrame(columns=['Y', 'X'])

        # keep also annotations in the selected split (in split's coordinates)
        in_split = ((self.annot[['Y', 'X']] >= self.origin_yx) & (self.annot[['Y', 'X']] < self.limits_yx)).all(axis=1)
        self.split_annot = self.annot[in_split].copy()
        self.split_annot[['Y', 'X']] -= self.origin_yx

        # the number of patches in a row and a column
        self.num_patches = np.ceil(1 + ((self.region_hw - self.patch_hw) / self.stride_hw)).astype(np.int64)
        
    def __len__(self):
        # total number of patches
        return self.num_patches.prod().item()
    
    def __getitem__(self, index):
        n_rows, n_cols = self.num_patches
        # row and col indices of the patch
        row_col_idx = np.array((index // n_cols, index % n_cols))

        # patch boundaries
        start_yx = self.origin_yx + self.stride_hw * row_col_idx
        if self.random_offset:
            start_yx += np.random.randint(-self.random_offset, self.random_offset, size=2)
            start_yx = np.clip(start_yx, (0, 0), self.limits_yx - self.patch_hw)
        end_yx = np.minimum(start_yx + self.patch_hw, self.limits_yx)
        (sy, sx), (ey, ex) = start_yx, end_yx

        # read patch
        patch = self.data[sy:ey, sx:ex] / np.array(255., dtype=np.float32)
        patch_hw = np.array(patch.shape)  # before padding

        # patch coordinates in the region space (useful for reconstructing the full region)
        local_start_yx = start_yx - self.origin_yx

        if self.target_builder:
            # gather annotations
            selector = self.annot.X.between(sx, ex) & self.annot.Y.between(sy, ey)
            locations = self.annot.loc[selector, ['Y', 'X']].values
            patch_locations = locations - start_yx

            # build target
            target = self.target_builder.build(patch_hw, patch_locations)

        # pad patch (in case of patches in last col/rows)
        py, px = - patch_hw % self.patch_hw
        pad = ((0, py), (0, px))

        patch = np.pad(patch, pad)  # defaults to zero padding

        if self.target_builder:
            datum = self.target_builder.pack(patch, target, pad=pad)
        else:
            datum = np.expand_dims(patch, axis=-1)  # add channels dimension

        if self.transforms:
            datum = self.transforms(datum)

        patch_info = (patch_hw, local_start_yx, self.region_hw, self.image_id)
        return (datum,) + patch_info


class RandomAccessMultiImageDataset(ConcatDataset):

    @classmethod
    def from_paths_and_locs(cls, paths, locs, **kwargs):
        return cls([RandomAccessImageDataset(p, l, **kwargs) for p, l in zip(paths, locs)])
    
    def __init__(self, datasets):
        assert all(isinstance(d, RandomAccessImageDataset) for d in datasets), 'All datasets must be RandomAccessImageDataset.'
        super().__init__(datasets)   
    
    def num_images(self):
        return len(self.datasets)

    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{len(self.datasets)} image(s), ' \
            f'{len(self)} patches'
        return s


class RandomAccessImageDataset(Dataset):
    """ Dataset that provides random access to patches belonging to a single big image file
        stored in TIFF or HDF5 format. """

    def __init__(
        self,
        path,
        locations,
        patch_size=64,
        transforms=None,
        max_cache_mem=None
    ):
        """ Constructor.

        Args:
            path (str): Path to image or HDF5 file.
            locations (ndarray): (N,2)-shaped array of YX location to extract patches from.
            patch_size (int, optional): Size of the patch to be extracted. Defaults to 16.
            transforms (callable, optional): A callable to apply transformations to patches. Defaults to None.
            max_cache_mem (int, optional): Cache size in bytes (only for HDF5). Defaults to None.
        """

        self.path = Path(path)
        self.locations = locations
        self.patch_size = patch_size
        self.transforms = transforms

        # hdf5 dataset
        if self.path.suffix.lower() in ('.h5', '.hdf5'):
            self.data = h5py.File(path, 'r', rdcc_nbytes=max_cache_mem)['data']
        else:  # image format
            image = io.imread(path).astype(np.float32)
            self.data = rgb2gray(image) if image.ndim > 2 else image  # TODO add custom transform params
        
        self.patch_hw = np.array((patch_size, patch_size), dtype=int)
        self.half_hw = self.patch_hw // 2
    
    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        y, x = self.locations[index]
        hy, hx = self.half_hw
        h, w = self.data.shape

        sy, sx = max(y - hy, 0), max(x - hx, 0)
        ey, ex = min(y + hy, h), min(x + hx, w)

        patch = self.data[sy:ey, sx:ex]
        patch_hw = np.array(patch.shape)
        py, px = - patch_hw % self.patch_hw

        if py or px:  # pad is needed
            pad = ((py if sy == 0 else 0, py if ey == h else 0),
                   (px if sx == 0 else 0, px if ex == w else 0))
            patch = np.pad(patch, pad)
        
        if self.transforms:
            patch = self.transforms(patch)
        
        return patch


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid, save_image
    from torchvision.transforms import ToTensor

    data_path = 'data/perineuronal-nets/test/'
    image_name = '061_A1_s08_C1_crop.tif'
    path = data_path + 'fullFrames/' + image_name
    locations = pd.read_csv(data_path + 'annotations.csv', index_col=0).loc[image_name, ['Y', 'X']].values
    dset = RandomAccessImageDataset(path, locations, transforms=ToTensor())

    samples = torch.stack([dset[i] for i in range(64)])

    img = make_grid(samples, padding=2, normalize=True)
    save_image(img, 'pnn_samples.png')
