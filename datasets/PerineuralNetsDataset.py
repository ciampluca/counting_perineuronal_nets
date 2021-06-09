
import itertools
import numpy as np
import pandas as pd
import h5py

from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

from segmentation.target_builder import SegmentationTargetBuilder
from detection.target_builder import DetectionTargetBuilder
from density.target_builder import DensityTargetBuilder


class PerineuralNetsDataset(ConcatDataset):
    """ Dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    def __init__(self,
                 root='data/perineuronal_nets',
                 split='all',
                 patch_size=640,
                 overlap=0,
                 random_offset=None,
                 target=None,
                 target_params={},
                 transforms=None,
                 max_cache_mem=None):

        self.root = Path(root)
        self.transforms = transforms
        self.patch_size = patch_size
        # TODO move overlap in run config
        # self.overlap = overlap if overlap is not None else int(4 * self.gt_params['radius_ignore'])
        self.overlap = overlap
        self.random_offset = random_offset if random_offset is not None else patch_size // 2

        assert target in (None, 'segmentation', 'detection', 'density'), f'Unsupported target type: {target}'
        self.target = target

        if target == 'segmentation':
            target_builder = SegmentationTargetBuilder
        elif target == 'detection':
            target_builder = DetectionTargetBuilder
        elif target == 'density':
            target_builder = DensityTargetBuilder

        target_builder = target_builder(**target_params) if target else None

        assert split in ('train', 'validation', 'train-specular', 'validation-specular', 'all'), \
            "split must be one of ('train', 'validation', 'train-specular', 'validation-specular', 'all')"
        self.split = split

        annot_path = self.root / 'annotation' / 'annotations.csv'
        all_annot = pd.read_csv(annot_path, index_col=0)

        image_files = sorted((self.root / 'fullFramesH5').glob('*.h5'))[:1]  ## REMOVE! FO DEBUG ONLY
        assert len(image_files) > 0, "No images found"

        if max_cache_mem:
            max_cache_mem /= len(image_files)

        splits = ('all',)
        if self.split == 'train':  # remove validation images from list: ttVtt, V removed
            del image_files[2::5]
        elif self.split == 'validation':  # keep only validation elements: ttVtt, only V kept
            image_files = image_files[2::5]
        if self.split == 'train-specular':
            splits = ('left', 'right')
        elif self.split == 'validation-specular':
            splits = ('right', 'left')
        # elif self.split == 'all':
            # pass
        splits = itertools.cycle(splits)

        stride = patch_size - self.overlap
        kwargs = dict(
            target_builder=target_builder,
            patch_size=patch_size,
            stride=stride,
            random_offset=self.random_offset,
            max_cache_mem=max_cache_mem
        )
        datasets = [_PerineuralNetsImage(image_path, all_annot, split=s, **kwargs) for image_path, s in zip(image_files, splits)]
        self.annot = pd.concat([d.split_annot for d in datasets])

        super(self.__class__, self).__init__(datasets)

    def __getitem__(self, index):
        sample = super(self.__class__, self).__getitem__(index)

        if self.transforms:
            sample = (self.transforms(sample[0]),) + sample[1:]

        return sample       


class _PerineuralNetsImage(Dataset):
    """ Dataset that provides per-patch iteration of a single big image file. """
    
    def __init__(self,
                 h5_path,
                 annotations,
                 split='left',
                 patch_size=640,
                 stride=None,
                 random_offset=0,
                 target_builder=None,
                 max_cache_mem=None):
        
        self.h5_path = h5_path
        self.random_offset = random_offset
        self.target_builder = target_builder
        
        assert split in ('left', 'right', 'all'), "split must be one of ('left', 'right', 'all')"
        self.split = split

        # patch size (height and width)
        self.patch_hw = np.array((patch_size, patch_size), dtype=np.int64)

        # windows stride size (height and width)
        self.stride_hw = np.array((stride, stride), dtype=np.int64) if stride else self.patch_hw

        # hdf5 dataset
        self.data = h5py.File(h5_path, 'r', rdcc_nbytes=max_cache_mem)['data']

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
        self.image_id = Path(h5_path).with_suffix('.tif').name
        self.annot = annotations.loc[self.image_id]

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
            target = self.target_builder.build(patch, patch_locations)

        # pad patch (in case of patches in last col/rows)
        py, px = - patch_hw % self.patch_hw
        pad = ((0, py), (0, px))

        patch = np.pad(patch, pad)  # defaults to zero padding

        if self.target_builder:
            datum = self.target_builder.pack(patch, target, pad=pad)
        else:
            datum = np.expand_dims(patch, axis=-1)  # add channels dimension

        patch_info = (patch_hw, local_start_yx, self.region_hw, self.image_id)
        return datum, *patch_info


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils.detection import ToTensor, collate_fn, build_coco_compliant_batch
    from tqdm import tqdm

    data_root = 'data/perineuronal_nets'

    for split in ('train', 'validation', 'train-specular', 'validation-specular', 'all'):
        dataset = PerineuralNetsDataset(data_root, split=split)
        print(split, len(dataset))

    dataset = PerineuralNetsDataset(data_root, split='all', patch_size=640, overlap=120, random_offset=320, target='detection', transforms=ToTensor(), max_cache_mem=8*1024**3)  # bytes = 8 GiB
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        coco_batch = build_coco_compliant_batch(batch[0])
        import pdb; pdb.set_trace()