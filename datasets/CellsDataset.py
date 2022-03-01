import collections
from pathlib import Path
import random

import numpy as np
import pandas as pd
from skimage import io

from datasets.patched_datasets import PatchedImageDataset, PatchedMultiImageDataset
from methods.segmentation.target_builder import SegmentationTargetBuilder
from methods.detection.target_builder import DetectionTargetBuilder
from methods.density.target_builder import DensityTargetBuilder
from methods.countmap.target_builder import CountmapTargetBuilder


class CellsDataset(PatchedMultiImageDataset):
    """ Dataset class implementation for the following cells datasets: VGGCells, MBMCells, AdipocyteCells
    """

    def __init__(
            self,
            root='data/vgg-cells',
            split='all',
            max_num_train_val_sample=30,   # should be 100 for VGGCells and 30 for MBMCells
            num_test_samples=10,    # should be 100 for VGGCells and 10 for MBMCells
            split_seed=None,
            num_samples=None,
            target_=None,
            target_params={},
            cache_targets=False,
            transforms=None,
            as_gray=False,
    ):

        target = target_  # XXX TOREMOVE for hydra bug

        assert target in (None, 'segmentation', 'detection', 'density', 'countmap'), f'Unsupported target type: {target}'
        assert split in (
        'all', 'train', 'validation', 'test'), "Split must be one of ('train', 'validation', 'test', 'all')"
        assert split == 'all' or ((split_seed is not None) and (
                    num_samples is not None)), "You must supply split_seed and num_samples when split != 'all'"
        assert split == 'all' or (isinstance(num_samples, collections.abc.Sequence) and len(
            num_samples) == 2), 'num_samples must be a tuple of two ints'
        assert split == 'all' or sum(abs(n) for n in num_samples) <= max_num_train_val_sample, \
            f'n_train + n_val samples must be <= {max_num_train_val_sample}'

        self.root = Path(root)

        self.split = split
        self.split_seed = split_seed
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples

        self.transforms = transforms
        self.as_gray = as_gray

        self.target = target
        self.target_params = target_params

        if target == 'segmentation':
            target_builder = SegmentationTargetBuilder
        elif target == 'detection':
            target_builder = DetectionTargetBuilder
        elif target == 'density':
            target_builder = DensityTargetBuilder
        elif target == 'countmap':
            target_builder = CountmapTargetBuilder

        self.target_builder = target_builder(**target_params) if target else None

        # get list of images in the given split
        self.image_paths = self._get_images_in_split()

        # load pandas dataframe containing dot annotations
        self.annot = pd.read_csv(Path(self.root / 'annotations.csv'))
        self.annot = self.annot.set_index('imgName')

        data_params = dict(
            split='all',
            patch_size=None,
            annotations=self.annot,
            target_builder=self.target_builder,
            transforms=self.transforms,
            cache_targets=cache_targets,
            as_gray=as_gray,
        )
        datasets = [PatchedImageDataset(p, **data_params) for p in self.image_paths]
        super().__init__(datasets)

    def __len__(self):
        return len(self.image_paths)

    def _get_images_in_split(self):
        image_paths = self.root.glob('imgs/*cell.*')
        image_paths = sorted(image_paths)

        if self.split == 'all':
            return image_paths

        # reproducible shuffle
        random.Random(self.split_seed).shuffle(image_paths)

        n_train, n_val = self.num_samples
        if self.split == 'train':
            start, end = (None, n_train) if n_train >= 0 else (n_train, None)

        elif self.split == 'validation':
            if n_train >= 0 and n_val >= 0:
                start, end = n_train, n_train + n_val
            elif n_train >= 0 and n_val < 0:
                start, end = n_val, None
            elif n_train < 0 and n_val >= 0:
                start, end = None, n_val
            else:  # n_train_samples < 0 and n_val_samples < 0:
                start, end = n_train + n_val, n_train

        else:  # elif self.split == 'test':
            if n_train >= 0 and n_val >= 0:
                start, end = n_train + n_val, n_train + n_val + self.num_test_samples
            elif n_train >= 0 and n_val < 0:
                start, end = n_train, n_train + self.num_test_samples
            elif n_train < 0 and n_val >= 0:
                start, end = n_val, n_val + self.num_test_samples
            else:  # n_train_samples < 0 and n_val_samples < 0:
                start, end = None, self.num_test_samples

        return image_paths[start:end]
