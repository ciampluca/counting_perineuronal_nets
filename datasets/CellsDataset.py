import collections
import itertools
import logging
from pathlib import Path
import random

import pandas as pd

from datasets.patched_datasets import PatchedImageDataset, PatchedMultiImageDataset
from methods.segmentation.target_builder import SegmentationTargetBuilder
from methods.detection.target_builder import DetectionTargetBuilder
from methods.density.target_builder import DensityTargetBuilder

log = logging.getLogger(__name__)


class CellsDataset(PatchedMultiImageDataset):
    """ Dataset class implementation for the following cells datasets: VGG, MBM, ADI, BCD
    """

    def __init__(
            self,
            root='data/vgg-cells',
            split='all',
            max_num_train_val_sample=30,
            num_test_samples=10,
            split_seed=None,
            num_samples=None,
            target=None,
            target_params={},
            target_cache=None,
            transforms=None,
            as_gray=False,
    ):
        assert target in (None, 'segmentation', 'detection', 'density'), f'Unsupported target type: {target}'
        assert split in (
        'all', 'train', 'validation', 'test'), "Split must be one of ('train', 'validation', 'test', 'all')"
        assert split == 'all' or ((split_seed is not None) and (
                    num_samples is not None)), "You must supply split_seed and num_samples when split != 'all'"
        assert split == 'all' or (isinstance(num_samples, collections.abc.Sequence) and len(
            num_samples) == 2), 'num_samples must be a tuple of two ints'
        assert split == 'all' or sum(num_samples) <= max_num_train_val_sample, \
            f'n_train + n_val samples must be <= {max_num_train_val_sample}'

        self.root = Path(root)

        self.split = split
        self.split_seed = None
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples

        self.transforms = transforms
        self.as_gray = as_gray

        self.target = target
        self.target_params = target_params
        self.target_cache = (self.root / 'cache') if target_cache is None else target_cache

        if target == 'segmentation':
            target_builder = SegmentationTargetBuilder
        elif target == 'detection':
            target_builder = DetectionTargetBuilder
        elif target == 'density':
            target_builder = DensityTargetBuilder

        self.target_builder = target_builder(**target_params) if target else None

        # get list of images in the given split
        self.image_paths = self._get_images_in_split()
        self.target_cache_paths = self._get_cache_paths()

        # load pandas dataframe containing dot annotations
        all_annot = pd.read_csv(Path(self.root / 'annotations.csv'))
        all_annot = all_annot.set_index('imgName')
        if not 'class' in all_annot.columns:
            all_annot['class'] = 0
        num_classes = all_annot['class'].nunique()

        data_params = dict(
            split='all',
            patch_size=None,
            annotations=all_annot,
            target_builder=self.target_builder,
            transforms=self.transforms,
            as_gray=as_gray,
            num_classes=num_classes,
        )
        datasets = [PatchedImageDataset(p, target_cache=c, **data_params) for p, c in zip(self.image_paths, self.target_cache_paths)]
        
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
            return image_paths[:n_train]
        elif self.split == 'validation':
            return image_paths[n_train:n_train + n_val]
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

    def _get_cache_paths(self):
        if not self.target or not self.target_cache:
            return itertools.repeat(None, len(self.image_paths))

        def stringify(builder):
            tokens = [k[0] + ('-' if isinstance(v, str) else '') + str(v) for k, v in builder.__dict__.items()]
            return '_'.join(tokens)

        cache_root = Path(self.target_cache)
        cache_name = f'{self.target}_{stringify(self.target_builder)}'
        cache_dir = cache_root / cache_name

        log.info(f'Using cache: {cache_dir}')

        cache_paths = [cache_dir / p.stem for p in self.image_paths]
        return cache_paths
