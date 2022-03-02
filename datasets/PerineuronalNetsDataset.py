
import itertools
import pandas as pd

from pathlib import Path

from .patched_datasets import PatchedMultiImageDataset, PatchedImageDataset
from methods.segmentation.target_builder import SegmentationTargetBuilder
from methods.detection.target_builder import DetectionTargetBuilder
from methods.density.target_builder import DensityTargetBuilder


class PerineuronalNetsDataset(PatchedMultiImageDataset):
    """ Dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    def __init__(
        self,
        root='data/perineuronal-nets',
        split='all',
        patch_size=640,
        overlap=0,
        random_offset=None,
        target=None,
        target_params={},
        transforms=None,
        max_cache_mem=None
    ):

        self.root = Path(root)
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

        assert split in ('train-fold1245', 'train-fold3', 'train-half1', 'train-half2', 'test'), \
            "split must be one of ('train-fold1245', 'train-fold3', 'train-half1', 'train-half2', 'test')"
        self.split = split
        split_dir = split.split('-')[0]

        annot_path = self.root / split_dir / 'annotations.csv'
        all_annot = pd.read_csv(annot_path, index_col=0)

        image_files = sorted((self.root / split_dir / 'fullFramesH5').glob('*.h5'))
        assert len(image_files) > 0, "No images found"

        if max_cache_mem:
            max_cache_mem /= len(image_files)

        splits = ('all',)
        if self.split == 'train-fold1245':  # remove validation images from list: ttVtt, V removed
            del image_files[2::5]
        elif self.split == 'train-fold3':  # keep only validation elements: ttVtt, only V kept
            image_files = image_files[2::5]
        if self.split == 'train-half1':
            splits = ('left', 'right')
        elif self.split == 'train-half2':
            splits = ('right', 'left')
        elif self.split == 'test':
            pass
        splits = itertools.cycle(splits)

        stride = patch_size - self.overlap if patch_size else None
        kwargs = dict(
            patch_size=patch_size,
            stride=stride,
            random_offset=self.random_offset,
            annotations=all_annot,
            target_builder=target_builder,
            transforms=transforms,
            max_cache_mem=max_cache_mem,
        )
        image_ids = [i.with_suffix('.tif').name for i in image_files]
        datasets = [
            PatchedImageDataset(image_path, split=s, image_id=i, **kwargs)
            for image_path, s, i in zip(image_files, splits, image_ids)
        ]

        super().__init__(datasets)
    
    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{self.split} split, ' \
            f'{len(self.datasets)} images, ' \
            f'{len(self)} patches ({self.patch_size}x{self.patch_size})'
        return s
