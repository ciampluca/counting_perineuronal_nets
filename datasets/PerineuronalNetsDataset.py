
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
        target_=None,
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

        target = target_  # XXX TOREMOVE for hydra bug
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
            max_cache_mem=max_cache_mem
        )
        datasets = [PatchedImageDataset(image_path, split=s, **kwargs) for image_path, s in zip(image_files, splits)]

        super().__init__(datasets, transforms)
    
    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{self.split} split, ' \
            f'{len(self.datasets)} images, ' \
            f'{len(self)} patches ({self.patch_size}x{self.patch_size})'
        return s


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from detection.transforms import ToTensor
    from detection.utils import collate_fn, build_coco_compliant_batch
    from tqdm import tqdm

    data_root = 'data/perineuronal-nets'

    for split in ('train-fold1245', 'train-fold3', 'train-half1', 'train-half2', 'test'):
        dataset = PerineuronalNetsDataset(data_root, split=split)
        print(split, len(dataset))

    dataset = PerineuronalNetsDataset(data_root, split='test', patch_size=None, overlap=0, random_offset=320, target_='detection', transforms=ToTensor(), max_cache_mem=8*1024**3)  # bytes = 8 GiB
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        coco_batch = build_coco_compliant_batch(batch[0])
        import pdb; pdb.set_trace()