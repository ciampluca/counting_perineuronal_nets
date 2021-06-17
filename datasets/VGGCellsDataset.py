import collections
from pathlib import Path
import random

import numpy as np
import pandas as pd
from skimage import io

from .patched_datasets import PatchedImageDataset, PatchedMultiImageDataset
from segmentation.target_builder import SegmentationTargetBuilder
from detection.target_builder import DetectionTargetBuilder
from density.target_builder import DensityTargetBuilder


class VGGCellsDataset(PatchedMultiImageDataset):

    def __init__(
        self,
        root='data/vgg-cells',
        split='all',
        split_seed=None,
        num_samples=None,
        target_=None,
        target_params={},
        transforms=None,
    ):

        target = target_  # XXX TOREMOVE for hydra bug
        
        assert target in (None, 'segmentation', 'detection', 'density'), f'Unsupported target type: {target}'
        assert split in ('all', 'train', 'validation', 'test'), "Split must be one of ('train', 'validation', 'test', 'all')"
        assert split == 'all' or ((split_seed is not None) and (num_samples is not None)), "You must supply split_seed and num_samples when split != 'all'"
        assert split == 'all' or (isinstance(num_samples, collections.abc.Sequence) and len(num_samples) == 2), 'num_samples must be a tuple of two ints'
        assert split == 'all' or sum(num_samples) <= 100, 'n_train + n_val samples must be <= 100'
        
        self.root = Path(root)

        self.split = split
        self.split_seed = None
        self.num_samples = num_samples

        self.transforms = transforms

        self.target = target
        self.target_params = target_params

        if target == 'segmentation':
            target_builder = SegmentationTargetBuilder
        elif target == 'detection':
            target_builder = DetectionTargetBuilder
        elif target == 'density':
            target_builder = DensityTargetBuilder
        
        self.target_builder = target_builder(**target_params) if target else None

        # get list of images in the given split
        self.image_paths = self._get_images_in_split()

        # create pandas dataframe containing dot annotations (to be compliant with other implementation)
        self.annot = self._load_annotations()

        data_params = dict(
            split='all',
            patch_size=None,
            annotations=self.annot,
            target_builder=self.target_builder,
            transforms=self.transforms,
        )
        datasets = [PatchedImageDataset(p, **data_params) for p in self.image_paths]
        super().__init__(datasets)
            
    def __len__(self):
        return len(self.image_paths)
    
    def _get_images_in_split(self):
        image_paths = self.root.glob('*cell.png')
        image_paths = sorted(image_paths)

        if self.split == 'all':
            return image_paths
        
        # reproducible shuffle
        random.Random(self.split_seed).shuffle(image_paths)

        n_train_samples, n_val_samples = self.num_samples
        if self.split == 'train':
            return image_paths[:n_train_samples]
        elif self.split == 'validation':
            return image_paths[n_train_samples:n_train_samples + n_val_samples]
        else:  # elif self.split == 'test':
            return image_paths[n_train_samples + n_val_samples:]

    def _load_annotations(self):

        def _load_one_annotation(image_name):
            image_id = image_name.name
            label_map_path = self.root / image_id.replace('cell', 'dots')
            label_map = io.imread(label_map_path)[:, :, 0]  # / 255
            y, x = np.nonzero(label_map)
            annot = pd.DataFrame({'Y': y, 'X': x})
            annot['imgName'] = image_id
            return annot

        annot = map(_load_one_annotation, self.image_paths)
        annot = pd.concat(annot, ignore_index=True)
        annot = annot.set_index('imgName')
        return annot


if __name__ == "__main__":
    from density.utils import normalize_map
    from points.utils import draw_points
    from skimage import io
    from tqdm import trange

    dataset = VGGCellsDataset(target_='detection', target_params={'side': 12})
    print(dataset)

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, boxes = datum   

        image = (255 * image.squeeze()).astype(np.uint8)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        image = draw_points(image, centers, radius=5)

        io.imsave('trash/debug/annot' + image_id, image)

    dataset = VGGCellsDataset(target_='density', target_params={'k_size': 33, 'sigma': 5, 'method': 'normalize'})
    datum, patch_hw, start_yx, image_hw, image_name = dataset[0]

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        density_map = datum[:, :, 1]

        assert np.allclose(density_map.sum(), len(dataset.annot.loc[image_id]))

        density_map = (255 * normalize_map(density_map)).astype(np.uint8)
        io.imsave('trash/debug/dmap' + image_id, density_map)

