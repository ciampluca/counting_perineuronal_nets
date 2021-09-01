import collections
from pathlib import Path
import random

import numpy as np
import pandas as pd
from skimage import io

from .patched_datasets import PatchedImageDataset, PatchedMultiImageDataset
from methods.segmentation.target_builder import SegmentationTargetBuilder
from methods.detection.target_builder import DetectionTargetBuilder
from methods.density.target_builder import DensityTargetBuilder


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
            transforms=None,
    ):

        target = target_  # XXX TOREMOVE for hydra bug

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
        image_paths = self.root.glob('*cell.*')
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
            return image_paths[n_train_samples + n_val_samples:n_train_samples + n_val_samples + self.num_test_samples]

    def _load_annotations(self):

        def _load_one_annotation(image_name):
            image_id = image_name.name
            label_map_path = self.root / image_id.replace('cell', 'dots')
            label_map = io.imread(label_map_path)
            if label_map.ndim == 3:
                label_map = label_map[:, :, 0]
            y, x = np.where((label_map == 255) | (label_map == 254))
            annot = pd.DataFrame({'Y': y, 'X': x})
            annot['imgName'] = image_id
            return annot

        annot = map(_load_one_annotation, self.image_paths)
        annot = pd.concat(annot, ignore_index=True)
        annot = annot.set_index('imgName')
        return annot


if __name__ == "__main__":
    from methods.density.utils import normalize_map
    from methods.points.utils import draw_points
    from skimage import io
    from tqdm import trange
    from methods.detection.transforms import RandomVerticalFlip, RandomHorizontalFlip, Compose
    import torchvision
    from PIL import ImageDraw, Image

    # vgg-cells --> side: 12, mbm-cells --> side: 20
    side = 20
    transforms = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ])
    dataset = CellsDataset(target_='detection', target_params={'side': side}, transforms=None, root="/home/luca/luca-cnr/mnt/datino/MBM_cells")
    print(dataset)

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, boxes = datum

        image = (255 * image.squeeze()).astype(np.uint8)
        img_draw = ImageDraw.Draw(image)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        image = draw_points(image, centers, radius=int(side/2))

        io.imsave('trash/debug/annot' + image_id, image)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]
    )
    dataset = CellsDataset(target_='density', transforms=transforms, root="/home/luca/luca-cnr/mnt/datino/VGG_cells",
                              target_params={'k_size': 31, 'sigma': int(side/2), 'method': 'reflect'})
    datum, patch_hw, start_yx, image_hw, image_name = dataset[0]

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        datum = datum.permute(1, 2, 0).numpy()
        density_map = datum[:, :, 1]

        assert np.allclose(density_map.sum(), len(dataset.annot.loc[image_id]))

        density_map = (255 * normalize_map(density_map)).astype(np.uint8)
        io.imsave('trash/debug/dmap' + image_id, density_map)

