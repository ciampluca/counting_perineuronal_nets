import collections
from pathlib import Path
import random

import pandas as pd

from datasets.patched_datasets import PatchedImageDataset, PatchedMultiImageDataset
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
            max_num_train_val_sample=30,  
            num_test_samples=10,   
            split_seed=None,
            num_samples=None,
            target_=None,
            target_params={},
            cache_targets=False,
            transforms=None,
            as_gray=False,
    ):

        target = target_  # XXX TOREMOVE for hydra bug

        assert target in (None, 'segmentation', 'detection', 'density'), f'Unsupported target type: {target}'
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



if __name__ == "__main__":
    from methods.density.utils import normalize_map
    from methods.points.utils import draw_points
    from skimage import io
    from tqdm import trange
    from methods.detection.transforms import RandomVerticalFlip, RandomHorizontalFlip, Compose
    from PIL import ImageDraw

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

    dataset = CellsDataset(target_='segmentation', root="data/vgg-cells",
                              target_params={
                                'radius': 5,         # radius (in px) of the dot placed on a cell in the segmentation map
                                'radius_ignore': 6,  # radius (in px) of the 'ignore' zone surrounding the cell
                                'v_bal': 0.1,         # weight of the loss of bg pixels
                                'sigma_bal': 3,       # gaussian stddev (in px) to blur loss weights of bg pixels near fg pixels
                                'sep_width': 1,       # width (in px) of bg ridge separating two overlapping foreground cells
                                'sigma_sep': 3,       # gaussian stddev (in px) to blur loss weights of bg pixels near bg ridge pixels
                                'lambda_sep': 50  
                              })
    datum, patch_hw, start_yx, image_hw, image_name = dataset[0]

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        segmentation_map = datum[:, :, 1]
        weights_map = datum[:, :, 2]

        segmentation_map = (255 * normalize_map(segmentation_map)).astype(np.uint8)
        weights_map = (255 * normalize_map(weights_map)).astype(np.uint8)
        io.imsave('debug/segm' + image_id, segmentation_map)
        io.imsave('debug/segm_weights' + image_id, weights_map)

        break