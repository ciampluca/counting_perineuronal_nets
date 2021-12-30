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

        n_train_samples, n_val_samples = self.num_samples
        if self.split == 'train':
            return image_paths[:n_train_samples]
        elif self.split == 'validation':
            return image_paths[n_train_samples:n_train_samples + n_val_samples]
        else:  # elif self.split == 'test':
            return image_paths[n_train_samples + n_val_samples:n_train_samples + n_val_samples + self.num_test_samples]



# Testing Code
if __name__ == "__main__":
    from methods.density.utils import normalize_map
    from methods.points.utils import draw_points
    from skimage import io
    from tqdm import trange
    from methods.detection.transforms import RandomVerticalFlip, RandomHorizontalFlip, Compose, ToTensor
    import torchvision.transforms
    import os

    # Check data loading for detection
    ######################################
    # Side --> MBM=20, VGG=12, BCD=30, ADIPOCYTE=12
    data_path="data/mbm-cells"
    side = 20
    target_params = {
        'side': side,
    }
    as_gray = False
    transforms = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
    dataset = CellsDataset(target_='detection', target_params=target_params, transforms=transforms, root=data_path, as_gray=as_gray)
    print(dataset)

    for i in trange(0, 40, 2):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, boxes = datum
        image = image.cpu().detach().permute(1, 2, 0).numpy()

        image = (255 * image).astype(np.uint8)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        image = draw_points(image, centers, radius=int(side/2))
        io.imsave(os.path.dirname(__file__) + '/trash/debug/annot_' + image_id, image)
        
        # break

    # Check data loading for segmentation
    ######################################
    # Radius --> MBM=12, VGG=5, BCD=15, ADIPOCYTE=5
    # Radius_Ignore --> MBM=15, VGG=6, BCD=18, ADIPOCYTE=6
    # Sigma_Bal --> MBM=5, VGG=3, BCD=7, ADIPOCYTE=3
    # Sep_Width --> MBM=1, VGG=1, BCD=2, ADIPOCYTE=1
    # Sigma_Sep --> MBM=4, VGG=3, BCD=8, ADIPOCYTE=3
    data_path="data/mbm-cells"
    radius = 12
    radius_ignore = 15
    sigma_bal = 5
    sep_width = 1
    sigma_sep = 4
    target_params = {
                        'radius': radius,         # radius (in px) of the dot placed on a cell in the segmentation map
                        'radius_ignore': radius_ignore,  # radius (in px) of the 'ignore' zone surrounding the cell
                        'v_bal': 0.1,         # weight of the loss of bg pixels
                        'sigma_bal': sigma_bal,       # gaussian stddev (in px) to blur loss weights of bg pixels near fg pixels
                        'sep_width': sep_width,       # width (in px) of bg ridge separating two overlapping foreground cells
                        'sigma_sep': sigma_sep,       # gaussian stddev (in px) to blur loss weights of bg pixels near bg ridge pixels
                        'lambda_sep': 50  
                    }
    as_gray = False
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])
    dataset = CellsDataset(target_='segmentation', root=data_path, target_params=target_params, transforms=transforms, as_gray=as_gray)
    print(dataset)

    for i in trange(0, 40, 2):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        n_channels = datum.shape[0]
        image, segmentation_map, weights_map = datum.split((n_channels - 2, 1, 1), dim=0)
        image, segmentation_map, weights_map = image.cpu().detach().permute(1, 2, 0).numpy(), segmentation_map.cpu().detach().permute(1, 2, 0).numpy(), weights_map.cpu().detach().permute(1, 2, 0).numpy()

        image = (255 * image).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/image_segm_' + image_id, image)
        segmentation_map = (255 * normalize_map(segmentation_map)).astype(np.uint8)
        weights_map = (255 * normalize_map(weights_map)).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/segm_' + image_id, segmentation_map)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/segm_weights_' + image_id, weights_map)

        # break
    
    # Check data loading for density
    ######################################
    # Sigma --> MBM=10, VGG=5, BCD=15, ADIPOCYTE=5, BCD=15
    # K_Size --> MBM=51, VGG=41, BCD=, ADIPOCYTE=41, BCD=81
    data_path="data/mbm-cells/train"
    sigma = 15
    k_size = 81
    target_params = {
        'k_size': k_size,
        'sigma': sigma,
        'method': 'reflect',
    }
    as_gray = False
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])
    dataset = CellsDataset(target_='density', target_params=target_params, transforms=transforms, root=data_path, as_gray=as_gray)
    print(dataset)
    
    for i in trange(0, 40, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        n_channels = datum.shape[0]
        image, dmap = datum.split((n_channels - 1, 1), dim=0)
        image, dmap = image.cpu().detach().permute(1, 2, 0).numpy(), dmap.cpu().detach().permute(1, 2, 0).numpy()

        image = (255 * image).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/image_den_' + image_id, image)
        dmap = (255 * normalize_map(dmap.squeeze())).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/den_' + image_id, dmap)
    
        # break

