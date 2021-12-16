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
            annot_transforms=None,
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
            cache_targets=cache_targets
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
    import torchvision.transforms
    import os

    # Check 2-fold cross-validation splits for nuclei dataset
    fold1 = CellsDataset(root="data/nuclei-cells", split='train', split_seed=13, num_samples=(50, -50), max_num_train_val_sample=100)
    fold2 = CellsDataset(root="data/nuclei-cells", split='validation', split_seed=13, num_samples=(-50, 50), max_num_train_val_sample=100)

    f1 = set(map(str, fold1.image_paths))
    f2 = set(map(str, fold2.image_paths))
    assert f1 == f2

    # Check data loading for detection
    # Radius --> MBM=10, VGG=6, DCC=15, NUCLEI=9, BCD=15, HeLa=12, PSU=12, ADIPOCYTE=5
    data_path="data/mbm-cells"
    radius = 10
    transforms = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ])
    dataset = CellsDataset(target_='detection', target_params={'side': radius*2}, transforms=transforms, root=data_path)
    print(dataset)

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, boxes = datum

        image = (255 * image.squeeze()).astype(np.uint8)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        image = draw_points(image, centers, radius=int(radius))
        io.imsave(os.path.dirname(__file__) + '/trash/debug/annot_' + image_id, image)
        
        break

    # Check data loading for segmentation
    data_path="data/vgg-cells"
    target_params = {
                        'radius': 5,         # radius (in px) of the dot placed on a cell in the segmentation map
                        'radius_ignore': 6,  # radius (in px) of the 'ignore' zone surrounding the cell
                        'v_bal': 0.1,         # weight of the loss of bg pixels
                        'sigma_bal': 3,       # gaussian stddev (in px) to blur loss weights of bg pixels near fg pixels
                        'sep_width': 1,       # width (in px) of bg ridge separating two overlapping foreground cells
                        'sigma_sep': 3,       # gaussian stddev (in px) to blur loss weights of bg pixels near bg ridge pixels
                        'lambda_sep': 50  
                    }
    dataset = CellsDataset(target_='segmentation', root=data_path, target_params=target_params)
    datum, patch_hw, start_yx, image_hw, image_name = dataset[0]

    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        segmentation_map = datum[:, :, 1]
        weights_map = datum[:, :, 2]

        segmentation_map = (255 * normalize_map(segmentation_map)).astype(np.uint8)
        weights_map = (255 * normalize_map(weights_map)).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/segm_' + image_id, segmentation_map)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/segm_weights_' + image_id, weights_map)

        break
    
    # Check data loading for density
    # Radius --> MBM=10, VGG=6, DCC=15, NUCLEI=9, BCD=15, HeLa=12, PSU=12, ADIPOCYTE=5
    data_path="data/hela-cells/train"
    radius = 12
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])
    dataset = CellsDataset(target_='density', target_params={'k_size': 51, 'sigma': radius}, transforms=transforms, root=data_path)
    print(dataset)
    
    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, dmap = datum.split(1, dim=0)
        image, dmap = image.cpu().detach().numpy(), dmap.cpu().detach().numpy()

        image = (255 * image.squeeze()).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/image_den_' + image_id, image)
        dmap = (255 * normalize_map(dmap.squeeze())).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/den_' + image_id, dmap)
    
        break
    
    # Check data loading for count-ception
    # Scale --> MBM=2, VGG=1, DCC=, NUCLEI=, BCD=, HeLa=, PSU=, ADIPOCYTE=
    # Target-patch-size --> MBM=32, VGG=32, DCC=, NUCLEI=, BCD=, HeLa=, PSU=, ADIPOCYTE=
    data_path="data/vgg-cells"
    scale = 2
    target_patch_size = 32
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])
    target_params = {
                        'scale': scale,       
                        'target_patch_size': target_patch_size, 
                        'stride': 1         
                    }
    dataset = CellsDataset(target_='countmap', target_params=target_params, transforms=transforms, root=data_path)
    print(dataset)
    
    for i in trange(0, 200, 5):
        datum, patch_hw, start_yx, image_hw, image_id = dataset[i]
        image, label = datum.split(1, dim=0)
        image, label = image.cpu().detach().numpy().squeeze(), label.cpu().detach().numpy().squeeze()
        
        gt_count = (label / (target_patch_size ** 2.0)).sum()
        image_hw = int(image_hw[0] / scale), int(image_hw[1] / scale)
        pad_to_remove_hw = int((image.shape[0]-image_hw[0])/2), int((image.shape[1]-image_hw[1])/2)
        image = image[pad_to_remove_hw[0]:image.shape[0]-pad_to_remove_hw[0], pad_to_remove_hw[1]:image.shape[1]-pad_to_remove_hw[1]]
        image = (255 * image).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/image_countception_' + image_id, image)
        label = (255 * normalize_map(label)).astype(np.uint8)
        io.imsave(os.path.dirname(__file__) + '/trash/debug/label_countception_' + image_id, label)
        
        #break
