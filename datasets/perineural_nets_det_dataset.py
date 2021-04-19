from pathlib import Path
from copy import deepcopy
import itertools
import pandas as pd
import numpy as np
import h5py
from PIL import ImageDraw
import os

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.functional import to_pil_image


class PerineuralNetsDetDataset(ConcatDataset):
    """ Dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    # params for groundtruth segmentation maps generation
    DEFAULT_GT_PARAMS = {
        'side': 60,         # side (in px) of the bounding box localizing a cell
    }

    def __init__(self,
                 root='data/perineuronal_nets',
                 split='all',
                 with_targets=True,
                 patch_size=640,
                 overlap=None,
                 random_offset=None,
                 gt_params={},
                 transforms=None,
                 max_cache_mem=None):

        self.root = Path(root)
        self.transforms = transforms
        self.patch_size = patch_size
        self.random_offset = random_offset if random_offset is not None else patch_size // 2

        # groundtruth parameters
        self.gt_params = deepcopy(self.DEFAULT_GT_PARAMS)
        self.gt_params.update(gt_params)

        self.overlap = overlap if overlap is not None else 0

        assert split in ('train', 'validation', 'train-specular', 'validation-specular', 'all'), \
            "split must be one of ('train', 'validation', 'train-specular', 'validation-specular', 'all')"
        self.split = split

        annot_path = self.root / 'annotation' / 'annotations.csv'
        self.annot = pd.read_csv(annot_path, index_col=0)

        image_files = sorted((self.root / 'fullFramesH5').glob('*.h5'))
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
            with_targets=with_targets,
            patch_size=patch_size,
            stride=stride,
            random_offset=self.random_offset,
            gt_params=self.gt_params,
            max_cache_mem=max_cache_mem
        )
        datasets = [_PerineuralNetsDetImage(image_path, self.annot, split=s, **kwargs) for image_path, s in zip(image_files, splits)]
        super(self.__class__, self).__init__(datasets)

    def __getitem__(self, index):
        sample = super(self.__class__, self).__getitem__(index)

        if self.transforms:
            sample = (self.transforms(sample[0]),) + sample[1:]

        return sample

    def custom_collate_fn(self, batch):
        return list(zip(*batch))

    def build_coco_compliant_batch(self, image_and_target_batch):
        targets, imgs = [], []

        for b in image_and_target_batch:
            imgs.append(b[0])

            if b[1].size != 0:
                target = {
                    'boxes': torch.as_tensor([[bb[1], bb[0], bb[3], bb[2]] for bb in b[1]], dtype=torch.float32),
                    'labels': torch.ones((len(b[1]),), dtype=torch.int64),  # there is only one class
                    'iscrowd': torch.zeros((len(b[1]),), dtype=torch.int64),     # suppose all instances are not crowd
                }
            else:
                target = {
                    'boxes': torch.as_tensor([[]], dtype=torch.float32),
                    'labels': torch.as_tensor([[]], dtype=torch.int64),
                    'iscrowd': torch.as_tensor([[]], dtype=torch.int64),
                }

            targets.append(target)

        return imgs, targets


class _PerineuralNetsDetImage(Dataset):
    """ Dataset that provides per-patch iteration of a single big image file. """

    def __init__(self,
                 h5_path,
                 annotations,
                 split='left',
                 with_targets=True,
                 patch_size=640,
                 stride=None,
                 random_offset=0,
                 gt_params=None,
                 max_cache_mem=None):
        self.h5_path = h5_path
        self.random_offset = random_offset
        self.with_targets = with_targets
        self.gt_params = gt_params if gt_params is not None else PerineuralNetsSegmDataset.DEFAULT_GT_PARAMS

        assert split in ('left', 'right', 'all'), "split must be one of ('left', 'right', 'all')"
        self.split = split

        # keep only annotations of this image
        self.image_id = Path(h5_path).with_suffix('.tif').name
        self.annot = annotations.loc[self.image_id]

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

        if self.with_targets:
            # gather annotations
            selector = self.annot.X.between(sx, ex) & self.annot.Y.between(sy, ey)
            locations = self.annot.loc[selector, ['Y', 'X']].values
            patch_locations = locations - start_yx

            # build target
            detection_target = self._build_detection_target(patch, patch_locations)

        # pad patch (in case of patches in last col/rows)
        py, px = - patch_hw % self.patch_hw
        pad = ((0, py), (0, px))

        patch = np.pad(patch, pad)  # defaults to zero padding

        if self.with_targets:
            # put in a unique tuple the patch and the target
            patch = np.expand_dims(patch, axis=-1)  # add channels dimension
            input_and_target = (patch, detection_target)
            datum = input_and_target
        else:
            datum = np.expand_dims(patch, axis=-1)  # add channels dimension

        return datum, patch_hw, local_start_yx, self.region_hw, self.image_id

    def _build_detection_target(self, patch, locations):
        """ This builds the detection target
        """
        side = self.gt_params['side']
        half_side = side / 2

        shape = patch.shape

        if len(locations) == 0:  # empty patch
            bbs = np.array([[]], dtype=np.float32)
        else:
            # bb format: [y1, x1, y2, x2]
            bbs = np.empty((locations.shape[0], 4), dtype=np.float32)
            for i, center in enumerate(locations):
                bbs[i] = [center[0]-half_side, center[1]-half_side, center[0]+half_side, center[1]+half_side]
                np.clip(bbs[i], [0, 0, 0, 0], [shape[0], shape[1], shape[0], shape[1]], out=bbs[i])

        return bbs


# Debug code
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from det_transforms import Compose, RandomHorizontalFlip, ToTensor
    from tqdm import tqdm

    def is_empty(l):
        return all(is_empty(i) if isinstance(i, list) else False for i in l)

    data_root = '/home/luca/luca-cnr/mnt/datino/perineural_nets'
    device = "cpu"
    output_folder = "output/gt/bbs_patches"

    for split in ('train', 'validation', 'train-specular', 'validation-specular', 'all'):
        dataset = PerineuralNetsDetDataset(data_root, split=split)
        print(split, len(dataset))

    dataset = PerineuralNetsDetDataset(data_root, split='all', patch_size=640, overlap=120, random_offset=320, with_targets=True, transforms=Compose([RandomHorizontalFlip(), ToTensor()]), max_cache_mem=8*1024**3)  # bytes = 8 GiB
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=dataset.custom_collate_fn)

    for i, (img_and_target, _, _, _, image_id) in enumerate(tqdm(dataloader)):
        imgs, targets = dataset.build_coco_compliant_batch(img_and_target)
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for img, target, img_id in zip(imgs, targets, image_id):
            img_name = img_id.rsplit(".", 1)[0] + "_{}.png".format(i)
            bboxes = target['boxes'].tolist()

            pil_image = to_pil_image(img.cpu())
            draw = ImageDraw.Draw(pil_image)
            if not is_empty(bboxes):
                for bb in bboxes:
                    draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline='red', width=3)
            pil_image.save(os.path.join(output_folder, img_name))
