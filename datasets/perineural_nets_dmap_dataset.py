from pathlib import Path
from copy import deepcopy
import itertools
import pandas as pd
import numpy as np
import h5py
import os
import cv2
from math import floor

from torch.utils.data import Dataset, ConcatDataset


class PerineuralNetsDMapDataset(ConcatDataset):
    """ Dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    # params for groundtruth segmentation maps generation
    DEFAULT_GT_PARAMS = {
        'k_size': 51,         # size (in px) of the kernel of the gaussian localizing a perineural nets
        'sigma': 30,            # sigma of the gaussian
    }

    def __init__(self, root='data/perineuronal_nets', split='all', with_targets=True, patch_size=640, overlap=None,
                 random_offset=None, gt_params={}, transforms=None, max_cache_mem=None, border_pad=None):

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
        all_annot = pd.read_csv(annot_path, index_col=0)

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

        # border pad (for validation, useful for reconstructing the image)
        if border_pad is not None:
            assert border_pad % 32 == 0, "Border pad value must be divisible by 32"
            self.border_pad = border_pad

        datasets = [_PerineuralNetsDMapImage(image_path, all_annot, split=s, **kwargs) for image_path, s in
                    zip(image_files, splits)]
        self.annot = pd.concat([d.split_annot for d in datasets])

        super(self.__class__, self).__init__(datasets)

    def __getitem__(self, index):
        sample = super(self.__class__, self).__getitem__(index)

        if self.transforms:
            sample = (self.transforms(sample[0]),) + sample[1:]

        return sample


class _PerineuralNetsDMapImage(Dataset):
    """ Dataset that provides per-patch iteration of a single big image file. """

    def __init__(self, h5_path, annotations, split='left', with_targets=True, patch_size=640, stride=None,
                 random_offset=0, gt_params=None, max_cache_mem=None):
        self.h5_path = h5_path
        self.random_offset = random_offset
        self.with_targets = with_targets
        self.gt_params = gt_params if gt_params is not None else PerineuralNetsDMapDataset.DEFAULT_GT_PARAMS

        assert split in ('left', 'right', 'all'), "split must be one of ('left', 'right', 'all')"
        self.split = split

        # patch size (height and width)
        assert patch_size % 32 == 0, "Patch size must be divisible by 32"
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

        # keep only annotations of this image
        self.image_id = Path(h5_path).with_suffix('.tif').name
        self.annot = annotations.loc[self.image_id]

        # keep also annotations in the selected split (in split's coordinates)
        in_split = ((self.annot[['Y', 'X']] >= self.origin_yx) & (self.annot[['Y', 'X']] < self.limits_yx)).all(axis=1)
        self.split_annot = self.annot[in_split].copy()
        self.split_annot[['Y', 'X']] -= self.origin_yx

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
            dmap = self._build_target_dmap(patch, patch_locations)

        # pad patch (in case of patches in last col/rows)
        py, px = - patch_hw % self.patch_hw
        pad = ((0, py), (0, px))

        patch = np.pad(patch, pad)  # defaults to zero padding

        if self.with_targets:
            dmap = np.pad(dmap, pad)

            # stack in a unique RGB-like tensor, useful for applying data augmentation
            input_and_target = np.stack((patch, dmap), axis=-1)
            datum = input_and_target
        else:
            datum = np.expand_dims(patch, axis=-1)  # add channels dimension

        return datum, patch_hw, local_start_yx, self.region_hw, self.image_id

    def _build_target_dmap(self, patch, locations):
        """ This builds the density map, putting a gaussian over each dots localizing a perineural net
        """

        kernel_size = self.gt_params['k_size']
        sigma = self.gt_params['sigma']

        shape = patch.shape
        dmap = np.zeros(shape, dtype=np.float32)

        if len(locations) == 0:  # empty patch
            return dmap

        for i, center in enumerate(locations):
            H = np.multiply(cv2.getGaussianKernel(kernel_size, sigma),
                            (cv2.getGaussianKernel(kernel_size, sigma)).T)

            x = min(shape[1], max(1, abs(int(floor(center[1])))))
            y = min(shape[0], max(1, abs(int(floor(center[0])))))

            if x > shape[1] or y > shape[0]:
                continue

            x1 = x - int(floor(kernel_size / 2))
            y1 = y - int(floor(kernel_size / 2))
            x2 = x + int(floor(kernel_size / 2))
            y2 = y + int(floor(kernel_size / 2))
            dfx1 = 0
            dfy1 = 0
            dfx2 = 0
            dfy2 = 0
            change_H = False

            if x1 < 0:
                dfx1 = abs(x1)
                x1 = 0
                change_H = True
            if y1 < 0:
                dfy1 = abs(y1)
                y1 = 0
                change_H = True
            if x2 > shape[1] - 1:
                dfx2 = x2 - (shape[1] - 1)
                x2 = shape[1] - 1
                change_H = True
            if y2 > shape[0] - 1:
                dfy2 = y2 - (shape[0] - 1)
                y2 = shape[0] - 1
                change_H = True

            x1h = 1 + dfx1
            y1h = 1 + dfy1
            x2h = kernel_size - dfx2
            y2h = kernel_size - dfy2
            if change_H is True:
                H = np.multiply(cv2.getGaussianKernel(int(y2h - y1h + 1), sigma),
                                (cv2.getGaussianKernel(int(x2h - x1h + 1), sigma)).T)  # H.shape == (r, c)

            dmap[y1: y2 + 1, x1: x2 + 1] = dmap[y1: y2 + 1, x1: x2 + 1] + H

        return dmap


# Debug code
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
    from utils.misc import normalize
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image

    data_root = '/home/luca/luca-cnr/mnt/datino/perineural_nets'
    device = "cpu"
    output_folder = "output/gt/dmaps_patches"

    # for split in ('train', 'validation', 'train-specular', 'validation-specular', 'all'):
    #     dataset = PerineuralNetsDMapDataset(data_root, split=split)
    #     print(split, len(dataset))

    dataset = PerineuralNetsDMapDataset(data_root, split='train-specular', patch_size=640, overlap=120, with_targets=True, transforms=Compose([ToTensor(), RandomHorizontalFlip()]), max_cache_mem=8*1024**3)  # bytes = 8 GiB
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    progress = tqdm(dataloader, desc='TRAIN', leave=False)
    for i, sample in enumerate(progress):
        input_and_target, patch_hw, start_yx, image_hw, image_id = sample
        input_and_target = input_and_target.to(device)
        # split channels to get input, target, and loss weights
        images, targets = input_and_target.split(1, dim=1)

        for img, dmap, img_id in zip(images, targets, image_id):
            img_name = img_id.rsplit(".", 1)[0] + "_{}.png".format(i)
            dmap_name = img_id.rsplit(".", 1)[0] + "_{}_dmap.png".format(i)

            pil_image = to_pil_image(img.cpu())
            pil_image.save(os.path.join(output_folder, img_name))

            pil_dmap = Image.fromarray(normalize(dmap.squeeze(dim=0).cpu().numpy()).astype('uint8'))
            pil_dmap.save(os.path.join(output_folder, dmap_name))





