import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.heatmaps import HeatmapsOnImage

import torch
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):

    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5)
        ])

    def __call__(self, image, dmap=None):
        if isinstance(image, torch.Tensor):
            image = np.asarray(image)
        if isinstance(dmap, torch.Tensor) and dmap is not None:
            dmap = np.asarray(dmap)

        if dmap is not None:
            dmap = HeatmapsOnImage(dmap, shape=image.shape, min_value=0.0, max_value=1.0)
            image, dmap = self.seq(image=image, heatmaps=dmap)
            dmap = np.asarray(dmap.get_arr())
        else:
            image = self.seq(image=image)

        return image, dmap


class PadToResizeFactor(object):

    def __init__(self, resize_factor=32):
        self.resize_factor = resize_factor

    def __call__(self, image, dmap=None):
        if isinstance(image, torch.Tensor):
            image = np.asarray(image)
        if isinstance(dmap, torch.Tensor) and dmap is not None:
            dmap = np.asarray(dmap)

        img_h, img_w = image.shape[:2]
        if img_w % self.resize_factor != 0:
            img_w = img_w + (self.resize_factor - (img_w % self.resize_factor))
        if img_h % self.resize_factor != 0:
            img_h = img_h + (self.resize_factor - (img_h % self.resize_factor))

        transf = iaa.PadToFixedSize(width=img_w, height=img_h, position='center')

        if dmap is not None:
            dmap = HeatmapsOnImage(dmap, shape=image.shape, min_value=0.0, max_value=1.0)
            image, dmap = transf(image=image, heatmaps=dmap)
            dmap = np.asarray(dmap.get_arr())
        else:
            image = transf(image=image)

        return image, dmap


class RandomCrop(object):

    def __init__(self, width, height):
        self.seq = iaa.Sequential([
            iaa.CropToFixedSize(width=width, height=height)
        ])

    def __call__(self, image, dmap=None):
        if isinstance(image, torch.Tensor):
            image = np.asarray(image)
        if isinstance(dmap, torch.Tensor) and dmap is not None:
            dmap = np.asarray(dmap)

        if dmap is not None:
            dmap = HeatmapsOnImage(dmap, shape=image.shape, min_value=0.0, max_value=1.0)
            image, dmap = self.seq(image=image, heatmaps=dmap)
            dmap = np.asarray(dmap.get_arr())
        else:
            image = self.seq(image=image)

        return image, dmap


class ToTensor(object):

    def __call__(self, image, dmap=None):
        image = F.to_tensor(image)

        if dmap is not None:
            dmap = torch.as_tensor(dmap, dtype=torch.float32)

        return image, dmap


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target
