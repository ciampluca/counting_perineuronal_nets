import albumentations as A

import torch
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.transform = A.Compose([
            A.HorizontalFlip(p=prob),
        ], additional_targets={'dmap': 'image'})

    def __call__(self, image, dmap=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        if dmap is not None:
            transformed = self.transform(image=image, dmap=dmap)
            image, dmap = transformed['image'], transformed['dmap']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, dmap


class PadToResizeFactor(object):

    def __init__(self, resize_factor=32):
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=resize_factor,
                          pad_width_divisor=resize_factor, value=0),
        ], additional_targets={'dmap': 'image'})

    def __call__(self, image, dmap=None):
        image_to_tensor_flag = 0
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                device = image.get_device()
            else:
                device = torch.device("cpu")
            image = image.permute(1, 2, 0).cpu().numpy()
            image_to_tensor_flag = 1

        dmap_to_tensor_flag = 0
        if dmap is not None and isinstance(dmap, torch.Tensor):
            if dmap.is_cuda:
                device = dmap.get_device()
            else:
                device = torch.device("cpu")
            dmap = dmap.permute(1, 2, 0).cpu().numpy()
            dmap_to_tensor_flag = 1

        if dmap is not None:
            transformed = self.transform(image=image, dmap=dmap)
            image, dmap = transformed['image'], transformed['dmap']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']

        if image_to_tensor_flag:
            image = F.to_tensor(image).to(device)
        if dmap_to_tensor_flag:
            dmap = torch.as_tensor(dmap, dtype=torch.float32).permute(2, 0, 1).to(device)

        return image, dmap


class PadToSize(object):

    def __call__(self, image, min_height, min_width, dmap=None):
        image_to_tensor_flag = 0
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                device = image.get_device()
            else:
                device = torch.device("cpu")
            image = image.permute(1, 2, 0).cpu().numpy()
            image_to_tensor_flag = 1

        dmap_to_tensor_flag = 0
        if dmap is not None and isinstance(dmap, torch.Tensor):
            if dmap.is_cuda:
                device = dmap.get_device()
            else:
                device = torch.device("cpu")
            dmap = dmap.permute(1, 2, 0).cpu().numpy()
            dmap_to_tensor_flag = 1

        transform = A.Compose([
            A.PadIfNeeded(min_height=min_height, min_width=min_width, pad_height_divisor=None, pad_width_divisor=None, value=0),
        ], additional_targets={'dmap': 'image'})

        if dmap is not None:
            transformed = transform(image=image, dmap=dmap)
            image, dmap = transformed['image'], transformed['dmap']
        else:
            transformed = transform(image=image)
            image = transformed['image']

        if image_to_tensor_flag:
            image = F.to_tensor(image).to(device)
        if dmap_to_tensor_flag:
            dmap = torch.as_tensor(dmap, dtype=torch.float32).permute(2, 0, 1).to(device)

        return image, dmap


class RandomCrop(object):

    def __init__(self, width, height):
        self.transform = A.Compose([
            A.RandomCrop(width=width, height=height),
        ], additional_targets={'dmap': 'image'})
        self.crop_width, self.crop_height = width, height

    def __call__(self, image, dmap=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        if image.shape[0] == self.crop_height and image.shape[1] == self.crop_width:
            return image, dmap

        if dmap is not None:
            transformed = self.transform(image=image, dmap=dmap)
            image, dmap = transformed['image'], transformed['dmap']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, dmap


class CropToFixedSize(object):

    def __call__(self, image, x_min, y_min, x_max, y_max, dmap=None):
        image_to_tensor_flag = 0
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                device = image.get_device()
            else:
                device = torch.device("cpu")
            image = image.permute(1, 2, 0).cpu().numpy()
            image_to_tensor_flag = 1

        dmap_to_tensor_flag = 0
        if dmap is not None and isinstance(dmap, torch.Tensor):
            if dmap.is_cuda:
                device = dmap.get_device()
            else:
                device = torch.device("cpu")
            dmap = dmap.permute(1, 2, 0).cpu().numpy()
            dmap_to_tensor_flag = 1

        transform = A.Compose([
            A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        ], additional_targets={'dmap': 'image'})

        if dmap is not None:
            transformed = transform(image=image, dmap=dmap)
            image, dmap = transformed['image'], transformed['dmap']
        else:
            transformed = transform(image=image)
            image = transformed['image']

        if image_to_tensor_flag:
            image = F.to_tensor(image).to(device)
        if dmap_to_tensor_flag:
            dmap = torch.as_tensor(dmap, dtype=torch.float32).permute(2, 0, 1).to(device)

        return image, dmap


class ToTensor(object):

    def __call__(self, image, dmap=None):
        image = F.to_tensor(image)

        if dmap is not None:
            dmap = torch.as_tensor(dmap, dtype=torch.float32).unsqueeze(dim=0)

        return image, dmap


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target
