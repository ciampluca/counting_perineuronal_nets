import albumentations as A

import torch
import torchvision.transforms.functional as F


class PadToResizeFactor(object):

    def __init__(self, resize_factor=32):
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=resize_factor,
                          pad_width_divisor=resize_factor, value=0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        if target is not None and target['boxes'].nelement() != 0:
            bboxes = target['boxes'][:, :4].tolist()
            class_labels = target['labels'].tolist()
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            target['boxes'] = torch.tensor(bboxes)
        else:
            image = self.transform(image=image)

        return image, target


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.transform = A.Compose([
            A.HorizontalFlip(p=prob),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        if target is not None and target['boxes'].nelement() != 0:
            bboxes = target['boxes'][:, :4].tolist()
            class_labels = target['labels'].tolist()
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            target['boxes'] = torch.tensor(bboxes)
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']

        return image, target


class RandomCrop(object):

    def __init__(self, width, height, min_visibility=0.0):
        self.transform = A.Compose([
            A.RandomCrop(width=width, height=height),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_visibility))

    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        if target is not None and target['boxes'].nelement() != 0:
            bboxes = target['boxes'][:, :4].tolist()
            class_labels = target['labels'].tolist()
            n_bboxes = len(bboxes)
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            target['boxes'] = torch.tensor(bboxes)
            target['labels'] = torch.tensor(labels)
            if len(bboxes) != n_bboxes:
                target['iscrowd'] = torch.zeros_like(target['labels'])
                target['area'] = torch.as_tensor([(bb[2] - bb[0]) * (bb[3] - bb[1])
                                                  for bb in bboxes], dtype=torch.float32)
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']

        return image, target


class CropToFixedSize(object):

    def __call__(self, image, x_min, y_min, x_max, y_max, target=None, min_visibility=0):
        image_to_tensor_flag = 0
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                device = image.get_device()
            else:
                device = torch.device("cpu")
            image = image.permute(1, 2, 0).cpu().numpy()
            image_to_tensor_flag = 1

        transform = A.Compose([
            A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_visibility))

        if target is not None and target['boxes'].nelement() != 0:
            bboxes = target['boxes'][:, :4].tolist()
            class_labels = target['labels'].tolist()
            n_bboxes = len(bboxes)
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            target['boxes'] = torch.tensor(bboxes)
            target['labels'] = torch.tensor(labels)
            if len(bboxes) != n_bboxes:
                target['iscrowd'] = torch.zeros_like(target['labels'])
                target['area'] = torch.as_tensor([(bb[2] - bb[0]) * (bb[3] - bb[1])
                                                  for bb in bboxes], dtype=torch.float32)
        else:
            transformed = transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']

        if image_to_tensor_flag:
            image = F.to_tensor(image).to(device)
            target = {k: v.to(device) for k, v in target.items()}

        return image, target


class ToTensor(object):

    def __call__(self, image, target=None):
        image = F.to_tensor(image)

        return image, target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target
