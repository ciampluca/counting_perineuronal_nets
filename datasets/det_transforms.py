import random
import numpy as np

import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `y1,x1,y2,x2` of the box
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, datum):
        if isinstance(datum, tuple):
            img, bboxes = datum
        else:
            img = datum
            bboxes = None

        if random.random() < self.p:
            img = img[:, ::-1, :] - np.zeros_like(img)      # np.zeros_like is added to tackle 'RuntimeError: some of the strides of a given numpy array are negative.' see https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663

            if bboxes is not None:
                if bboxes.size != 0:
                    img_center = np.array(img.shape[:2])[::-1] / 2
                    img_center = np.hstack((img_center, img_center))
                    bboxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - bboxes[:, [1, 3]])

                    box_w = abs(bboxes[:, 1] - bboxes[:, 3])

                    bboxes[:, 1] -= box_w
                    bboxes[:, 3] += box_w

        return (img, bboxes) if bboxes is not None else img


class ToTensor(object):

    def __call__(self, datum):
        if isinstance(datum, tuple):
            img, bboxes = datum
        else:
            img = datum
            bboxes = None

        img = F.to_tensor(img)

        return (img, bboxes) if bboxes is not None else img


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datum):
        for t in self.transforms:
            datum = t(datum)

        return datum