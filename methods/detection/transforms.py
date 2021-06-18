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

    numpy.ndarray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `y1,x1,y2,x2` of the box
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, datum):
        if isinstance(datum, tuple):
            image, boxes = datum
        else:
            image = datum
            boxes = None

        if random.random() < self.p:
            # .copy() is needed to tackle 'RuntimeError: some of the strides of a given numpy array are negative.'
            # see https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
            image = image[:, ::-1, :].copy()

            if boxes is not None:
                if boxes.size != 0:
                    image_center = np.array(image.shape[:2])[::-1] / 2
                    image_center = np.hstack((image_center, image_center))
                    boxes[:, [1, 3]] += 2 * (image_center[[1, 3]] - boxes[:, [1, 3]])

                    box_w = abs(boxes[:, 1] - boxes[:, 3])

                    boxes[:, 1] -= box_w
                    boxes[:, 3] += box_w

        return (image, boxes) if boxes is not None else image


class ToTensor(object):

    def __call__(self, datum):
        if isinstance(datum, tuple):
            image, boxes = datum
        else:
            image = datum
            boxes = None

        image = F.to_tensor(image)

        return (image, boxes) if boxes is not None else image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datum):
        for t in self.transforms:
            datum = t(datum)

        return datum


