import random
import numpy as np

import torchvision.transforms.functional as F


class BaseRandomFlip(object):
    """ Base class for random flipping aumgmentation. """
    def __init__(self, p, orient):
        self.p = p
        self.orient = orient  # 'h' or 'v'
    
    def __call__(self, datum):
        if random.random() < self.p:
            if isinstance(datum, tuple):
                image, boxes, labels = datum

                if self.orient == 'h':
                    image = image[:, ::-1, :]
                else:
                    image = image[::-1, :, :]
                
                # .copy() is needed to tackle 'RuntimeError: some of the strides of a given numpy array are negative.'
                # see https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
                image = image.copy()

                if boxes.size != 0:
                    image_center = np.array(image.shape[:2])[::-1] / 2
                    image_center = np.hstack((image_center, image_center))

                    min_idx = 1 if self.orient == 'h' else 0
                    max_idx = 3 if self.orient == 'h' else 2

                    boxes[:, [min_idx, max_idx]] += 2 * (image_center[[min_idx, max_idx]] - boxes[:, [min_idx, max_idx]])

                    box_size = np.abs(boxes[:, min_idx] - boxes[:, max_idx])

                    boxes[:, min_idx] -= box_size
                    boxes[:, max_idx] += box_size
                
                datum = (image, boxes, labels)

            else:
                if self.orient == 'h':
                    datum = datum[:, ::-1, :]
                else:
                    datum = datum[::-1, :, :]

                datum = datum.copy()

        return datum


class RandomHorizontalFlip(BaseRandomFlip):
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
        super().__init__(p=p, orient='h')


class RandomVerticalFlip(BaseRandomFlip):
    """Randomly vertically flips the Image with the probability *p*

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
        super().__init__(p=p, orient='v')


class ToTensor(object):

    def __call__(self, datum):
        if isinstance(datum, tuple):
            image, *targets = datum
            image = F.to_tensor(image)
            datum = (image,) + tuple(targets)
        else:
            datum = F.to_tensor(datum)
            
        return datum


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datum):
        for t in self.transforms:
            datum = t(datum)

        return datum


