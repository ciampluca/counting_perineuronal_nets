import numpy as np

from skimage.draw import disk

from methods.base_target_builder import BaseTargetBuilder


class DetectionTargetBuilder(BaseTargetBuilder):
    """ This builds the detection bounding boxes from points. """

    def __init__(self, side=60, mask=False, **kwargs):
        """ Constructor.
        Args:
            side (int, optional): Side (in px) of the bounding box localizing a cell. Defaults to 60.
            mask (bool, optional): if True, it also build binary masks for each instance with circles centered in the bbs and radius of side/2
        """
        self.side = side
        self.mask = mask

    def build(self, shape, locations, n_classes=None):
        """ Builds the detection target. """
        points_yx = locations[['Y', 'X']].values
        labels = locations['class'].values

        half_side = self.side / 2
        hwhw = np.tile(shape, 2)

        tl = points_yx - half_side
        br = points_yx + half_side

        bbs = np.hstack((tl, br))
        bbs = np.clip(bbs, 0, hwhw)
        
        if self.mask:
            mask_shape = (*shape, len(points_yx))
            mask_segmentation = np.zeros(mask_shape, dtype=np.int64)
            radius = self.side / 2
            for i, center in enumerate(points_yx):
                rr, cc = disk(center, radius, shape=shape)
                mask_segmentation[rr, cc, i] = 1

            return bbs, labels, mask_segmentation
        
        return bbs, labels
    
    def pack(self, image, target, pad=None):
        if self.mask and pad:
            bbs, labels, mask_segmentations = target
            mask_segmentations = np.pad(mask_segmentations, pad)
            target = bbs, labels, mask_segmentations
        
        # put in a unique tuple the patch and the target
        return (image,) + target
