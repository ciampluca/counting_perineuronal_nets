import numpy as np


class DetectionTargetBuilder:
    """ This builds the detection bounding boxes from points. """

    def __init__(self, side=60, **kwargs):
        """ Constructor.
        Args:
            side (int, optional): Side (in px) of the bounding box localizing a cell. Defaults to 60.
        """
        self.side = side

    def build(self, image_hw, points_yx):
        """ Builds the detection target. """
        half_side = self.side / 2
        hwhw = np.tile(image_hw, 2)

        tl = points_yx - half_side
        br = points_yx + half_side

        bbs = np.hstack((tl, br))
        bbs = np.clip(bbs, 0, hwhw)

        return bbs
    
    def pack(self, image, target, pad=None):
        # put in a unique tuple the patch and the target
        return image, target
