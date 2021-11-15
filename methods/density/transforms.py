import numpy as np


class Resize(object):
    """ Resize the image and the associated density map """

    def __init__(self, source_size, destination_size):
        """ Constructor. Sizes are in (width, height) format.

        Args:
            source_size (tuple, ndarray): original size of annotated image
            destination_size (tuple, ndarray): new size of annotated image
        """
        self.source_size = source_size
        self.destination_size = destination_size
        self.f = np.array(destination_size) / np.array(source_size)
