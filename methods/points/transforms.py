import numpy as np

class AnnotationResize(object):
    """ Adjust points annotations to align to a resized version of the image they belong. """

    def __init__(self, source_size, destination_size):
        """ Constructor. Sizes are in (width, height) format.

        Args:
            source_size (tuple, ndarray): original size of annotated image
            destination_size (tuple, ndarray): new size of annotated image
        """
        self.source_size = source_size
        self.destination_size = destination_size
        self.f = np.array(destination_size) / np.array(source_size)

    def __call__(self, annot):
        """ Transforms the X and Y columns of a pandas.DataFrame.

        Args:
            annot (pandas.DataFrame): dataframe to modify.
        """
        annot = annot.copy()
        annot[['X', 'Y']] = (annot[['X', 'Y']] * self.f).round().astype(int)
        return annot

