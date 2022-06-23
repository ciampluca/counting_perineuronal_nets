from abc import ABC, abstractmethod


class BaseTargetBuilder(ABC):
    """ Base class for Target Builders. """

    @abstractmethod
    def build(self, shape, locations, n_classes=None):
        """ Builds the desired target given object locations, image shape,
            and number of classes.

        Args:
            shape (tuple): size as a (height, width) tuple.
            locations (pd.DataFrame): dataframe with 'X', 'Y', and 'class' columns;
                                      'class' column can be omitted for the
                                      single-class case.
            n_classes (int, optional): number of total classes; if None, it is
                                       estimated as number of unique classes
                                       encountered. Defaults to None.
        """
        pass

    @abstractmethod
    def pack(self, image, target, pad=None):
        """ Packs the input image and the target in a convenient format, e.g.,
        for easier data augmentation.

        Args:
            image (np.ndarray): (H,W,C)-shaped image
            target (object): the generated target
            pad (int, optional): padding to be inserted. Defaults to None.
        """
        pass
    