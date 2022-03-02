import unittest

from datasets.CellsDataset import CellsDataset


class TestDataLoader(unittest.TestCase):
    """ Tests for dataloaders. """

    def _test_dataset_basic(self, root, expected_side, as_gray=False):
        common = dict(root=root, split='all', as_gray=as_gray)
        num_channels = 1 if as_gray else 3

        # no target
        with self.subTest(target='none'):
            dataset = CellsDataset(**common)
            datum, *_ = dataset[0]
            self.assertEqual(datum.shape, (expected_side, expected_side, num_channels))

        # segmentation
        with self.subTest(target='segmentation'):
            dataset = CellsDataset(target='segmentation', **common)
            datum, *_ = dataset[0]
            self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 2))

        # detection
        with self.subTest(target='detection'):
            dataset = CellsDataset(target='detection', **common)
            datum, *_ = dataset[0]
            x, y = datum
            self.assertEqual(x.shape, (expected_side, expected_side, num_channels))
            self.assertTrue(y.ndim == 2)
            self.assertTrue(y.shape[1] == 4)

        # density
        with self.subTest(target='density'):
            dataset = CellsDataset(target='density', **common)
            datum, *_ = dataset[0]
            self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 1))

    def test_vgg(self):
        self._test_dataset_basic(root='data/vgg-cells', expected_side=256, as_gray=True)

    def test_mbm(self):
        self._test_dataset_basic(root='data/mbm-cells', expected_side=600, as_gray=False)

    def test_nuclei(self):
        self._test_dataset_basic(root='data/nuclei-cells', expected_side=500, as_gray=False)
    