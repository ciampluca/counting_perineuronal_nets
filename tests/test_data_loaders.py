import logging
import shutil
import unittest

import numpy as np

from datasets.CellsDataset import CellsDataset


logging.basicConfig(level=logging.INFO)


class TestDataLoader(unittest.TestCase):
    """ Tests for dataloaders. """

    def _test_dataset_basic(self, root, expected_side, as_gray=False):
        common = dict(root=root, split='all', as_gray=as_gray, target_cache=False)
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

        # countmap
        with self.subTest(target='countmap'):
            dataset = CellsDataset(target='countmap', **common)
            datum, *_ = dataset[0]
            # self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 1))  # <-- TODO
            self.assertEqual(len(datum.shape), 3)
            self.assertEqual(datum.shape[2], num_channels + 1)

    def test_adi(self):
        self._test_dataset_basic(root='data/adipocyte-cells', expected_side=150)

    def test_bcd(self):
        self._test_dataset_basic(root='data/bcd-cells/validation', expected_side=640)

    def test_vgg(self):
        self._test_dataset_basic(root='data/vgg-cells', expected_side=256, as_gray=True)

    def test_mbm(self):
        self._test_dataset_basic(root='data/mbm-cells', expected_side=600)

    def test_nuclei(self):
        self._test_dataset_basic(root='data/nuclei-cells', expected_side=500)


class TestCache(unittest.TestCase):
    
    def tearDown(self):
        shutil.rmtree('./test_cache', ignore_errors=True)
    
    def _test_dataset_cache(self, root):
        common = dict(root=root, split='all', as_gray=False)

        for target in ('segmentation', 'detection', 'density', 'countmap'):
            with self.subTest(target=target):
                dataset = CellsDataset(target=target, target_cache='./test_cache', **common)
                x = dataset[0]
                
                dataset = CellsDataset(target=target, target_cache='./test_cache', **common)
                cached_x = dataset[0]

                for xi, xci in zip(x, cached_x):
                    np.testing.assert_equal(xi, xci)

    def test_cache(self):
        datasets = [
            ('adi', 'data/adipocyte-cells'),
            ('bcd', 'data/bcd-cells/validation'),
            ('vgg', 'data/vgg-cells'),
            ('mbm', 'data/mbm-cells'),
            ('nuclei', 'data/nuclei-cells'),
        ]

        for dataset, root in datasets:
            with self.subTest(dataset=dataset):
                self._test_dataset_cache(root=root)