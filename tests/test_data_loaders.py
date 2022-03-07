import logging
from pathlib import Path
import shutil
import unittest

import numpy as np
from skimage.io import imsave

from datasets.CellsDataset import CellsDataset
from methods.segmentation.utils import segmentation_map_to_points


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
            target_params = {
                'radius': expected_side // 25,
                'radius_ign': expected_side // 24
            }
            dataset = CellsDataset(target='segmentation', target_params=target_params, **common)
            datum, *_ = dataset[0]
            self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 2))

            n_cells = len(dataset.datasets[0].split_annot)
            segmentation_map = datum[:, :, num_channels]
            n_cells_in_target = len(segmentation_map_to_points(segmentation_map))

            self.assertAlmostEqual(n_cells, n_cells_in_target)
            if n_cells != n_cells_in_target:
                path = Path('./test_fails/')
                path.mkdir(exist_ok=True)
                path = path / (self.id() + '_segm.png')
                imsave(path, segmentation_map)

                self.fail(f'n_cells != n_cells_in_target: {n_cells} vs {n_cells_in_target}')

        # detection
        with self.subTest(target='detection'):
            dataset = CellsDataset(target='detection', **common)
            (x, y), *_ = dataset[0]

            self.assertEqual(x.shape, (expected_side, expected_side, num_channels))
            self.assertTrue(y.ndim == 2)
            self.assertTrue(y.shape[1] == 4)

            n_cells = len(dataset.datasets[0].split_annot)
            n_cells_in_target = y.shape[0]
            self.assertEqual(n_cells, n_cells_in_target)

        # density
        with self.subTest(target='density'):
            dataset = CellsDataset(target='density', **common)
            datum, *_ = dataset[0]
            self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 1))

            n_cells = len(dataset.datasets[0].split_annot)
            n_cells_in_target = datum[:, :, num_channels:].sum()
            self.assertAlmostEqual(n_cells_in_target, n_cells, places=3)

        # countmap
        with self.subTest(target='countmap'):
            target_params = {'target_patch_size': 32}
            dataset = CellsDataset(target='countmap', target_params=target_params, **common)
            datum, *_ = dataset[0]
            # self.assertEqual(datum.shape, (expected_side, expected_side, num_channels + 1))  # <-- TODO
            self.assertEqual(len(datum.shape), 3)
            self.assertEqual(datum.shape[2], num_channels + 1)

            n_cells = len(dataset.datasets[0].split_annot)
            countmap = datum[:, :, num_channels]
            n_cells_in_target = (countmap / 32 ** 2.0).sum()

            self.assertAlmostEqual(n_cells, n_cells_in_target, delta=1)
            if n_cells != n_cells_in_target:
                path = Path('./test_fails/')
                path.mkdir(exist_ok=True)
                path = path / (self.id() + '_countmap.png')
                imsave(path, countmap)

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