import functools
import itertools
import unittest

import torch

from models import *


class TestModels(unittest.TestCase):
    """ Tests for dataloaders. """

    @torch.no_grad()
    def _test_map_model(
        self,
        model_class,
        in_side=(256, 500, 600),
        in_channels=(1,3),
        out_channels=(1,4)
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for input_channels, output_channels in itertools.product(in_channels, out_channels):
            model = model_class(
                in_channels=input_channels,
                out_channels=output_channels,
                skip_weights_loading=True
            ).to(device)

            for input_side in in_side:
                with self.subTest(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    input_side=input_side
                ):
                    in_shape = (2, input_channels, input_side, input_side)
                    out_shape = (2, output_channels, input_side, input_side)
                    x = torch.rand(*in_shape, dtype=torch.float32, device=device)
                    y = model(x)
                    self.assertEqual(y.shape, out_shape)
            
            del model

    def test_unet(self):
        self._test_map_model(model_class=UNet)
    
    def test_csrnet(self):
        self._test_map_model(model_class=CSRNet)
    
    def test_countception(self):
        self._test_map_model(model_class=CountCeption)
    
    def test_fcrn(self):
        for version in ('A', 'B'):
            with self.subTest(version=version):
                model_class = functools.partial(FCRN, version=version)
                self._test_map_model(model_class=model_class)