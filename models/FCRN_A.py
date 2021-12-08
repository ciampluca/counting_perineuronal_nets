from typing import Tuple
import math

import torch
from torch import nn
from torchvision.transforms.functional import resize


def conv_block(channels: Tuple[int, int], size: Tuple[int, int], stride: Tuple[int, int]=(1, 1), N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.
    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers
    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class FCRN_A(nn.Module):
    """
    Fully Convolutional Regression Network A (FCRN-A)
    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional Regression Networks'
    """

    def __init__(self, N: int=1, input_filters: int=3, **kwargs):
        """
        Create FCRN-A model with:
            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1
        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A, self).__init__()
        
        self.model = nn.Sequential(
            # downsampling
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=N),

            # upsampling
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=N),
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        need_resize = (h % 8) or (w % 8)
        
        if need_resize:
            newH = math.ceil(h / 8) * 8
            newW = math.ceil(w / 8) * 8
            x = resize(x, (newH, newW))
            
        x = self.model(x)
        
        if need_resize:
            x = resize(x, (h, w))
        
        return x
    
    
# Testing code
if __name__ == "__main__":
    torch.hub.set_dir('../model_zoo/')
    num_channels = 1
    model = FCRN_A(input_filters=num_channels)
    input_img = torch.rand(1, num_channels, 600, 600)
    density = model(input_img)

    print(density.shape)
    #print(density)
