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


class FCRN(nn.Module):
    """
    Fully Convolutional Regression Network A and B (FCRN-A, FCRN-B)
    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional Regression Networks'
    Code inspired (for FCRN-A) by: https://github.com/NeuroSYS-pl/objects_counting_dmap
    
    Args:
    
    """

    def __init__(
        self,
        N: int=1,
        in_channels: int=3,
        out_channels: int=1,
        version='A',
        **kwargs
    ):
        """
        Create FCRN model
        
        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN, self).__init__()
        
        assert version in ['A', 'B'], "Not implemented version (possible values are A or B)"
        self.version = version
        
        if version == 'A':
            self.conv_block1 = nn.Sequential(
                conv_block(channels=(in_channels, 32), size=(3, 3), N=N),
                nn.MaxPool2d(2)
            )
            self.conv_block2 = nn.Sequential(
                conv_block(channels=(32, 64), size=(3, 3), N=N),
                nn.MaxPool2d(2)
            )
            self.conv_block3 = nn.Sequential(
                conv_block(channels=(64, 128), size=(3, 3), N=N),
                nn.MaxPool2d(2)
            )
            
            self.conv_block4 = conv_block(channels=(128, 512), size=(3, 3), N=N)
            
            self.conv_block5 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                conv_block(channels=(512, 128), size=(3, 3), N=N)
            )
            self.conv_block6 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                conv_block(channels=(128, 64), size=(3, 3), N=N)
            )
            self.conv_block7 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                conv_block(channels=(64, 32), size=(3, 3), N=N)
            )
            
            self.conv_block8 = conv_block(channels=(32, out_channels), size=(3, 3), N=N)
            
        else:     # version 'B'
            self.conv_block1 = conv_block(channels=(in_channels, 32), size=(3, 3), N=N)
            self.conv_block2 = nn.Sequential(
                conv_block(channels=(32, 64), size=(3, 3), N=N),
                nn.MaxPool2d(2)
            )
            self.conv_block3 = conv_block(channels=(64, 128), size=(3, 3), N=N)
            self.conv_block4 = nn.Sequential(
                conv_block(channels=(128, 256), size=(3, 3), N=N),
                nn.MaxPool2d(2)
            )
            self.conv_block5 = conv_block(channels=(256, 256), size=(5, 5), N=N)
            
            self.conv_block6 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                conv_block(channels=(256, 256), size=(5, 5), N=N)
            )
            
            self.conv_block7 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                conv_block(channels=(256, out_channels), size=(5, 5), N=N)
            )
        

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        need_resize = (h % 8) or (w % 8)
        
        if need_resize:
            newH = math.ceil(h / 8) * 8
            newW = math.ceil(w / 8) * 8
            x = resize(x, (newH, newW))
            
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        if self.version == 'A':
            x = self.conv_block8(x)    
        
        if need_resize:
            x = resize(x, (h, w))
        
        return x
    
    
# Testing code
if __name__ == "__main__":
    # It works with 1 or 3 channels input images
    in_channels = 3
    num_classes = 2
    version = 'B'
    
    model = FCRN(in_channels=in_channels, out_channels=num_classes, version=version)
    input_img = torch.rand(1, in_channels, 100, 100)
    density = model(input_img)

    print(density.shape)
    #print(density)
