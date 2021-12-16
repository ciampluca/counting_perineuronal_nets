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


class C_FCRN(nn.Module):
    """
    Concatenated FCRN (C-FCRN)
    Concatenated FCRN with auxiliary CNNs (C-FCRN-Aux)
    Ref. S.He et al. 'Deeply-supervised density regression for automatic cell counting in microscopy images'
    
    Args:
    
    """

    def __init__(self, N: int=1, input_filters: int=3, n_classes: int=1, with_aux=True, **kwargs):
        """
        Create C-FCRN or C-FCRN-Aux model.
        
        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(C_FCRN, self).__init__()
        
        self.with_aux = with_aux
        
        # downsampling
        self.conv_block1 = nn.Sequential(
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
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

        # "convolutional fully connected"
        self.conv_block4 = conv_block(channels=(128, 512), size=(3, 3), N=N)
        
        # upsampling
        self.conv_block5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512+128, 128), size=(3, 3), N=N)
        )
        self.conv_block6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(channels=(128+64, 64), size=(3, 3), N=N)
        )
        self.conv_block7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(channels=(64+32, 32), size=(3, 3), N=N)
        )
        
        # Final layer
        self.conv_block8 = conv_block(channels=(32, n_classes), size=(1, 1), N=N)
        
        self.aux_conv_block1 = nn.Sequential(
            conv_block(channels=(512, 32), size=(3, 3), N=N),
            conv_block(channels=(32, 1), size=(1, 1), N=N)
        )
        
        self.aux_conv_block2 = nn.Sequential(
            conv_block(channels=(128, 32), size=(3, 3), N=N),
            conv_block(channels=(32, 1), size=(1, 1), N=N)
        )
                
        self.aux_conv_block3 = nn.Sequential(
            conv_block(channels=(64, 32), size=(3, 3), N=N),
            conv_block(channels=(32, 1), size=(1, 1), N=N)
        )
        

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        need_resize = (h % 8) or (w % 8)
        
        if need_resize:
            newH = math.ceil(h / 8) * 8
            newW = math.ceil(w / 8) * 8
            x = resize(x, (newH, newW))
            
        x = self.conv_block1(x)
        residual_block1 = x
        x = self.conv_block2(x)
        residual_block2 = x
        x = self.conv_block3(x)
        residual_block3 = x
        x = self.conv_block4(x)
        if self.with_aux:
            aux_out1 = self.aux_conv_block1(x)
        x = torch.cat((x, residual_block3), dim=1)
        x = self.conv_block5(x)
        if self.with_aux:
            aux_out2 = self.aux_conv_block2(x)
        x = torch.cat((x, residual_block2), dim=1)
        x = self.conv_block6(x)
        if self.with_aux:
            aux_out3 = self.aux_conv_block3(x)
        x = torch.cat((x, residual_block1), dim=1)
        x = self.conv_block7(x) 
        final_out = self.conv_block8(x)     
        
        if need_resize:
            final_out = resize(final_out, (h, w))
        
        if self.with_aux:
            return aux_out1, aux_out2, aux_out3, final_out
        else:
            return final_out
    
    
# Testing code
if __name__ == "__main__":
    # It works with 1 or 3 channels input images
    in_channels = 3
    num_classes = 1
    with_aux = True
    # TODO check with aux the generated dmaps when need resize 
    model = C_FCRN(input_filters=in_channels, n_classes=num_classes, with_aux=with_aux)
    input_img = torch.rand(1, in_channels, 128, 128)
    output = model(input_img)

    for out in output:
        print(out.shape)
