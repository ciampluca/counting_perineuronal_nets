import math 

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize


class UNet(nn.Module):
    """
    UNet
    Ref. O. Ronneberger et al. 'U-Net: Convolutional Networks for Biomedical Image Segmentation'
    Adapted from https://discuss.pytorch.org/t/unet-implementation/426
    
    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output channels
        depth (int): depth of the network
        wf (int): number of filters in the first layer is 2**wf
        padding (bool): if True, apply padding such that the input shape is the same as the output. This may introduce artifacts
        batch_norm (bool): Use BatchNorm after layers with an activation function
        up_mode (str): one of 'upconv' or 'upsample'. 'upconv' will use transposed convolutions for learned upsampling.
                       'upsample' will use bilinear upsampling.
    """
    
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv', last_bias=False, skip_weights_loading=True):
        super(UNet, self).__init__()
        
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=last_bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        need_resize = (h % 32) or (w % 32)

        if need_resize:
            newH = math.ceil(h / 32) * 32
            newW = math.ceil(w / 32) * 32
            x = resize(x, (newH, newW))

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)

        if need_resize:
            output = resize(output, (h, w))
        
        return output


class UNetConvBlock(nn.Module):
    
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        
        return out


class UNetUpBlock(nn.Module):
    
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


# Testing code
if __name__ == "__main__":
    # It works with 1 or 3 channels input images
    in_channels = 3
    num_classes = 1
    
    model = UNet(padding=True, batch_norm=True, in_channels=in_channels, n_classes=num_classes)
    input_img = torch.rand(2, in_channels, 640, 640)
    density = model(input_img)

    print(density.shape)
    print(density)

