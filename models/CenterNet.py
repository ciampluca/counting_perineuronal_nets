import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class CenterNet(nn.Module):
    """
    CenterNet
    Ref. Xingyi Zhou et al. 'Objects as Points'
    Code based on: https://www.kaggle.com/code/kyoshioka47/centernet-starterkit-pytorch/notebook
    """
    
    def __init__(
        self, 
        in_channels=3,
        out_channels=1,
        backbone="resnet50",
        cache_folder='./model_zoo',
        skip_weights_loading=False,
        progress=True,
        ):
        
            assert backbone in ("resnet50", "resnet101"), f"Backbone not supported: {backbone}"
            
            super(CenterNet, self).__init__()
            
            # defining the backbone
            os.environ['TORCH_HOME'] = cache_folder
            backbone_module = resnet.__dict__[backbone](pretrained=not skip_weights_loading, progress=progress)
            self.backbone = nn.Sequential(*list(backbone_module.children())[:-2])
            
            self.up1 = Up(backbone_module.inplanes, 512)
            self.up2 = Up(512, 256)
            self.up3 = Up(256, 256)
            
            # output classification
            self.outc = nn.Conv2d(256, out_channels, 1)
            # output residue
            self.outr = nn.Conv2d(256, 2, 1)
        
    def forward(self, x):
        x = self.backbone(x)
        
        # Add positional info        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        outc = self.outc(x)
        outr = self.outr(x)
        
        return outc, outr
    

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
        
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        else:
            x = x1
            
        x = self.conv(x)
        return x
