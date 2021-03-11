from _collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = OrderedDict()
            # 10 convolutions *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x, bmask=None):
        if bmask is not None:
            x = x * bmask   # zero input values outside the active region
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # x = nn.functional.interpolate(x, scale_factor=8, mode="nearest")
        x = nn.functional.interpolate(x, scale_factor=8, mode="bilinear")

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)


# Testing code
if __name__ == "__main__":
    torch.hub.set_dir('../model_zoo/')
    model = CSRNet()
    input_img = torch.rand(1, 3, 256, 256)
    bmask = torch.ones(1, 1, 256, 256)
    density = model(input_img, bmask=bmask)

    print(density.shape)
    print(density)
