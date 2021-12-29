from _collections import OrderedDict
import math

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms.functional import resize


class CSRNet(nn.Module):
    """
    Congested Scene Recognition Network (CSRNet)
    Ref. Y. Li et al. 'CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes'
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        skip_weights_loading=False
    ):
        super(CSRNet, self).__init__()

        self.in_channels = in_channels
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat, in_channels=3)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

        if not skip_weights_loading:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = OrderedDict()
            # 10 convolutions *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):

        if self.in_channels == 1:
            x = x.expand(-1, 3, -1, -1)  # gray to RGB

        h, w = x.shape[-2:]
        need_resize = (h % 8) or (w % 8)

        if need_resize:
            newH = math.ceil(h / 8) * 8
            newW = math.ceil(w / 8) * 8
            x = resize(x, (newH, newW))

        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # Since upsample_bilinear2d cuda does not have a deterministic implementation, a workaround is to move the
        # tensor in the cpu (to guarantee reproducibility)
        device = "cpu"
        if x.is_cuda:
            device = x.get_device()
            torch.cuda.synchronize()
            x = x.cpu()
        x = nn.functional.interpolate(x, scale_factor=8, mode="bilinear")
        x = x.to(device)

        if need_resize:
            x = resize(x, (h, w))

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
    in_channels = 3
    batch_size = 2
    shape = (256, 256)
    torch.hub.set_dir('../model_zoo/')
    
    model = CSRNet(skip_weights_loading=True, in_channels=in_channels)
    input_img = torch.rand(batch_size, in_channels, shape[0], shape[1])
    density = model(input_img)

    print(density.shape)
    # print(density)
