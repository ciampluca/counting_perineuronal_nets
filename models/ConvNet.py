from torch import nn

class ConvNet(nn.Sequential):
    def __init__(
            self,
            in_channels=1,
            num_classes=2
    ):
        common = dict(kernel_size=3, padding=1)
        norm = lambda d: nn.GroupNorm(min(32, d), d)
        # norm = lambda d: nn.BatchnNorm2d(min(32, d), d)

        super(ConvNet, self).__init__(
            nn.Conv2d(in_channels, 32, **common),
            norm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, **common),
            norm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=2, **common),
            norm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, **common),
            norm(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, stride=2,**common),
            norm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, **common),
            norm(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, stride=2,**common),
            norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, **common),
            norm(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
