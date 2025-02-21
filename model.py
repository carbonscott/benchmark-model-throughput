import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNetScaler(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** min(i, 3))
            layers.append(ResNetBlock(in_ch, out_ch))
            in_ch = out_ch

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layers(x)
        return self.avgpool(x)
