import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        relu=True,
        same_padding=False,
        bn=False,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MCNN(nn.Module):
    """Multi-column CNN for crowd/tree counting density estimation."""

    def __init__(self, bn=False):
        super().__init__()

        self.branch1 = nn.Sequential(
            Conv2d(1, 16, 9, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(16, 32, 7, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7, same_padding=True, bn=bn),
            Conv2d(16, 8, 7, same_padding=True, bn=bn),
        )

        self.branch2 = nn.Sequential(
            Conv2d(1, 20, 7, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(20, 40, 5, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(40, 20, 5, same_padding=True, bn=bn),
            Conv2d(20, 10, 5, same_padding=True, bn=bn),
        )

        self.branch3 = nn.Sequential(
            Conv2d(1, 24, 5, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(24, 48, 3, same_padding=True, bn=bn),
            nn.MaxPool2d(2),
            Conv2d(48, 24, 3, same_padding=True, bn=bn),
            Conv2d(24, 12, 3, same_padding=True, bn=bn),
        )

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn, relu=False))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.fuse(x)
