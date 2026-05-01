"""ResNet-{20, 32, 44, 56, 110} for CIFAR.

Follows He et al. (2016) "Deep Residual Learning for Image Recognition" Sec.
4.2, option A shortcut (identity with zero-padding). Depth formula: for n
stacked blocks per stage, total depth = 6n + 2.

  ResNet-20:  n=3   (~0.27M params)
  ResNet-32:  n=5   (~0.47M params)
  ResNet-44:  n=7   (~0.66M params)
  ResNet-56:  n=9   (~0.86M params)
  ResNet-110: n=18  (~1.73M params)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class _IdentityPad(nn.Module):
    """Option A shortcut: identity + zero-pad channels for stride>1 blocks."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        pad = self.out_ch - self.in_ch
        if pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = _IdentityPad(in_ch, out_ch, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """ResNet-{6n+2} for 32x32 inputs."""

    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(16, 16, n, stride=1)
        self.layer2 = self._make_stage(16, 32, n, stride=2)
        self.layer3 = self._make_stage(32, 64, n, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=3, num_classes=num_classes)


def resnet32(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=5, num_classes=num_classes)


def resnet44(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=7, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=9, num_classes=num_classes)


def resnet110(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=18, num_classes=num_classes)
