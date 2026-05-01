"""Wide Residual Networks for CIFAR.

Zagoruyko & Komodakis (2016), "Wide Residual Networks". The pre-activation
variant is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WideBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.project: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.project = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x), inplace=True)
        shortcut = self.project(h) if not isinstance(self.project, nn.Identity) else x
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen: int = 10, num_classes: int = 100, dropout: float = 0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]

        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.stage2 = self._make_stage(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.stage3 = self._make_stage(widths[2], widths[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int, dropout: float):
        layers = [_WideBasicBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(n_blocks - 1):
            layers.append(_WideBasicBlock(out_ch, out_ch, stride=1, dropout=dropout))
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
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def wrn_28_10(num_classes: int = 100, dropout: float = 0.3) -> WideResNet:
    return WideResNet(depth=28, widen=10, num_classes=num_classes, dropout=dropout)
