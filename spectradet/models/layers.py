"""Reusable conv building blocks (depthwise-separable for the lite model)."""
from __future__ import annotations

import torch.nn as nn


def autopad(k, p=None):
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    """Conv -> BatchNorm -> SiLU."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise-separable conv = depthwise (per-channel) + pointwise (1x1).

    This is the main lever for the 4x parameter reduction: a kxk conv costs
    ``c1*c2*k*k`` params, while depthwise-separable costs ``c1*k*k + c1*c2``.
    """

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k, s, g=c1, act=act)
        self.pw = ConvBNAct(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


def make_conv(c1, c2, k=3, s=1, depthwise=True):
    if depthwise and k > 1:
        return DWConv(c1, c2, k, s)
    return ConvBNAct(c1, c2, k, s)


class Bottleneck(nn.Module):
    """Residual bottleneck: 1x1 squeeze -> 3x3 -> add."""

    def __init__(self, c, depthwise=True, expansion=0.5):
        super().__init__()
        hidden = max(8, int(c * expansion))
        self.cv1 = ConvBNAct(c, hidden, 1)
        self.cv2 = make_conv(hidden, c, 3, depthwise=depthwise)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPStage(nn.Module):
    """Downsample (stride 2) then ``n`` residual bottlenecks."""

    def __init__(self, c1, c2, n=1, depthwise=True):
        super().__init__()
        self.down = make_conv(c1, c2, 3, s=2, depthwise=depthwise)
        self.blocks = nn.Sequential(*[Bottleneck(c2, depthwise) for _ in range(n)])

    def forward(self, x):
        return self.blocks(self.down(x))
