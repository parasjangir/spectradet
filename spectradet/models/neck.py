"""Tiny PAN-FPN: top-down + bottom-up fusion, uniform output channels."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct, make_conv


class PAFPN(nn.Module):
    def __init__(self, channels, out_ch: int = 64, depthwise: bool = True):
        super().__init__()
        c3, c4, c5 = channels
        # lateral 1x1 projections to a common width
        self.l3 = ConvBNAct(c3, out_ch, 1)
        self.l4 = ConvBNAct(c4, out_ch, 1)
        self.l5 = ConvBNAct(c5, out_ch, 1)
        # top-down smoothing
        self.smooth4 = make_conv(out_ch, out_ch, 3, depthwise=depthwise)
        self.smooth3 = make_conv(out_ch, out_ch, 3, depthwise=depthwise)
        # bottom-up path
        self.down3 = make_conv(out_ch, out_ch, 3, s=2, depthwise=depthwise)
        self.down4 = make_conv(out_ch, out_ch, 3, s=2, depthwise=depthwise)
        self.pan4 = make_conv(out_ch, out_ch, 3, depthwise=depthwise)
        self.pan5 = make_conv(out_ch, out_ch, 3, depthwise=depthwise)
        self.out_channels = (out_ch, out_ch, out_ch)

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.l5(c5)
        p4 = self.smooth4(self.l4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.smooth3(self.l3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest"))
        n4 = self.pan4(p4 + self.down3(p3))
        n5 = self.pan5(p5 + self.down4(n4))
        return [p3, n4, n5]
