"""Backbone producing P3/P4/P5 (strides 8/16/32).

``hires_blocks > 0`` inserts a dense high-resolution encoder right after the stem
(operating at stride 2, the most expensive resolution).  The lite model uses 0
(it downsamples aggressively for speed); the DrebNET-style baseline uses several,
which is what makes it genuinely slow in wall-clock — high-resolution dense convs
dominate latency, mirroring a real deblurring/restoration encoder.
"""
from __future__ import annotations

import torch.nn as nn

from .layers import Bottleneck, ConvBNAct, CSPStage


class Backbone(nn.Module):
    def __init__(
        self,
        width: int = 24,
        depth: int = 1,
        depthwise: bool = True,
        in_ch: int = 3,
        hires_blocks: int = 0,
        hires_ch: int = 64,
    ):
        super().__init__()
        w = width
        self.stem = ConvBNAct(in_ch, w, 3, s=2)                              # /2

        # optional heavyweight high-resolution encoder (baseline only)
        self.hires = nn.Identity()
        if hires_blocks > 0:
            layers = [ConvBNAct(w, hires_ch, 3)]
            layers += [Bottleneck(hires_ch, depthwise=False) for _ in range(hires_blocks)]
            layers += [ConvBNAct(hires_ch, w, 3)]
            self.hires = nn.Sequential(*layers)

        self.stage1 = CSPStage(w, w * 2, n=depth, depthwise=depthwise)        # /4
        self.stage2 = CSPStage(w * 2, w * 4, n=depth * 2, depthwise=depthwise)  # /8  -> P3
        self.stage3 = CSPStage(w * 4, w * 8, n=depth * 2, depthwise=depthwise)  # /16 -> P4
        self.stage4 = CSPStage(w * 8, w * 16, n=depth, depthwise=depthwise)     # /32 -> P5
        self.out_channels = (w * 4, w * 8, w * 16)

    def forward(self, x):
        x = self.stem(x)
        x = self.hires(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]
