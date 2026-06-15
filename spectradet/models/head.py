"""Decoupled anchor-free detection head (YOLOX-style)."""
from __future__ import annotations

import math

import torch.nn as nn

from .layers import ConvBNAct, make_conv


class DecoupledHead(nn.Module):
    """Separate classification and regression branches per FPN level.

    Decoupling cls from reg consistently helps detection accuracy: the two tasks
    want different features.  Each level predicts, per spatial cell:
      * ``num_classes`` class logits,
      * 4 box terms (tx, ty, tw, th) decoded anchor-free,
      * 1 objectness logit.
    """

    def __init__(self, num_classes, in_ch=64, n_convs=2, strides=(8, 16, 32), depthwise=True):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_pred = nn.ModuleList()
        self.obj_pred = nn.ModuleList()

        for _ in strides:
            self.stems.append(ConvBNAct(in_ch, in_ch, 1))
            self.cls_convs.append(nn.Sequential(*[make_conv(in_ch, in_ch, 3, depthwise=depthwise) for _ in range(n_convs)]))
            self.reg_convs.append(nn.Sequential(*[make_conv(in_ch, in_ch, 3, depthwise=depthwise) for _ in range(n_convs)]))
            self.cls_pred.append(nn.Conv2d(in_ch, num_classes, 1))
            self.reg_pred.append(nn.Conv2d(in_ch, 4, 1))
            self.obj_pred.append(nn.Conv2d(in_ch, 1, 1))

        self._init_biases()

    def _init_biases(self, prior: float = 0.01):
        # Focal-style prior so early training isn't swamped by the background class.
        b = -math.log((1 - prior) / prior)
        for m in self.cls_pred:
            nn.init.constant_(m.bias, b)
        for m in self.obj_pred:
            nn.init.constant_(m.bias, b)

    def forward(self, feats):
        cls_outs, reg_outs, obj_outs = [], [], []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            cls_outs.append(self.cls_pred[i](cls_feat))
            reg_outs.append(self.reg_pred[i](reg_feat))
            obj_outs.append(self.obj_pred[i](reg_feat))
        return cls_outs, reg_outs, obj_outs
