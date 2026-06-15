"""SpectraDet — backbone (+ FFT) -> PAN-FPN -> decoupled head, with decoding."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .backbone import Backbone
from .fft_module import SpectralGate
from .head import DecoupledHead
from .neck import PAFPN


class SpectraDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 24,
        depth: int = 1,
        neck_ch: int = 64,
        depthwise: bool = True,
        img_size: int = 256,
        fft_levels: Tuple[int, ...] = (2,),
        n_head_convs: int = 2,
        hires_blocks: int = 0,
        hires_ch: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = (8, 16, 32)
        self.img_size = img_size

        self.backbone = Backbone(width, depth, depthwise, hires_blocks=hires_blocks, hires_ch=hires_ch)
        c3, c4, c5 = self.backbone.out_channels
        chans = (c3, c4, c5)
        sizes = (img_size // 8, img_size // 16, img_size // 32)

        # FFT spectral gates on selected backbone levels (0=P3, 1=P4, 2=P5)
        self.fft_levels = tuple(fft_levels)
        self.fft = nn.ModuleDict(
            {str(lvl): SpectralGate(chans[lvl], sizes[lvl], sizes[lvl]) for lvl in self.fft_levels}
        )

        self.neck = PAFPN((c3, c4, c5), neck_ch, depthwise)
        self.head = DecoupledHead(num_classes, neck_ch, n_head_convs, self.strides, depthwise)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        feats = self.backbone(x)
        for lvl in self.fft_levels:
            feats[lvl] = self.fft[str(lvl)](feats[lvl])
        feats = self.neck(feats)
        return self.head(feats)

    # ------------------------------------------------------------------ #
    def decode(self, outputs) -> dict:
        """Flatten multi-level raw outputs into per-anchor tensors.

        Returns a dict with:
          cls_logits [B, A, C], obj_logits [B, A, 1], boxes_cxcywh [B, A, 4] (px),
          centers [A, 2] (px, cell centres for the SimOTA centre-prior), strides [A].
        ``A`` = sum of H*W over the three levels.
        """
        cls_outs, reg_outs, obj_outs = outputs
        device = cls_outs[0].device
        cls_l, obj_l, box_l, ctr_l, str_l = [], [], [], [], []

        for i, (c, r, o) in enumerate(zip(cls_outs, reg_outs, obj_outs)):
            B, _, H, W = c.shape
            stride = self.strides[i]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
            )
            grid = torch.stack((xv, yv), dim=2).reshape(-1, 2).float()  # [HW,2] cell coords

            c = c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            o = o.permute(0, 2, 3, 1).reshape(B, -1, 1)
            r = r.permute(0, 2, 3, 1).reshape(B, -1, 4)

            xy = (r[..., :2] + grid) * stride          # anchor-free centre decode
            wh = torch.exp(r[..., 2:]) * stride
            box = torch.cat([xy, wh], dim=-1)          # cxcywh in pixels

            cls_l.append(c)
            obj_l.append(o)
            box_l.append(box)
            ctr_l.append((grid + 0.5) * stride)        # cell centres for centre-prior
            str_l.append(torch.full((H * W,), float(stride), device=device))

        return {
            "cls_logits": torch.cat(cls_l, dim=1),
            "obj_logits": torch.cat(obj_l, dim=1),
            "boxes_cxcywh": torch.cat(box_l, dim=1),
            "centers": torch.cat(ctr_l, dim=0),
            "strides": torch.cat(str_l, dim=0),
        }


def build_model(cfg: dict, num_classes: int) -> SpectraDet:
    m = cfg["model"]
    return SpectraDet(
        num_classes=num_classes,
        width=m.get("width", 24),
        depth=m.get("depth", 1),
        neck_ch=m.get("neck_ch", 64),
        depthwise=m.get("depthwise", True),
        img_size=cfg["data"].get("img_size", 256),
        fft_levels=tuple(m.get("fft_levels", [2])),
        n_head_convs=m.get("n_head_convs", 2),
        hires_blocks=m.get("hires_blocks", 0),
        hires_ch=m.get("hires_ch", 64),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
