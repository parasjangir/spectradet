"""The custom FFT module — a learnable global spectral filter.

Why this helps with degradation
--------------------------------
A convolution has a *local* receptive field, so a single conv can't easily
reason about global frequency statistics.  But degradations live in the spectrum:
blur is a low-pass filter (it attenuates high frequencies), sensor noise adds a
broadband high-frequency floor, JPEG injects block-frequency artefacts.

``SpectralGate`` moves a feature map into the Fourier domain with a 2-D real FFT,
multiplies it by a *learnable complex gain per frequency* (an element-wise filter
with a full-image receptive field), then transforms back.  The network can thus
learn to suppress noise frequencies and re-emphasise the high frequencies that
blur destroyed — making it degradation-aware.  This is closely related to the
"global filter" idea (GFNet) adapted here as a residual side-branch on the CNN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct


class SpectralGate(nn.Module):
    def __init__(self, channels: int, h: int, w: int):
        super().__init__()
        self.h, self.w = h, w
        self.pre = ConvBNAct(channels, channels, 1)
        # complex filter over the rfft2 spectrum: [C, H, W//2+1] stored as (real, imag)
        self.weight = nn.Parameter(torch.randn(channels, h, w // 2 + 1, 2) * 0.02)
        self.post = ConvBNAct(channels, channels, 1)

    def _resize_filter(self, wt: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Bilinearly resize the learned filter if the feature map size changed
        (keeps the module valid under multi-scale / variable input sizes)."""
        target = (H, W // 2 + 1)
        real = F.interpolate(wt.real.unsqueeze(0), size=target, mode="bilinear", align_corners=False)
        imag = F.interpolate(wt.imag.unsqueeze(0), size=target, mode="bilinear", align_corners=False)
        return torch.complex(real[0], imag[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        y = self.pre(x)
        freq = torch.fft.rfft2(y.float(), norm="ortho")          # [B,C,H,W//2+1] complex
        wt = torch.view_as_complex(self.weight)
        if (H, W) != (self.h, self.w):
            wt = self._resize_filter(wt, H, W)
        freq = freq * wt                                          # global per-frequency gain
        y = torch.fft.irfft2(freq, s=(H, W), norm="ortho").to(x.dtype)
        return residual + self.post(y)
