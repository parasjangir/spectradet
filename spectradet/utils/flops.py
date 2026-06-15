"""Analytical FLOPs (MACs) counter via forward hooks on Conv2d / Linear.

This is the *honest* efficiency metric for depthwise models: a depthwise conv has
``groups == in_channels`` so its MACs are ~``k*k`` per output element instead of
``in_ch*k*k`` — exactly the compute reduction that params alone understate.

Counts convolution + linear MACs (which dominate); pointwise FFT ops are not
included and are a small fraction of total compute.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def count_macs(model, img_size: int = 256, device: str = "cpu") -> int:
    model = model.to(device).eval()
    total = 0
    hooks = []

    def conv_hook(m, inp, out):
        nonlocal total
        oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
        kh, kw = m.kernel_size
        total += oh * ow * oc * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        nonlocal total
        total += m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))

    x = torch.randn(1, 3, img_size, img_size, device=device)
    model.decode(model(x))
    for h in hooks:
        h.remove()
    return total


def count_gflops(model, img_size: int = 256, device: str = "cpu") -> float:
    """GFLOPs = 2 * MACs (multiply + add), in billions."""
    return 2 * count_macs(model, img_size, device) / 1e9
