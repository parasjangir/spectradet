"""Latency + parameter benchmark — the headline efficiency claim."""
from __future__ import annotations

import time
from typing import Dict

import torch

from ..models.detector import count_parameters
from ..utils.flops import count_gflops


@torch.no_grad()
def benchmark_latency(model, img_size: int = 256, device: str = "cpu",
                      runs: int = 50, warmup: int = 10) -> Dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(1, 3, img_size, img_size, device=dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model.decode(model(x))
    sync()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.decode(model(x))
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    mean = sum(times) / len(times)
    return {
        "params": count_parameters(model),
        "params_M": round(count_parameters(model) / 1e6, 3),
        "gflops": round(count_gflops(model, img_size, device), 2),
        "latency_ms_mean": round(mean, 1),
        "latency_ms_median": round(times[len(times) // 2], 1),
        "fps": round(1000.0 / mean, 1),
        "device": device,
        "img_size": img_size,
    }
