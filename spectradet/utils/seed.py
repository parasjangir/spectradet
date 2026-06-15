"""Reproducibility + device helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> None:
    """Seed python / numpy / torch so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a compute device. ``auto`` -> cuda > mps (Apple Silicon) > cpu."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
