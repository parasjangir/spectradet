"""Dataset plumbing: collate + a factory that builds train/val sets from config."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .degradations import RandomMultiDegradation, default_degradations
from .synthetic import SyntheticDegradedDataset


def detection_collate(batch) -> Tuple[Tensor, List[Dict]]:
    """Stack images, keep per-image targets as a list (variable #objects)."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


def build_datasets(cfg: dict):
    """Return ``(train_ds, val_ds, class_names)`` from a config dict."""
    data = cfg["data"]
    name = data.get("name", "synthetic")

    if name == "synthetic":
        deg = RandomMultiDegradation(
            default_degradations(),
            min_n=data.get("min_degradations", 1),
            max_n=data.get("max_degradations", 3),
            p=data.get("degrade_prob", 0.9),
        )
        common = dict(
            img_size=data.get("img_size", 256),
            max_objects=data.get("max_objects", 6),
            multi_degradation=deg,
            degrade=data.get("degrade", True),
        )
        train_ds = SyntheticDegradedDataset(
            length=data.get("train_size", 2000), base_seed=1, **common
        )
        val_ds = SyntheticDegradedDataset(
            length=data.get("val_size", 400), base_seed=999, **common
        )
        return train_ds, val_ds, SyntheticDegradedDataset.CLASSES

    if name == "voc":
        from .voc import VOCDegradedDataset

        deg = RandomMultiDegradation(
            default_degradations(),
            min_n=data.get("min_degradations", 1),
            max_n=data.get("max_degradations", 3),
            p=data.get("degrade_prob", 0.9),
        )
        train_ds = VOCDegradedDataset(
            root=data["root"], split=data.get("train_split", "train"),
            img_size=data.get("img_size", 256),
            multi_degradation=deg if data.get("degrade", True) else None,
        )
        val_ds = VOCDegradedDataset(
            root=data["root"], split=data.get("val_split", "val"),
            img_size=data.get("img_size", 256),
            multi_degradation=deg if data.get("degrade", True) else None,
        )
        return train_ds, val_ds, train_ds.classes

    raise ValueError(f"unknown dataset '{name}'")
