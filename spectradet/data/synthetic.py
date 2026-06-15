"""Procedural multi-object scenes for offline, zero-download training.

Each image contains several textured shapes (6 classes) on a textured background,
with exact bounding boxes.  The whole scene is then passed through
``RandomMultiDegradation`` so the detector must localise/classify objects that
are simultaneously blurred, noisy, compressed and/or dark.

Determinism: sample ``idx`` is rendered from ``np.random.default_rng`` seeded by
``(base_seed, idx)``, so a dataset is fully reproducible and stable across epochs
(important for fair eval).  Set ``degrade=False`` to get the clean ground-truth
image (used by the sample-grid visualiser).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .degradations import RandomMultiDegradation, default_degradations


def _regular_polygon(cx, cy, r, n, rot=0.0):
    return [
        (cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n))
        for i in range(n)
    ]


def _star(cx, cy, r, n=5, rot=-math.pi / 2, inner=0.45):
    pts = []
    for i in range(2 * n):
        rr = r if i % 2 == 0 else r * inner
        a = rot + math.pi * i / n
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _bbox_of(points) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


class SyntheticDegradedDataset(Dataset):
    CLASSES = ["rectangle", "circle", "triangle", "ellipse", "star", "pentagon"]

    def __init__(
        self,
        length: int = 2000,
        img_size: int = 256,
        max_objects: int = 6,
        min_objects: int = 1,
        degrade: bool = True,
        base_seed: int = 0,
        multi_degradation: RandomMultiDegradation | None = None,
    ):
        self.length = length
        self.img_size = img_size
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.degrade = degrade
        self.base_seed = base_seed
        self.multi_degradation = multi_degradation or RandomMultiDegradation(default_degradations())

    # ------------------------------------------------------------------ #
    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    def __len__(self) -> int:
        return self.length

    # ------------------------------------------------------------------ #
    def _render(self, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = self.img_size
        # textured background: smooth low-freq gradient + speckle (gives FFT content)
        base = rng.integers(20, 90)
        grad = np.linspace(0, rng.integers(20, 70), S, dtype=np.float32)
        bg = base + grad[None, :] + grad[:, None] * 0.5
        bg = np.repeat(bg[..., None], 3, axis=2)
        bg += rng.normal(0, 6, bg.shape)
        bg = np.clip(bg, 0, 255).astype(np.uint8)

        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)

        n = int(rng.integers(self.min_objects, self.max_objects + 1))
        boxes: List[List[float]] = []
        labels: List[int] = []
        placed: List[Tuple[float, float, float, float]] = []

        for _ in range(n):
            for _try in range(12):  # rejection sampling to limit heavy overlap
                cls = int(rng.integers(0, len(self.CLASSES)))
                r = int(rng.integers(S // 12, S // 5))
                cx = int(rng.integers(r + 2, S - r - 2))
                cy = int(rng.integers(r + 2, S - r - 2))
                name = self.CLASSES[cls]
                if name == "rectangle":
                    w, h = r * 2, int(r * rng.uniform(1.0, 1.8))
                    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
                elif name == "circle":
                    box = (cx - r, cy - r, cx + r, cy + r)
                elif name == "ellipse":
                    rx, ry = r, int(r * rng.uniform(0.4, 0.7))
                    box = (cx - rx, cy - ry, cx + rx, cy + ry)
                elif name == "triangle":
                    pts = _regular_polygon(cx, cy, r, 3, rot=rng.uniform(0, 2 * math.pi))
                    box = _bbox_of(pts)
                elif name == "pentagon":
                    pts = _regular_polygon(cx, cy, r, 5, rot=rng.uniform(0, 2 * math.pi))
                    box = _bbox_of(pts)
                else:  # star
                    pts = _star(cx, cy, r, n=5, rot=rng.uniform(0, 2 * math.pi))
                    box = _bbox_of(pts)

                # reject if it overlaps an existing object too much
                if self._max_iou(box, placed) > 0.35:
                    continue

                color = tuple(int(c) for c in rng.integers(60, 256, size=3))
                outline = tuple(int(c) for c in rng.integers(0, 80, size=3))
                if name in ("rectangle",):
                    draw.rectangle(box, fill=color, outline=outline, width=2)
                elif name in ("circle", "ellipse"):
                    draw.ellipse(box, fill=color, outline=outline, width=2)
                else:
                    draw.polygon(pts, fill=color, outline=outline)

                placed.append(box)
                boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                labels.append(cls)
                break

        img_np = np.array(img)
        if not boxes:  # guarantee at least one object
            return self._render(np.random.default_rng(rng.integers(0, 2**31)))
        return img_np, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    @staticmethod
    def _max_iou(box, placed) -> float:
        if not placed:
            return 0.0
        bx1, by1, bx2, by2 = box
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        best = 0.0
        for px1, py1, px2, py2 in placed:
            ix1, iy1 = max(bx1, px1), max(by1, py1)
            ix2, iy2 = min(bx2, px2), min(by2, py2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            union = area_b + area_p - inter + 1e-6
            best = max(best, inter / union)
        return best

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        rng = np.random.default_rng(self.base_seed * 1_000_003 + idx)
        img, boxes, labels = self._render(rng)
        applied: List[str] = []
        if self.degrade:
            img, applied = self.multi_degradation(img, rng)

        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes),          # [N,4] xyxy (pixels)
            "labels": torch.from_numpy(labels),        # [N]
            "image_id": int(idx),
            "applied": applied,
            "orig_size": (self.img_size, self.img_size),
        }
        return tensor, target
