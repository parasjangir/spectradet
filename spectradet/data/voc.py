"""Real-data loader: Pascal-VOC-style folders of *real* blurred/degraded images.

This is the path the project's dataset contribution plugs into.  Point ``root``
at a directory laid out as::

    root/
      JPEGImages/        *.jpg
      Annotations/       *.xml   (VOC format: <object><name>..<bndbox>..)
      ImageSets/Main/    train.txt  val.txt   (one image id per line)

Real images can be left as-is (genuine in-the-wild degradation) or further
degraded via ``multi_degradation`` for controlled robustness experiments.
No torchvision required — annotations are parsed straight from XML.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradations import RandomMultiDegradation


class VOCDegradedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 256,
        classes: Optional[List[str]] = None,
        multi_degradation: Optional[RandomMultiDegradation] = None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.multi_degradation = multi_degradation

        ids_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"split file not found: {ids_file}")
        self.ids = [x.strip() for x in ids_file.read_text().splitlines() if x.strip()]

        if classes is None:
            classes = self._scan_classes()
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _scan_classes(self) -> List[str]:
        names = set()
        for img_id in self.ids:
            xml = self.root / "Annotations" / f"{img_id}.xml"
            if not xml.exists():
                continue
            for obj in ET.parse(xml).getroot().findall("object"):
                name = obj.findtext("name")
                if name:
                    names.add(name.strip())
        return sorted(names)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_annotation(self, img_id: str, sx: float, sy: float):
        xml = self.root / "Annotations" / f"{img_id}.xml"
        boxes, labels = [], []
        if xml.exists():
            for obj in ET.parse(xml).getroot().findall("object"):
                name = obj.findtext("name")
                if name not in self.cls_to_idx:
                    continue
                bb = obj.find("bndbox")
                x1 = float(bb.findtext("xmin")) * sx
                y1 = float(bb.findtext("ymin")) * sy
                x2 = float(bb.findtext("xmax")) * sx
                y2 = float(bb.findtext("ymax")) * sy
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cls_to_idx[name])
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.ids[idx]
        path = self.root / "JPEGImages" / f"{img_id}.jpg"
        img = Image.open(path).convert("RGB")
        ow, oh = img.size
        sx, sy = self.img_size / ow, self.img_size / oh
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        boxes, labels = self._load_annotation(img_id, sx, sy)

        arr = np.array(img)
        applied: List[str] = []
        if self.multi_degradation is not None:
            arr, applied = self.multi_degradation(arr, np.random.default_rng(idx))

        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": idx,
            "applied": applied,
            "orig_size": (oh, ow),
        }
        return tensor, target
