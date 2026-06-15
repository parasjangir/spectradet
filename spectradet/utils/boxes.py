"""Bounding-box maths — all hand-rolled (no torchvision).

Canonical format across the codebase is ``xyxy`` = (x1, y1, x2, y2) in pixels.
The head predicts boxes in ``cxcywh`` (centre-x, centre-y, w, h) and converts.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# format conversions
# --------------------------------------------------------------------------- #
def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


# --------------------------------------------------------------------------- #
# areas + IoU
# --------------------------------------------------------------------------- #
def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def box_iou(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise IoU. ``a``:[N,4], ``b``:[M,4] (xyxy) -> [N,M]."""
    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    return inter / union.clamp(min=eps)


def bbox_ciou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise Complete-IoU. ``pred``/``target`` broadcast over [...,4] (xyxy).

    Returns CIoU in (-1, 1]; the regression loss is ``1 - ciou``.
    CIoU = IoU - centre-distance penalty - aspect-ratio penalty, so it gives a
    useful gradient even when two boxes don't overlap (unlike raw IoU).
    """
    # intersection
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
    area_t = (target[..., 2] - target[..., 0]).clamp(min=0) * (target[..., 3] - target[..., 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # smallest enclosing box diagonal
    cx1 = torch.min(pred[..., 0], target[..., 0])
    cy1 = torch.min(pred[..., 1], target[..., 1])
    cx2 = torch.max(pred[..., 2], target[..., 2])
    cy2 = torch.max(pred[..., 3], target[..., 3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    # centre distance
    pcx, pcy = (pred[..., 0] + pred[..., 2]) / 2, (pred[..., 1] + pred[..., 3]) / 2
    tcx, tcy = (target[..., 0] + target[..., 2]) / 2, (target[..., 1] + target[..., 3]) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # aspect-ratio consistency
    pw = (pred[..., 2] - pred[..., 0]).clamp(min=eps)
    ph = (pred[..., 3] - pred[..., 1]).clamp(min=eps)
    tw = (target[..., 2] - target[..., 0]).clamp(min=eps)
    th = (target[..., 3] - target[..., 1]).clamp(min=eps)
    v = (4 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - rho2 / c2 - alpha * v


# --------------------------------------------------------------------------- #
# NMS (greedy, hand-rolled)
# --------------------------------------------------------------------------- #
def nms(boxes: Tensor, scores: Tensor, iou_thr: float = 0.5) -> Tensor:
    """Greedy non-maximum suppression. Returns kept indices (sorted by score)."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep)


def batched_nms(boxes: Tensor, scores: Tensor, labels: Tensor, iou_thr: float = 0.5) -> Tensor:
    """Per-class NMS via the coordinate-offset trick (boxes of different classes
    are shifted far apart so they can never suppress each other)."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    offsets = labels.to(boxes) * (max_coord + 1)
    return nms(boxes + offsets[:, None], scores, iou_thr)
