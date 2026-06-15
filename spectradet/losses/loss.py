"""SpectraDet detection loss = IoU-guided objectness + cls + CIoU regression."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.boxes import bbox_ciou, cxcywh_to_xyxy
from .simota import simota_assign


class SpectraDetLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cls_weight: float = 1.0,
        obj_weight: float = 1.0,
        reg_weight: float = 5.0,
        iou_guided: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.reg_weight = reg_weight
        self.iou_guided = iou_guided

    def forward(self, decoded: Dict[str, Tensor], targets: List[Dict]) -> Dict[str, Tensor]:
        cls_logits = decoded["cls_logits"]      # [B,A,C]
        obj_logits = decoded["obj_logits"]      # [B,A,1]
        boxes_cxcywh = decoded["boxes_cxcywh"]  # [B,A,4]
        centers = decoded["centers"]            # [A,2]
        strides = decoded["strides"]            # [A]
        B, A, C = cls_logits.shape
        device = cls_logits.device

        obj_targets = torch.zeros((B, A, 1), device=device)
        cls_loss = torch.zeros((), device=device)
        reg_loss = torch.zeros((), device=device)
        total_fg = 0

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)
            pred_xyxy = cxcywh_to_xyxy(boxes_cxcywh[b])           # [A,4]

            if gt_boxes.numel() == 0:
                continue  # all-background image: only objectness (target already 0)

            fg_mask, matched_gt, matched_iou = simota_assign(
                pred_xyxy, cls_logits[b], obj_logits[b], centers, strides, gt_boxes, gt_labels
            )
            n_fg = int(fg_mask.sum())
            if n_fg == 0:
                continue
            total_fg += n_fg

            # objectness target: IoU of matched prediction (IoU-guided) or hard 1.0
            obj_tgt = matched_iou.detach().clamp(0, 1) if self.iou_guided else torch.ones(n_fg, device=device)
            obj_targets[b, fg_mask, 0] = obj_tgt

            # classification (one-hot over matched GT class), positives only
            cls_tgt = F.one_hot(gt_labels[matched_gt], C).float()
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                cls_logits[b, fg_mask], cls_tgt, reduction="sum"
            )

            # regression: 1 - CIoU on matched boxes
            ciou = bbox_ciou(pred_xyxy[fg_mask], gt_boxes[matched_gt])
            reg_loss = reg_loss + (1.0 - ciou).sum()

        norm = max(total_fg, 1)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_targets, reduction="sum") / norm
        cls_loss = cls_loss / norm
        reg_loss = reg_loss / norm

        total = self.cls_weight * cls_loss + self.obj_weight * obj_loss + self.reg_weight * reg_loss
        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_obj": obj_loss.detach(),
            "loss_reg": reg_loss.detach(),
            "num_fg": torch.tensor(float(total_fg)),
        }


def build_loss(cfg: dict, num_classes: int) -> SpectraDetLoss:
    lc = cfg.get("loss", {})
    return SpectraDetLoss(
        num_classes=num_classes,
        cls_weight=lc.get("cls_weight", 1.0),
        obj_weight=lc.get("obj_weight", 1.0),
        reg_weight=lc.get("reg_weight", 5.0),
        iou_guided=lc.get("iou_guided", True),
    )
