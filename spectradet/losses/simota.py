"""SimOTA — Simplified Optimal Transport Assignment (YOLOX), from scratch.

For one image, decide *which* anchors are positive for *which* ground-truth box.
Naive "one GT -> one anchor" wastes signal; SimOTA instead:
  1. restricts candidates with a centre-prior (anchor centre near/inside the GT),
  2. scores each (GT, candidate) pair by  cls_cost + 3*iou_cost,
  3. gives each GT a *dynamic* number k of positives (k = sum of its top IoUs),
  4. resolves anchors claimed by multiple GTs in favour of the lowest cost.

Returns, per image, the positive-anchor mask plus the matched GT index and the
matched IoU (the latter feeds the IoU-guided objectness target).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils.boxes import box_iou


@torch.no_grad()
def _in_boxes_and_center(
    gt_xyxy: Tensor, centers: Tensor, strides: Tensor, radius: float = 2.5
) -> Tuple[Tensor, Tensor]:
    """Return (fg_mask [A], in_boxes_and_center [G, num_fg])."""
    x = centers[:, 0]
    y = centers[:, 1]
    gl, gt_, gr, gb = gt_xyxy[:, 0], gt_xyxy[:, 1], gt_xyxy[:, 2], gt_xyxy[:, 3]

    # anchor centre inside the GT box
    b_l = x[None, :] - gl[:, None]
    b_r = gr[:, None] - x[None, :]
    b_t = y[None, :] - gt_[:, None]
    b_b = gb[:, None] - y[None, :]
    in_boxes = torch.stack([b_l, b_t, b_r, b_b], dim=-1).amin(-1) > 0  # [G, A]

    # anchor centre within radius*stride of the GT centre
    gcx = (gl + gr) / 2
    gcy = (gt_ + gb) / 2
    rad = radius * strides[None, :]
    c_l = x[None, :] - (gcx[:, None] - rad)
    c_r = (gcx[:, None] + rad) - x[None, :]
    c_t = y[None, :] - (gcy[:, None] - rad)
    c_b = (gcy[:, None] + rad) - y[None, :]
    in_centers = torch.stack([c_l, c_t, c_r, c_b], dim=-1).amin(-1) > 0  # [G, A]

    fg_mask = (in_boxes | in_centers).any(0)                            # [A]
    in_boxes_and_center = (in_boxes[:, fg_mask] & in_centers[:, fg_mask])  # [G, num_fg]
    return fg_mask, in_boxes_and_center


@torch.no_grad()
def simota_assign(
    pred_xyxy: Tensor,        # [A,4] decoded predicted boxes
    cls_logits: Tensor,       # [A,C]
    obj_logits: Tensor,       # [A,1]
    centers: Tensor,          # [A,2]
    strides: Tensor,          # [A]
    gt_xyxy: Tensor,          # [G,4]
    gt_labels: Tensor,        # [G]
    radius: float = 2.5,
    n_candidate: int = 10,
):
    """Returns (fg_mask [A] bool, matched_gt [P], matched_iou [P])."""
    num_anchors = pred_xyxy.shape[0]
    device = pred_xyxy.device
    G = gt_xyxy.shape[0]
    if G == 0:
        return torch.zeros(num_anchors, dtype=torch.bool, device=device), \
            torch.zeros(0, dtype=torch.long, device=device), \
            torch.zeros(0, device=device)

    fg_mask, in_boxes_and_center = _in_boxes_and_center(gt_xyxy, centers, strides, radius)
    if fg_mask.sum() == 0:
        return fg_mask, torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, device=device)

    cand = pred_xyxy[fg_mask]                                   # [num_fg, 4]
    pair_iou = box_iou(gt_xyxy, cand).clamp(min=1e-8)           # [G, num_fg]
    iou_cost = -torch.log(pair_iou)

    # classification cost: combine class prob with objectness (joint confidence)
    cls_prob = (
        cls_logits[fg_mask].sigmoid() * obj_logits[fg_mask].sigmoid()
    ).sqrt().clamp(1e-6, 1 - 1e-6)                              # [num_fg, C]
    gt_onehot = F.one_hot(gt_labels, cls_logits.shape[1]).float()  # [G, C]
    cls_cost = F.binary_cross_entropy(
        cls_prob[None].expand(G, -1, -1),
        gt_onehot[:, None, :].expand(-1, cand.shape[0], -1),
        reduction="none",
    ).sum(-1)                                                   # [G, num_fg]

    cost = cls_cost + 3.0 * iou_cost + 1e5 * (~in_boxes_and_center)

    # --- dynamic-k matching ---
    matching = torch.zeros_like(cost)
    k = min(n_candidate, pair_iou.shape[1])
    topk_iou, _ = torch.topk(pair_iou, k, dim=1)
    dyn_k = torch.clamp(topk_iou.sum(1).int(), min=1)          # [G]
    for g in range(G):
        _, pos = torch.topk(cost[g], k=int(dyn_k[g]), largest=False)
        matching[g, pos] = 1.0

    # an anchor claimed by >1 GT -> keep the lowest-cost GT
    anchor_hits = matching.sum(0)
    if (anchor_hits > 1).any():
        multi = anchor_hits > 1
        _, best = torch.min(cost[:, multi], dim=0)
        matching[:, multi] = 0.0
        matching[best, multi] = 1.0

    fg_inboxes = matching.sum(0) > 0                            # [num_fg]
    matched_gt = matching[:, fg_inboxes].argmax(0)             # [P]
    matched_iou = pair_iou[matched_gt, fg_inboxes]             # [P]

    # lift fg_inboxes (over candidates) back to the full anchor grid
    final_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
    fg_idx = torch.nonzero(fg_mask, as_tuple=False).squeeze(1)
    final_mask[fg_idx[fg_inboxes]] = True
    return final_mask, matched_gt, matched_iou
