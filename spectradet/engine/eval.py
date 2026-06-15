"""From-scratch mean Average Precision (COCO-style 101-point interpolation)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.boxes import box_iou
from .infer import postprocess


def _ap_single_class(preds: List[Dict], gts: List[Dict], cls: int, iou_thr: float) -> float:
    """AP for one class at one IoU threshold; returns nan if the class has no GT."""
    n_gt = 0
    gt_boxes_per_img, gt_used = {}, {}
    for i, g in enumerate(gts):
        mask = g["labels"] == cls
        gb = g["boxes"][mask]
        gt_boxes_per_img[i] = gb
        gt_used[i] = np.zeros(len(gb), dtype=bool)
        n_gt += len(gb)
    if n_gt == 0:
        return float("nan")

    dets = []  # (score, img_idx, box)
    for i, p in enumerate(preds):
        mask = p["labels"] == cls
        for box, s in zip(p["boxes"][mask], p["scores"][mask]):
            dets.append((float(s), i, box))
    if not dets:
        return 0.0
    dets.sort(key=lambda d: -d[0])

    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for di, (_, i, box) in enumerate(dets):
        gb = gt_boxes_per_img[i]
        if len(gb) == 0:
            fp[di] = 1
            continue
        ious = box_iou(box[None], gb)[0]
        j = int(ious.argmax())
        if float(ious[j]) >= iou_thr and not gt_used[i][j]:
            tp[di] = 1
            gt_used[i][j] = True
        else:
            fp[di] = 1

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / (n_gt + 1e-9)
    prec = tp_c / (tp_c + fp_c + 1e-9)
    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return float(ap)


def compute_map(preds: List[Dict], gts: List[Dict], num_classes: int, iou_thrs=None) -> Dict:
    if iou_thrs is None:
        iou_thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # .50:.05:.95
    aps = np.full((num_classes, len(iou_thrs)), np.nan)
    for c in range(num_classes):
        for ti, thr in enumerate(iou_thrs):
            aps[c, ti] = _ap_single_class(preds, gts, c, thr)

    def safe_mean(x):
        return float(np.nanmean(x)) if not np.all(np.isnan(x)) else 0.0

    return {
        "map": safe_mean(aps),
        "map50": safe_mean(aps[:, 0]),
        "map75": safe_mean(aps[:, 5]),
        "per_class_ap50": np.nan_to_num(aps[:, 0]).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, num_classes: int,
             conf_thr: float = 0.05, nms_thr: float = 0.6) -> Dict:
    model.eval()
    preds_all, gts_all = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        dec = model.decode(model(imgs))
        for r, t in zip(postprocess(dec, num_classes, conf_thr, nms_thr), targets):
            preds_all.append({k: v.cpu() for k, v in r.items()})
            gts_all.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})
    return compute_map(preds_all, gts_all, num_classes)
