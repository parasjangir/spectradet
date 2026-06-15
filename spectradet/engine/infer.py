"""Turn raw decoded outputs into final detections (confidence filter + NMS)."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from ..utils.boxes import batched_nms, cxcywh_to_xyxy


@torch.no_grad()
def postprocess(
    decoded: Dict[str, Tensor],
    num_classes: int,
    conf_thr: float = 0.05,
    nms_thr: float = 0.6,
    max_det: int = 300,
) -> List[Dict[str, Tensor]]:
    """Per image -> dict(boxes [n,4] xyxy, scores [n], labels [n])."""
    cls = decoded["cls_logits"].sigmoid()        # [B,A,C]
    obj = decoded["obj_logits"].sigmoid()        # [B,A,1]
    boxes = cxcywh_to_xyxy(decoded["boxes_cxcywh"])  # [B,A,4]
    scores = cls * obj                            # joint confidence

    results: List[Dict[str, Tensor]] = []
    for b in range(scores.shape[0]):
        conf, labels = scores[b].max(dim=1)       # best class per anchor
        keep = conf > conf_thr
        bx, sc, lb = boxes[b][keep], conf[keep], labels[keep]
        if bx.numel() == 0:
            results.append({"boxes": bx.new_zeros((0, 4)), "scores": sc.new_zeros((0,)),
                            "labels": lb.new_zeros((0,), dtype=torch.long)})
            continue
        kept = batched_nms(bx, sc, lb, nms_thr)[:max_det]
        results.append({"boxes": bx[kept], "scores": sc[kept], "labels": lb[kept]})
    return results
