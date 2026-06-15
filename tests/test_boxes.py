import torch

from spectradet.utils.boxes import (
    batched_nms,
    bbox_ciou,
    box_iou,
    cxcywh_to_xyxy,
    nms,
    xyxy_to_cxcywh,
)


def test_format_roundtrip():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 4.0, 6.0]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes, atol=1e-5)


def test_iou_known_values():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)[0]
    assert torch.isclose(iou[0], torch.tensor(1.0), atol=1e-4)          # identical
    assert torch.isclose(iou[1], torch.tensor(25 / 175), atol=1e-4)     # quarter overlap
    assert torch.isclose(iou[2], torch.tensor(0.0), atol=1e-4)          # disjoint


def test_ciou_bounds_and_perfect():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.isclose(bbox_ciou(box, box)[0], torch.tensor(1.0), atol=1e-4)
    far = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    assert bbox_ciou(box, far)[0] < 0  # disjoint + distant -> negative


def test_nms_suppresses_overlap():
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert set(keep.tolist()) == {0, 2}  # box 1 suppressed by box 0


def test_batched_nms_per_class():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])  # different classes -> both kept
    keep = batched_nms(boxes, scores, labels, iou_thr=0.5)
    assert set(keep.tolist()) == {0, 1}
