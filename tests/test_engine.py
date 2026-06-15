import torch

from spectradet.engine.eval import compute_map
from spectradet.engine.infer import postprocess


def test_postprocess_filters_and_returns_dicts():
    # one anchor with high score, others near zero
    A = 5
    decoded = {
        "cls_logits": torch.full((1, A, 3), -10.0),
        "obj_logits": torch.full((1, A, 1), -10.0),
        "boxes_cxcywh": torch.tensor([[[30.0, 30.0, 20.0, 20.0]] * A]),
    }
    decoded["cls_logits"][0, 0, 1] = 10.0   # anchor 0 -> class 1, high prob
    decoded["obj_logits"][0, 0, 0] = 10.0
    out = postprocess(decoded, num_classes=3, conf_thr=0.3)[0]
    assert out["boxes"].shape[0] == 1
    assert out["labels"].item() == 1


def test_map_perfect_prediction():
    gts = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    preds = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
              "scores": torch.tensor([0.95]), "labels": torch.tensor([0])}]
    m = compute_map(preds, gts, num_classes=3)
    assert m["map50"] > 0.99
    assert m["map"] > 0.99


def test_map_zero_when_wrong_class():
    gts = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
    preds = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
              "scores": torch.tensor([0.95]), "labels": torch.tensor([2])}]
    m = compute_map(preds, gts, num_classes=3)
    assert m["map50"] == 0.0
