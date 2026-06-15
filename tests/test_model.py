import torch

from spectradet.models.detector import SpectraDet, count_parameters
from spectradet.losses.loss import SpectraDetLoss
from spectradet.losses.simota import simota_assign


def _tiny_model():
    return SpectraDet(num_classes=4, width=16, neck_ch=32, img_size=128, fft_levels=(2,))


def test_forward_decode_shapes():
    model = _tiny_model().eval()
    x = torch.randn(2, 3, 128, 128)
    dec = model.decode(model(x))
    A = (128 // 8) ** 2 + (128 // 16) ** 2 + (128 // 32) ** 2
    assert dec["cls_logits"].shape == (2, A, 4)
    assert dec["boxes_cxcywh"].shape == (2, A, 4)
    assert dec["centers"].shape == (A, 2)


def test_loss_backward_runs():
    model = _tiny_model().train()
    crit = SpectraDetLoss(num_classes=4)
    x = torch.randn(2, 3, 128, 128)
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20.0, 30.0, 80.0, 90.0], [5.0, 5.0, 25.0, 25.0]]),
         "labels": torch.tensor([0, 3])},
    ]
    out = crit(model.decode(model(x)), targets)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert any(p.grad is not None for p in model.parameters())


def test_empty_target_image_is_handled():
    model = _tiny_model().train()
    crit = SpectraDetLoss(num_classes=4)
    x = torch.randn(1, 3, 128, 128)
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]
    out = crit(model.decode(model(x)), targets)
    out["loss"].backward()  # background-only image -> objectness loss only
    assert torch.isfinite(out["loss"])


def test_simota_picks_good_anchors():
    xs = torch.linspace(8, 72, 9)
    cx, cy = torch.meshgrid(xs, xs, indexing="ij")
    centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)
    A = centers.shape[0]
    strides = torch.full((A,), 8.0)
    gt = torch.tensor([[20.0, 20.0, 60.0, 60.0]])
    labels = torch.tensor([0])
    # each anchor predicts a 40px box centred on itself
    pred = torch.stack(
        [centers[:, 0] - 20, centers[:, 1] - 20, centers[:, 0] + 20, centers[:, 1] + 20], dim=1
    )
    mask, matched_gt, matched_iou = simota_assign(
        pred, torch.zeros(A, 2), torch.zeros(A, 1), centers, strides, gt, labels
    )
    assert mask.sum() >= 1
    assert (matched_gt == 0).all()
    assert matched_iou.min() >= 0 and matched_iou.max() <= 1
