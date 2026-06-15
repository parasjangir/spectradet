"""Visual demo: run a trained checkpoint on degraded val images (GT vs predicted).

    python scripts/demo.py --ckpt runs/lite/best.pt --num 6 --conf 0.3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from spectradet.data.dataset import build_datasets  # noqa: E402
from spectradet.engine.infer import postprocess  # noqa: E402
from spectradet.models.detector import build_model  # noqa: E402
from spectradet.utils.seed import get_device  # noqa: E402

COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _draw(ax, img, boxes, labels, classes, title, scores=None):
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        c = COLORS[int(label) % len(COLORS)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.8, edgecolor=c, facecolor="none"))
        tag = classes[int(label)] + (f" {scores[i]:.2f}" if scores is not None else "")
        ax.text(x1, max(0, y1 - 2), tag, color="white", fontsize=7,
                bbox=dict(facecolor=c, edgecolor="none", pad=0.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="assets/demo_predictions.png")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    device = get_device("auto")
    model = build_model(cfg, len(classes)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    _, val_ds, _ = build_datasets(cfg)

    fig, axes = plt.subplots(args.num, 2, figsize=(6, 3 * args.num))
    if args.num == 1:
        axes = axes[None, :]
    for r in range(args.num):
        img, target = val_ds[r]
        with torch.no_grad():
            dec = model.decode(model(img[None].to(device)))
        pred = postprocess(dec, len(classes), args.conf, args.nms)[0]
        applied = ", ".join(target["applied"]) or "none"
        _draw(axes[r, 0], img, target["boxes"], target["labels"], classes,
              f"GT | degraded: {applied}")
        _draw(axes[r, 1], img, pred["boxes"], pred["labels"], classes,
              f"PRED ({len(pred['labels'])} det)", scores=pred["scores"])

    fig.suptitle(f"SpectraDet predictions  ({ckpt.get('params', 0)/1e6:.2f}M params)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
