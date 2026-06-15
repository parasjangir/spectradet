"""Visualise the data pipeline: clean scene vs. its multi-degraded version + boxes.

    python scripts/make_sample_grid.py --rows 4 --out assets/sample_grid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectradet.data.synthetic import SyntheticDegradedDataset  # noqa: E402

COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def draw(ax, img_tensor, target, classes, title):
    ax.imshow(img_tensor.permute(1, 2, 0).numpy())
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        c = COLORS[int(label) % len(COLORS)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.6, edgecolor=c, facecolor="none"))
        ax.text(x1, max(0, y1 - 2), classes[int(label)], color="white", fontsize=7,
                bbox=dict(facecolor=c, edgecolor="none", pad=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="assets/sample_grid.png")
    args = ap.parse_args()

    clean = SyntheticDegradedDataset(length=64, img_size=args.img_size, degrade=False, base_seed=42)
    degraded = SyntheticDegradedDataset(length=64, img_size=args.img_size, degrade=True, base_seed=42)
    classes = SyntheticDegradedDataset.CLASSES

    fig, axes = plt.subplots(args.rows, 2, figsize=(6, 3 * args.rows))
    if args.rows == 1:
        axes = axes[None, :]
    for r in range(args.rows):
        idx = r * 3 + 1
        ci, ct = clean[idx]
        di, dt = degraded[idx]
        applied = ", ".join(dt["applied"]) or "none"
        draw(axes[r, 0], ci, ct, classes, f"#{idx} clean ({len(ct['labels'])} objs)")
        draw(axes[r, 1], di, dt, classes, f"degraded: {applied}")

    fig.suptitle("SpectraDet — synthetic multi-degraded scenes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
