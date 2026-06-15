"""Plot training curves (loss + mAP vs epoch) from runs/*/history.json.

    python scripts/plot_curves.py                         # auto-discover runs/
    python scripts/plot_curves.py --runs runs/lite runs/baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"lite": "#3cb44b", "baseline": "#e6194B"}


def _load(run_dir: Path):
    f = run_dir / "history.json"
    if not f.exists():
        return None
    hist = json.loads(f.read_text())
    name = run_dir.name
    epochs = [h["epoch"] for h in hist]
    loss = [h["loss"] for h in hist]
    map_e = [(h["epoch"], h["map"]) for h in hist if "map" in h]
    map50_e = [(h["epoch"], h["map50"]) for h in hist if "map50" in h]
    return name, epochs, loss, map_e, map50_e


def make_curves(run_dirs, out="assets/training_curves.png") -> bool:
    """Render loss + mAP curves for the given run dirs. Returns True if anything plotted."""
    runs = [r for r in (_load(Path(d)) for d in run_dirs) if r]
    if not runs:
        print("no history.json found — train a model first.")
        return False

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, epochs, loss, map_e, map50_e in runs:
        color = COLORS.get(name, None)
        ax[0].plot(epochs, loss, marker="o", ms=3, label=name, color=color)
        if map_e:
            xs, ys = zip(*map_e)
            ax[1].plot(xs, ys, marker="o", ms=3, label=f"{name} mAP@[.5:.95]", color=color)
        if map50_e:
            xs, ys = zip(*map50_e)
            ax[1].plot(xs, ys, marker="s", ms=3, ls="--", label=f"{name} mAP@.50", color=color, alpha=0.6)

    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].set_title("Validation mAP")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mAP"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_ylim(0, 1)
    fig.suptitle("SpectraDet training curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved -> {out.resolve()}  ({len(runs)} run(s))")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="assets/training_curves.png")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(p.parent for p in Path("runs").glob("*/history.json")) if Path("runs").exists() else []

    make_curves(run_dirs, args.out)


if __name__ == "__main__":
    main()
