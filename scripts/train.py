"""Train SpectraDet.

    python scripts/train.py --config configs/lite.yaml
    python scripts/train.py --config configs/lite.yaml --epochs 6 --train-size 800 --out-dir runs/lite_quick
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from spectradet.engine.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--val-size", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.train_size is not None:
        cfg["data"]["train_size"] = args.train_size
    if args.val_size is not None:
        cfg["data"]["val_size"] = args.val_size
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size

    overrides = {}
    for k_cli, k_cfg in [("epochs", "epochs"), ("batch_size", "batch_size"),
                         ("device", "device"), ("lr", "lr"), ("out_dir", "out_dir")]:
        v = getattr(args, k_cli)
        if v is not None:
            overrides[k_cfg] = v

    summary = train(cfg, overrides)
    print("\n=== done ===")
    print(f"best mAP@[.5:.95] = {summary['best_map']:.4f}  ({summary['params']/1e6:.3f}M params)")


if __name__ == "__main__":
    main()
