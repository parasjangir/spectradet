"""Evaluate a trained checkpoint: mAP@[.5:.95], mAP50, per-class AP50.

    python scripts/evaluate.py --ckpt runs/lite/best.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from spectradet.data.dataset import build_datasets, detection_collate  # noqa: E402
from spectradet.engine.eval import evaluate  # noqa: E402
from spectradet.models.detector import build_model  # noqa: E402
from spectradet.utils.seed import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.6)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    device = get_device("auto")
    model = build_model(cfg, len(classes)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    _, val_ds, _ = build_datasets(cfg)
    loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=detection_collate)
    m = evaluate(model, loader, device, len(classes), args.conf, args.nms)

    print(f"\nmodel={cfg['name']}  params={ckpt.get('params', 0)/1e6:.3f}M")
    print(f"mAP@[.5:.95] = {m['map']:.4f}")
    print(f"mAP@.50      = {m['map50']:.4f}")
    print(f"mAP@.75      = {m['map75']:.4f}")
    print("\nper-class AP@.50:")
    for name, ap in zip(classes, m["per_class_ap50"]):
        print(f"  {name:12} {ap:.3f}")


if __name__ == "__main__":
    main()
