"""Benchmark params + inference latency for lite vs baseline (the efficiency claim).

    python scripts/benchmark.py --devices cpu mps --runs 60
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from spectradet.engine.benchmark import benchmark_latency  # noqa: E402
from spectradet.models.detector import build_model  # noqa: E402


def _best_map(out_dir: str):
    s = Path(out_dir) / "summary.json"
    if s.exists():
        return json.loads(s.read_text()).get("best_map")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/lite.yaml", "configs/baseline.yaml"])
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--out", type=str, default="assets/benchmark")
    args = ap.parse_args()

    rows = []
    for cfg_path in args.configs:
        cfg = yaml.safe_load(open(cfg_path))
        model = build_model(cfg, num_classes=len(cfg["data"].get("classes", range(6))))
        bm = _best_map(cfg["train"].get("out_dir", ""))
        for dev in args.devices:
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            r = benchmark_latency(model, args.img_size, dev, runs=args.runs)
            rows.append({
                "model": cfg["name"], "device": dev,
                "params_M": r["params_M"], "gflops": r["gflops"],
                "latency_ms": r["latency_ms_mean"],
                "fps": r["fps"], "mAP": round(bm, 4) if bm is not None else None,
                "fft": bool(cfg["model"]["fft_levels"]),
            })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # ratios (lite vs baseline) per device
    print("\n--- lite vs baseline ---")
    for dev in df["device"].unique():
        sub = df[df["device"] == dev].set_index("model")
        if {"spectradet_lite", "spectradet_baseline"} <= set(sub.index):
            lite, base = sub.loc["spectradet_lite"], sub.loc["spectradet_baseline"]
            print(f"[{dev}] params: {base.params_M/lite.params_M:.2f}x fewer | "
                  f"FLOPs: {base.gflops/lite.gflops:.1f}x fewer | "
                  f"latency: {base.latency_ms/lite.latency_ms:.2f}x faster "
                  f"({lite.latency_ms}ms vs {base.latency_ms}ms)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out.with_suffix(".json"), orient="records", indent=2)

    # bar chart of latency per device
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pivot_l = df.pivot_table(index="device", columns="model", values="latency_ms")
    pivot_p = df.drop_duplicates("model").set_index("model")["params_M"]
    pivot_l.plot(kind="bar", ax=ax[0], rot=0)
    ax[0].set_title("Inference latency (ms, lower=better)")
    ax[0].set_ylabel("ms / image")
    pivot_p.plot(kind="bar", ax=ax[1], rot=0, color=["#3cb44b", "#e6194B"])
    ax[1].set_title("Parameters (M, lower=better)")
    ax[1].set_ylabel("million params")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(f"\nsaved -> {out.with_suffix('.json')} , {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
