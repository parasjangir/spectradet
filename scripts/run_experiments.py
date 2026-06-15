"""Train lite + baseline under identical data settings, then assemble the
headline comparison (params / FLOPs / latency / mAP) into assets/.

    python scripts/run_experiments.py --epochs 18 --train-size 1000 --val-size 300
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
import yaml  # noqa: E402

from spectradet.engine.benchmark import benchmark_latency  # noqa: E402
from spectradet.engine.train import train  # noqa: E402
from spectradet.models.detector import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--train-size", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--bench-runs", type=int, default=60)
    args = ap.parse_args()

    results = []
    for cfg_path in ["configs/lite.yaml", "configs/baseline.yaml"]:
        cfg = yaml.safe_load(open(cfg_path))
        cfg["data"]["train_size"] = args.train_size
        cfg["data"]["val_size"] = args.val_size
        cfg["data"]["img_size"] = args.img_size
        overrides = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}

        print(f"\n########## training {cfg['name']} ##########")
        summary = train(cfg, overrides)

        model = build_model(cfg, len(summary["classes"]))
        bm = benchmark_latency(model, args.img_size, "cpu", runs=args.bench_runs)
        best = max((h.get("map", 0) for h in summary["history"]), default=0)
        best50 = max((h.get("map50", 0) for h in summary["history"]), default=0)
        results.append({
            "model": cfg["name"], "params_M": bm["params_M"], "gflops": bm["gflops"],
            "latency_ms_cpu": bm["latency_ms_mean"], "fps_cpu": bm["fps"],
            "mAP": round(best, 4), "mAP50": round(best50, 4),
            "fft": bool(cfg["model"]["fft_levels"]),
        })

    Path("assets").mkdir(exist_ok=True)
    Path("assets/results.json").write_text(json.dumps(results, indent=2))

    lite = next(r for r in results if r["model"] == "spectradet_lite")
    base = next(r for r in results if r["model"] == "spectradet_baseline")
    drop = (base["mAP"] - lite["mAP"]) * 100

    print("\n================ HEADLINE COMPARISON ================")
    hdr = f"{'model':22}{'params':>9}{'GFLOPs':>9}{'CPU ms':>9}{'mAP':>8}{'mAP50':>8}"
    print(hdr)
    for r in results:
        print(f"{r['model']:22}{r['params_M']:>8.2f}M{r['gflops']:>9.2f}"
              f"{r['latency_ms_cpu']:>9.1f}{r['mAP']:>8.3f}{r['mAP50']:>8.3f}")
    print("-" * len(hdr))
    print(f"lite vs baseline: {base['params_M']/lite['params_M']:.2f}x fewer params | "
          f"{base['gflops']/lite['gflops']:.1f}x fewer FLOPs | "
          f"{base['latency_ms_cpu']/lite['latency_ms_cpu']:.2f}x faster | "
          f"mAP change: {drop:+.2f} pts")

    # comparison figure
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    names = [r["model"].replace("spectradet_", "") for r in results]
    colors = ["#3cb44b", "#e6194B"]
    for a, key, title in [
        (ax[0], "params_M", "Params (M) ↓"),
        (ax[1], "gflops", "GFLOPs ↓"),
        (ax[2], "latency_ms_cpu", "CPU latency (ms) ↓"),
        (ax[3], "mAP", "mAP@[.5:.95] ↑"),
    ]:
        a.bar(names, [r[key] for r in results], color=colors)
        a.set_title(title)
    fig.suptitle("SpectraDet-Lite (FFT, depthwise) vs DrebNET-style baseline", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("assets/comparison.png", dpi=130)

    # training curves for both runs
    from plot_curves import make_curves
    make_curves(["runs/lite", "runs/baseline"], "assets/training_curves.png")
    print("\nsaved -> assets/results.json , assets/comparison.png , assets/training_curves.png")


if __name__ == "__main__":
    main()
