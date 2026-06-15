"""Training loop: warmup + cosine LR, periodic mAP eval, best-checkpoint saving."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import build_datasets, detection_collate
from ..losses.loss import build_loss
from ..models.detector import build_model, count_parameters
from ..utils.seed import get_device, seed_everything
from .eval import evaluate


def _lr_at(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def train(cfg: Dict, overrides: Dict | None = None) -> Dict:
    overrides = overrides or {}
    tc = {**cfg["train"], **overrides}
    seed_everything(tc.get("seed", 0))
    device = get_device(tc.get("device", "auto"))

    train_ds, val_ds, classes = build_datasets(cfg)
    num_classes = len(classes)
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc.get("num_workers", 0), collate_fn=detection_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc.get("num_workers", 0), collate_fn=detection_collate,
    )

    model = build_model(cfg, num_classes).to(device)
    criterion = build_loss(cfg, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    epochs = tc["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(tc.get("warmup_epochs", 1) * steps_per_epoch)

    out_dir = Path(tc.get("out_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = count_parameters(model)
    print(f"model={cfg['name']} params={n_params/1e6:.3f}M device={device} "
          f"train={len(train_ds)} val={len(val_ds)} steps/epoch={steps_per_epoch}")

    history, best_map, gstep = [], -1.0, 0
    eval_interval = tc.get("eval_interval", 1)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for imgs, targets in pbar:
            lr = _lr_at(gstep, total_steps, warmup_steps, tc["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            imgs = imgs.to(device)
            out = criterion(model.decode(model(imgs)), targets)
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running += out["loss"].item()
            gstep += 1
            pbar.set_postfix(loss=f"{out['loss'].item():.3f}", lr=f"{lr:.1e}")

        avg_loss = running / max(1, steps_per_epoch)
        rec = {"epoch": epoch + 1, "loss": avg_loss, "time_s": round(time.time() - t0, 1)}

        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            metrics = evaluate(model, val_loader, device, num_classes,
                               tc.get("conf_thr", 0.05), tc.get("nms_thr", 0.6))
            rec.update({"map": metrics["map"], "map50": metrics["map50"], "map75": metrics["map75"]})
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                torch.save({"model": model.state_dict(), "cfg": cfg, "classes": classes,
                            "metrics": metrics, "params": n_params}, out_dir / "best.pt")
            print(f"  epoch {epoch+1}: loss={avg_loss:.3f} "
                  f"mAP={metrics['map']:.3f} mAP50={metrics['map50']:.3f} (best {best_map:.3f})")
        history.append(rec)
        # dump curve data every epoch (survives interruption, enables live plots)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({"model": model.state_dict(), "cfg": cfg, "classes": classes,
                "params": n_params}, out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {"name": cfg["name"], "params": n_params, "best_map": best_map,
               "history": history, "classes": classes}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
