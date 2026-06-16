"""Run a trained SpectraDet checkpoint on YOUR OWN image(s).

    # single image
    python scripts/predict.py --ckpt runs/lite/best.pt --source path/to/image.jpg

    # a whole folder (or glob)
    python scripts/predict.py --ckpt runs/lite/best.pt --source my_images/ --conf 0.3

Annotated copies are written to --out (default runs/predict/), and detections are
printed to the console.

NOTE: this checkpoint only knows the classes it was TRAINED on. The default
synthetic model knows 6 shapes (rectangle/circle/triangle/ellipse/star/pentagon),
so it will NOT detect real-world objects (cats, cars, ...). To detect real objects,
train on a real dataset via the VOC loader (configs: data.name = voc).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from spectradet.engine.infer import postprocess  # noqa: E402
from spectradet.models.detector import build_model  # noqa: E402
from spectradet.utils.seed import get_device  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(230, 25, 75), (60, 180, 75), (67, 99, 216),
          (245, 130, 49), (145, 30, 180), (66, 212, 244)]


def gather_images(source: str):
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
    if p.is_file():
        return [p]
    matches = sorted(Path().glob(source))  # treat as glob
    return [m for m in matches if m.suffix.lower() in IMG_EXTS]


def draw(img: Image.Image, pred, classes):
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        c = COLORS[int(label) % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = f"{classes[int(label)]} {float(score):.2f}"
        d.text((x1 + 3, max(0, y1 - 12)), tag, fill=c, font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="image file, folder, or glob")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="runs/predict")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    img_size = cfg["data"].get("img_size", 256)
    device = get_device("auto")
    model = build_model(cfg, len(classes)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    files = gather_images(args.source)
    if not files:
        print(f"no images found at '{args.source}'")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={cfg['name']} | classes={classes}")
    print(f"running on {len(files)} image(s) -> {out_dir}/\n")

    for f in files:
        im = Image.open(f).convert("RGB")
        ow, oh = im.size
        arr = np.asarray(im.resize((img_size, img_size), Image.BILINEAR))
        tens = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            dec = model.decode(model(tens.to(device)))
        pred = postprocess(dec, len(classes), args.conf, args.nms)[0]
        pred = {k: v.cpu() for k, v in pred.items()}  # back to CPU for scaling + drawing

        # scale boxes from model space back to the original image resolution
        sx, sy = ow / img_size, oh / img_size
        if pred["boxes"].numel():
            pred["boxes"] = pred["boxes"] * torch.tensor([sx, sy, sx, sy])

        out_path = out_dir / f"{f.stem}_pred.png"
        draw(im.copy(), pred, classes).save(out_path)

        dets = ", ".join(
            f"{classes[int(l)]}:{float(s):.2f}" for l, s in zip(pred["labels"], pred["scores"])
        ) or "(nothing above threshold)"
        print(f"  {f.name:30} -> {len(pred['labels'])} det  [{dets}]  saved {out_path.name}")


if __name__ == "__main__":
    main()
