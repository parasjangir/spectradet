"""SpectraDet interactive demo.

    streamlit run app/streamlit_app.py

Generate (or upload) an image, stack degradations on it, and watch the
FFT-enhanced lite detector localise objects through blur/noise/compression.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

from spectradet.data.degradations import (
    DefocusBlur, GaussianBlur, GaussianNoise, JpegArtifacts, LowLight, MotionBlur,
)
from spectradet.data.synthetic import SyntheticDegradedDataset
from spectradet.engine.infer import postprocess
from spectradet.models.detector import build_model

COLORS = [(230, 25, 75), (60, 180, 75), (67, 99, 216),
          (245, 130, 49), (145, 30, 180), (66, 212, 244)]
DEG_MAP = {
    "motion_blur": MotionBlur(), "defocus_blur": DefocusBlur(), "gaussian_blur": GaussianBlur(),
    "gaussian_noise": GaussianNoise(), "jpeg_artifacts": JpegArtifacts(), "low_light": LowLight(),
}


@st.cache_resource
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["cfg"], len(ckpt["classes"])).eval()
    model.load_state_dict(ckpt["model"])
    return model, ckpt["classes"], ckpt.get("params", 0)


def draw_boxes(arr, boxes, labels, classes, scores=None):
    img = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (b, l) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [float(v) for v in b]
        c = COLORS[int(l) % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = classes[int(l)] + (f" {scores[i]:.2f}" if scores is not None else "")
        d.text((x1 + 2, max(0, y1 - 11)), tag, fill=c, font=font)
    return img


def main():
    st.set_page_config(page_title="SpectraDet", layout="wide")
    st.title("🔍 SpectraDet — detecting objects in multi-degraded images")
    st.caption("FFT-enhanced, ~2.5M-param anchor-free detector with SimOTA + IoU-guided loss.")

    ckpts = [str(p) for p in Path("runs").glob("**/best.pt")] if Path("runs").exists() else []
    with st.sidebar:
        st.header("Model")
        ckpt = st.selectbox("checkpoint", ckpts) if ckpts else st.text_input(
            "checkpoint path", "runs/lite/best.pt")
        conf = st.slider("confidence threshold", 0.05, 0.9, 0.3, 0.05)
        nms = st.slider("NMS IoU", 0.1, 0.9, 0.6, 0.05)
        st.header("Degradations")
        chosen = st.multiselect("apply (in order)", list(DEG_MAP), default=["motion_blur", "gaussian_noise"])
        seed = st.number_input("scene seed", 0, 9999, 7)

    if not ckpt or not Path(ckpt).exists():
        st.warning("Train a model first:  `python scripts/train.py --config configs/lite.yaml`")
        return

    model, classes, n_params = load_model(ckpt)
    st.sidebar.success(f"loaded ({n_params/1e6:.2f}M params)")

    src = st.radio("image source", ["synthetic scene", "upload"], horizontal=True)
    if src == "upload":
        up = st.file_uploader("image", type=["jpg", "jpeg", "png"])
        if up is None:
            st.info("Upload an image to run detection.")
            return
        img = Image.open(up).convert("RGB").resize((256, 256))
        arr = np.array(img)
        gt = None
    else:
        ds = SyntheticDegradedDataset(length=1, img_size=256, degrade=False, base_seed=int(seed))
        t_img, target = ds[0]
        arr = (t_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = target

    # apply chosen degradations
    rng = np.random.default_rng(int(seed) + 1)
    degraded = arr.copy()
    for name in chosen:
        degraded = DEG_MAP[name](degraded, rng)

    # detect
    tens = torch.from_numpy(np.ascontiguousarray(degraded)).permute(2, 0, 1).float()[None] / 255.0
    with torch.no_grad():
        dec = model.decode(model(tens))
    pred = postprocess(dec, len(classes), conf, nms)[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("clean")
        st.image(draw_boxes(arr, gt["boxes"], gt["labels"], classes) if gt else Image.fromarray(arr),
                 use_container_width=True)
    with c2:
        st.subheader("degraded (model input)")
        st.image(Image.fromarray(degraded), use_container_width=True)
    with c3:
        st.subheader(f"detections ({len(pred['labels'])})")
        st.image(draw_boxes(degraded, pred["boxes"], pred["labels"], classes, pred["scores"]),
                 use_container_width=True)

    if len(pred["labels"]):
        st.write({classes[int(l)]: round(float(s), 3) for l, s in zip(pred["labels"], pred["scores"])})


if __name__ == "__main__":
    main()
