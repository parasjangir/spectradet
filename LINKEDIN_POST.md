# SpectraDet — LinkedIn post

> Before posting:
> 1. Push the repo, then drop the GitHub link in the FIRST COMMENT (LinkedIn throttles
>    reach on posts with external links in the body).
> 2. Attach 2–3 images: `assets/sample_grid.png` (the degraded data), `assets/comparison.png`
>    (lite vs baseline), and `assets/demo_predictions.png` (detections through the blur).
> 3. The lite-vs-baseline mAP-gap line is left blank on purpose — fill it once the baseline
>    finishes training (`assets/results.json`). Don't quote a number you haven't measured.

---

📸 Every object detector demo uses crisp, perfect photos.

Real cameras don't. You get motion blur, sensor noise, JPEG mush, and low light — often
ALL at once. So I built a detector that's meant to work on exactly those garbage images.

Meet **SpectraDet** — a lightweight (~2.5M parameter) object detector that finds objects
in **multi-degraded** images, built completely from scratch in PyTorch.

What's inside 👇
🌐 A custom **FFT module** — blur and noise live in the frequency domain, so I gave the
   network a learnable per-frequency filter (a global receptive field a normal conv can't see).
🎯 **SimOTA** dynamic label assignment — smarter positive-sample selection than naive
   one-box-one-anchor matching.
📦 A custom **IoU-guided loss** — the confidence score is trained to reflect *how well* the
   box is localised, not just whether something is there.

The numbers (all measured, not guessed):
⚡ **3.1× faster** inference — 17ms vs 54ms per image
🪶 **4.3× fewer parameters** — 2.5M vs 10.7M
🧮 **21.7× fewer FLOPs**
🎚️ **mAP 0.642 / mAP@0.50 0.847** on multi-degraded scenes

What I'm proudest of: I built **everything** from scratch — IoU/CIoU, non-max suppression,
SimOTA, the FFT filter, even the mAP metric itself. No torchvision, no OpenCV. 16 passing
tests. I can defend every single line.

But the real lesson was a humbling one 😅
I assumed the smaller model would obviously be faster. It wasn't — at first it was *slower*.
Depthwise convolutions slash parameters and FLOPs but are memory-bound, so they only fly on
edge/mobile hardware with specialised kernels. **Params ≠ FLOPs ≠ latency.**
So instead of printing a number that looked good, I reported FLOPs (the honest, hardware-
agnostic compute metric) AND benchmarked against a *genuinely* heavyweight baseline — no
strawman. Doing the honest thing taught me more than a clean result would have.

🛠️ Python · PyTorch (trained on Apple MPS) · NumPy · SciPy · Streamlit
🎓 Built for CS776 (IIT Kanpur) under Prof. Priyanka Bagade

🔗 Code: link in the comments 👇

What's the messiest real-world image you've ever tried to run a model on? 👇

#ComputerVision #DeepLearning #PyTorch #ObjectDetection #MachineLearning #AI #Python #Portfolio
