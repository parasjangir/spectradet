"""Image degradations — the 'multi-degraded' core of the project.

Each degradation takes a uint8 HxWx3 RGB array and a numpy ``Generator`` and
returns a degraded uint8 array.  ``RandomMultiDegradation`` samples a *subset*
and applies them in sequence, so a single image can be simultaneously blurred,
noisy, compressed and dark — the realistic failure mode this project targets.

Design note (matches the report's motivation): purely synthetic *Gaussian* blur
is too clean to be representative of real cameras.  We therefore model the two
physically-grounded blurs separately — **motion** (linear streak from camera/
subject movement) and **defocus** (disk/bokeh from a mis-focused lens) — and let
them combine with noise / compression / illumination changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter, rotate

RNG = np.random.Generator


# --------------------------------------------------------------------------- #
# kernels + low-level ops
# --------------------------------------------------------------------------- #
def motion_blur_kernel(length: int, angle: float) -> np.ndarray:
    """A normalised linear streak kernel of given length, rotated by ``angle`` deg."""
    length = max(1, int(length))
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    k = rotate(k, angle, reshape=False, order=1)
    k = np.clip(k, 0.0, None)
    s = k.sum()
    return k / s if s > 0 else k


def disk_kernel(radius: int) -> np.ndarray:
    """A normalised filled-disk kernel — models defocus / bokeh blur."""
    radius = max(1, int(radius))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= radius * radius
    k = mask.astype(np.float32)
    return k / k.sum()


def _apply_kernel(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    out = np.empty_like(f)
    for c in range(f.shape[2]):
        out[..., c] = convolve(f[..., c], k, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# degradation classes
# --------------------------------------------------------------------------- #
class Degradation:
    name = "base"

    def __call__(self, img: np.ndarray, rng: RNG) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class MotionBlur(Degradation):
    length: Tuple[int, int] = (7, 21)
    angle: Tuple[float, float] = (0.0, 180.0)
    name: str = "motion_blur"

    def __call__(self, img, rng):
        L = int(rng.integers(self.length[0], self.length[1] + 1))
        a = float(rng.uniform(*self.angle))
        return _apply_kernel(img, motion_blur_kernel(L, a))


@dataclass
class DefocusBlur(Degradation):
    radius: Tuple[int, int] = (3, 9)
    name: str = "defocus_blur"

    def __call__(self, img, rng):
        r = int(rng.integers(self.radius[0], self.radius[1] + 1))
        return _apply_kernel(img, disk_kernel(r))


@dataclass
class GaussianBlur(Degradation):
    sigma: Tuple[float, float] = (1.0, 3.5)
    name: str = "gaussian_blur"

    def __call__(self, img, rng):
        s = float(rng.uniform(*self.sigma))
        out = gaussian_filter(img.astype(np.float32), sigma=(s, s, 0))
        return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class GaussianNoise(Degradation):
    sigma: Tuple[float, float] = (8.0, 30.0)
    name: str = "gaussian_noise"

    def __call__(self, img, rng):
        s = float(rng.uniform(*self.sigma))
        noise = rng.normal(0.0, s, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@dataclass
class JpegArtifacts(Degradation):
    quality: Tuple[int, int] = (10, 40)
    name: str = "jpeg_artifacts"

    def __call__(self, img, rng):
        q = int(rng.integers(self.quality[0], self.quality[1] + 1))
        buf = BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))


@dataclass
class LowLight(Degradation):
    gamma: Tuple[float, float] = (1.8, 3.0)
    gain: Tuple[float, float] = (0.3, 0.7)
    name: str = "low_light"

    def __call__(self, img, rng):
        g = float(rng.uniform(*self.gamma))
        k = float(rng.uniform(*self.gain))
        f = (img.astype(np.float32) / 255.0) ** g * k
        return np.clip(f * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def default_degradations() -> List[Degradation]:
    return [
        MotionBlur(),
        DefocusBlur(),
        GaussianBlur(),
        GaussianNoise(),
        JpegArtifacts(),
        LowLight(),
    ]


class RandomMultiDegradation:
    """Apply a random subset (size in ``[min_n, max_n]``) of degradations in sequence."""

    def __init__(
        self,
        degradations: List[Degradation] | None = None,
        min_n: int = 1,
        max_n: int = 3,
        p: float = 0.9,
    ):
        self.degradations = degradations if degradations is not None else default_degradations()
        self.min_n = min_n
        self.max_n = max_n
        self.p = p

    def __call__(self, img: np.ndarray, rng: RNG) -> Tuple[np.ndarray, List[str]]:
        if rng.random() > self.p:
            return img, []
        n = int(rng.integers(self.min_n, self.max_n + 1))
        n = min(n, len(self.degradations))
        idx = rng.choice(len(self.degradations), size=n, replace=False)
        applied: List[str] = []
        for i in sorted(idx.tolist()):
            deg = self.degradations[i]
            img = deg(img, rng)
            applied.append(deg.name)
        return img, applied
