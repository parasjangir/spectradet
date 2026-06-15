import numpy as np

from spectradet.data.degradations import (
    RandomMultiDegradation,
    default_degradations,
    disk_kernel,
    motion_blur_kernel,
)
from spectradet.data.synthetic import SyntheticDegradedDataset


def test_kernels_normalised():
    assert np.isclose(motion_blur_kernel(15, 30).sum(), 1.0, atol=1e-4)
    assert np.isclose(disk_kernel(5).sum(), 1.0, atol=1e-4)


def test_degradations_preserve_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for deg in default_degradations():
        out = deg(img, rng)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_multi_degradation_records_applied():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    md = RandomMultiDegradation(default_degradations(), min_n=2, max_n=2, p=1.0)
    out, applied = md(img, rng)
    assert out.shape == img.shape
    assert len(applied) == 2


def test_synthetic_dataset_deterministic():
    ds = SyntheticDegradedDataset(length=10, img_size=128, degrade=True, base_seed=7)
    img1, t1 = ds[3]
    img2, t2 = ds[3]
    assert np.allclose(img1.numpy(), img2.numpy())          # reproducible
    assert t1["boxes"].shape[0] >= 1                         # at least one object
    assert t1["boxes"].shape[1] == 4
    assert t1["labels"].max().item() < ds.num_classes
    # boxes within image bounds
    assert t1["boxes"].min().item() >= -1
    assert t1["boxes"].max().item() <= ds.img_size + 1
