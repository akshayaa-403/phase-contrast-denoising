"""Unit + integration tests for the classical preprocessor."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from phase_cleanup.preprocessor import Preprocessor


def test_dog_preserves_shape_and_dtype():
    """DoG returns a float32 image of identical H,W."""
    pre = Preprocessor()
    img = np.random.rand(128, 96).astype(np.float32)
    out = pre.halo_suppression_dog(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_dog_handles_color_input_as_single_channel():
    """A 3-channel input is collapsed to one channel."""
    pre = Preprocessor()
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = pre.halo_suppression_dog(img)
    assert out.ndim == 2
    assert out.shape == (64, 64)


def test_sigma_ordering_is_enforced():
    """sigma2 <= sigma1 must raise (halo sigma must be the larger one)."""
    with pytest.raises(ValueError):
        Preprocessor(dog_sigma1=2.0, dog_sigma2=1.0)


def test_clahe_no_nans_and_shape():
    """CLAHE never emits NaNs and preserves shape."""
    pre = Preprocessor()
    img = np.random.rand(200, 200).astype(np.float32)
    out = pre.apply_clahe(img)
    assert out.shape == img.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_clahe_constant_image_is_safe():
    """A constant image must not divide by zero."""
    pre = Preprocessor()
    img = np.full((32, 32), 0.5, dtype=np.float32)
    out = pre.apply_clahe(img)
    assert out.shape == img.shape
    assert not np.isnan(out).any()


def test_normalize_01_range():
    """normalize_01 output must lie in [0, 1]."""
    pre = Preprocessor()
    img = np.random.rand(50, 50).astype(np.float32) * 1000 - 500
    out = pre.normalize_01(img)
    assert out.min() >= 0.0 and out.max() <= 1.0


def _make_circle_with_halo(size: int = 256):
    """A bright disc surrounded by a broad, low-frequency halo glow."""
    img = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    radius = size // 6
    cv2.circle(img, center, radius, color=1.0, thickness=-1)

    glow = cv2.GaussianBlur(img, (0, 0), sigmaX=size / 14.0)
    glow = np.clip(glow - img, 0.0, None)
    glow /= max(glow.max(), 1e-8)
    haloed = img + 1.2 * glow
    return haloed, center, radius


def _halo_peak(img: np.ndarray, center, radius) -> float:
    """Peak intensity in the annulus outside the disc, above far background."""
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    ring = (dist > radius * 1.1) & (dist < radius * 2.2)
    far_bg = dist > radius * 3.0
    return float(img[ring].max() - img[far_bg].mean())


def test_integration_halo_peak_drops_30_percent():
    """End-to-end: the DoG step cuts the halo peak outside the disc by >=30%."""
    haloed, center, radius = _make_circle_with_halo(256)
    # sigma2 matched to the halo scale (~size/14 ≈ 18px); alpha tuned to remove it.
    pre = Preprocessor(dog_sigma1=1.0, dog_sigma2=20.0, dog_alpha=1.3)

    before = _halo_peak(haloed, center, radius)
    cleaned = pre.halo_suppression_dog(haloed)
    after = _halo_peak(cleaned, center, radius)

    drop = (before - after) / max(abs(before), 1e-8)
    assert drop >= 0.30, (
        f"Halo peak outside the disc only dropped {drop*100:.1f}% (need >=30%)."
    )


def test_pipeline_cv_only_runs_end_to_end():
    """The full pipeline (cv_only) runs on a dummy image without weights."""
    from phase_cleanup.pipeline import CleanupPipeline
    cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "default_config.yaml",
    )
    pipe = CleanupPipeline(config_path=cfg)
    img = np.random.rand(128, 128).astype(np.float32)
    out = pipe.run(img, mode="cv_only")
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_flatten_background_removes_linear_ramp():
    """A DoG ignores a linear illumination ramp; flattening must remove it."""
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32) / 128
    ramp = 0.2 + 0.6 * xx + 0.1 * yy
    img = ramp.copy()
    cv2.circle(img, (64, 64), 6, color=float(img[64, 64] + 0.5), thickness=-1)
    flat = Preprocessor(background_radius=10).flatten_background(img)
    dist = np.hypot(xx * 128 - 64, yy * 128 - 64)
    interior = (dist > 20) & (xx > 0.15) & (xx < 0.85) & (yy > 0.15) & (yy < 0.85)
    assert np.ptp(flat[interior]) < 0.1 * np.ptp(ramp)
    assert flat[64, 64] > 0.4  # the cell survives


def test_flatten_disabled_is_identity():
    img = np.random.rand(32, 32).astype(np.float32)
    np.testing.assert_array_equal(Preprocessor(background_radius=0).flatten_background(img), img)


def test_percentile_normalisation_ignores_hot_pixel():
    """One saturated pixel must not squash the rest of the image (min-max would)."""
    img = np.random.rand(64, 64).astype(np.float32) * 0.1
    img[0, 0] = 100.0
    robust = Preprocessor.normalize_percentile(img, 1.0, 99.8)
    assert robust.max() <= 1.0 and robust.min() >= 0.0
    assert np.median(robust) > 0.3
    assert np.median(Preprocessor.normalize_01(img)) < 0.01


def test_invert_and_polarity():
    img = np.full((8, 8), 0.2, dtype=np.float32)
    np.testing.assert_allclose(Preprocessor(invert=True).match_polarity(img), 0.8)
    assert Preprocessor(invert=False).match_polarity(img)[0, 0] == pytest.approx(0.2)


def test_uint8_clahe_clip_limit_has_effect():
    """With 256 bins the clip limit changes the output; with 65536 it is inert."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 3)

    def run(clip, dtype):
        return Preprocessor(clahe_clip_limit=clip, clahe_dtype=dtype).apply_clahe(img)

    assert np.abs(run(1.0, "uint8") - run(6.0, "uint8")).max() > 1e-3
    np.testing.assert_array_equal(run(1.0, "uint16"), run(6.0, "uint16"))


def test_config_round_trip():
    pre = Preprocessor(invert=True, background_radius=7, norm_percentiles=(2, 98), dog_sigma2=5)
    assert Preprocessor.from_config(pre.to_config()).to_config() == pre.to_config()
    assert Preprocessor.from_config(pre.to_config(), dog_alpha=0.9).dog_alpha == 0.9


def test_process_stages_and_range():
    pre = Preprocessor(invert=True, background_radius=8, dog_sigma2=6, norm_percentiles=(1, 99.8))
    img = np.random.rand(96, 80).astype(np.float32)
    stages = pre.process(img, return_stages=True)
    assert set(stages) == {"polarity", "flat", "dog", "clahe", "out"}
    out = stages["out"]
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_array_equal(out, pre.process(img))
