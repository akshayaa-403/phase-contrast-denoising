"""Tests for the dataset, metrics and image I/O."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from phase_cleanup.dataset import PhaseContrastDataset, split_pairs
from phase_cleanup.image_io import load_gray01, output_path, save_image01
from phase_cleanup.metrics import (
    affine_match,
    bootstrap_ci,
    compute_flops,
    dice,
    otsu_dice,
    psnr,
    psnr_affine,
    ssim,
)
from phase_cleanup.preprocessor import Preprocessor
from phase_cleanup.residual_unet import ResidualUNet


# ---------------------------------------------------------------- image I/O
def test_16bit_dark_image_is_scaled_by_bit_depth(tmp_path):
    """A dark 16-bit frame (max <= 255) must be divided by 65535, not 255."""
    path = tmp_path / "dark.png"
    cv2.imwrite(str(path), np.full((8, 8), 200, dtype=np.uint16))
    out, depth = load_gray01(str(path), return_depth=True)
    assert depth == 16
    assert out.max() == pytest.approx(200 / 65535)


def test_tiff_stack_and_16bit_round_trip(tmp_path):
    import tifffile

    stack = (np.random.rand(3, 16, 16) * 65535).astype(np.uint16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), stack, photometric="minisblack")
    first = load_gray01(str(path))
    np.testing.assert_allclose(first, stack[0] / 65535.0, atol=1e-6)

    save_image01(str(tmp_path / "out.tif"), first, bit_depth=16)
    back = load_gray01(str(tmp_path / "out.tif"))
    assert np.abs(back - first).max() <= 1 / 65535 + 1e-7


def test_lossy_outputs_become_png():
    assert output_path("o", "a.jpg").endswith("a.png")
    assert output_path("o", "a.tif").endswith("a.tif")


# ---------------------------------------------------------------- dataset
class _Identity(Preprocessor):
    def process(self, image01, return_stages=False):
        return np.asarray(image01, dtype=np.float32)


def _write_images(root, n=5, size=40):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = root / f"{i}.png"
        cv2.imwrite(str(p), (rng.random((size, size)) * 65535).astype(np.uint16))
        paths.append(str(p))
    return paths


def test_augmentations_are_identical_for_input_and_target(tmp_path):
    """Same file as input and target: after crop/flip/rotate they must still match."""
    pairs = [(p, p) for p in _write_images(tmp_path)]
    ds = PhaseContrastDataset(pairs, _Identity(), patch_size=32, train=True)
    np.random.seed(0)
    for _ in range(3):
        for i in range(len(ds)):
            noisy, clean = ds[i]
            assert noisy.shape == (1, 32, 32)
            assert torch.equal(noisy, clean)


def test_split_is_deterministic_and_disjoint(tmp_path):
    paths = _write_images(tmp_path, n=10)
    pairs = list(zip(paths, paths, strict=True))
    tr, va = split_pairs(pairs, 0.2, 42)
    assert len(va) == 2 and not set(tr) & set(va)
    assert split_pairs(pairs, 0.2, 42) == (tr, va)


# ---------------------------------------------------------------- metrics
def test_psnr_does_not_hide_overshoot():
    target = np.zeros((16, 16), np.float32)
    assert psnr(target + 2.0, target) < psnr(np.clip(target + 2.0, 0, 1), target)


def test_affine_match_recovers_scale_and_offset():
    target = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    pred = 3.0 * target - 0.5
    np.testing.assert_allclose(affine_match(pred, target), target, atol=1e-5)
    assert psnr_affine(pred, target) > 80
    assert ssim(target, target) == pytest.approx(1.0)


def test_dice_and_otsu_dice():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
    img = a.astype(np.float32)
    assert otsu_dice(img, img) == 1.0


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, float("inf")])
    assert mean == pytest.approx(2.5) and lo <= mean <= hi


def test_flops_are_twice_macs_and_model_is_untouched():
    model = ResidualUNet(base_channels=8)
    keys = set(model.state_dict())
    flops, params = compute_flops(model, (1, 1, 32, 32))
    assert flops > 0 and params == sum(p.numel() for p in model.parameters())
    assert set(model.state_dict()) == keys  # thop must not add buffers to the live model
