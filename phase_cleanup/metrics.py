"""Image-quality metrics, segmentation overlap, confidence intervals and compute cost.

PSNR / SSIM come in two flavours:

* **plain** -- the prediction is compared to the target as-is. Fair for a
  model trained to reproduce the target's intensities (the hybrid path).
* **affine-matched** -- the prediction is first fitted to the target with the
  least-squares ``a * x + b``, then scored. This is how CARE/CSBDeep evaluate
  restorations: a classical filter (or the raw image) cannot know the target's
  absolute grey levels, so an offset or contrast stretch should not be counted
  as error. Without it the raw input scores ~1 dB against a dark-background
  target simply because its background is grey.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity as _sk_ssim

log = logging.getLogger(__name__)


def _match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cast both to float32 and check the shapes agree.

    Inputs are *not* clipped: a prediction that overshoots [0, 1] is wrong by
    that much and the metric should say so.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for metric: {a.shape} vs {b.shape}")
    return a, b


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (data range 1), higher is better."""
    pred, target = _match(pred, target)
    if np.array_equal(pred, target):
        return float("inf")
    return float(_sk_psnr(target, pred, data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Structural Similarity Index (data range 1), higher is better."""
    pred, target = _match(pred, target)
    # win_size must be odd and <= the smallest image dimension.
    min_side = min(pred.shape)
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win = max(3, win)
    return float(_sk_ssim(target, pred, data_range=1.0, win_size=win))


def affine_match(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``a * pred + b`` with ``a, b`` minimising the squared error to ``target``."""
    pred, target = _match(pred, target)
    x = pred.astype(np.float64).ravel()
    y = target.astype(np.float64).ravel()
    var = x.var()
    if var < 1e-12:
        return np.full_like(pred, float(y.mean()))
    a = ((x - x.mean()) * (y - y.mean())).mean() / var
    b = y.mean() - a * x.mean()
    return (a * pred + b).astype(np.float32)


def psnr_affine(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR after fitting ``pred`` to ``target`` with ``a * x + b`` (see module docstring)."""
    return psnr(affine_match(pred, target), target)


def ssim_affine(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM after fitting ``pred`` to ``target`` with ``a * x + b``."""
    return ssim(affine_match(pred, target), target)


def otsu_mask(image01: np.ndarray) -> np.ndarray:
    """Foreground mask from Otsu's threshold on the 8-bit version of an image."""
    u8 = (np.clip(image01, 0.0, 1.0) * 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks (1.0 when both are empty)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    return float(2.0 * int((a & b).sum()) / denom) if denom else 1.0


def otsu_dice(cleaned: np.ndarray, clean_gt: np.ndarray) -> float:
    """Dice between the Otsu foreground masks of ``cleaned`` and the clean target.

    Dice (not pixel accuracy) because background dominates the frame; Dice
    reflects how well the cells themselves are recovered.
    """
    return dice(otsu_mask(cleaned), otsu_mask(clean_gt))


def otsu_dice_vs_mask(cleaned: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice between the Otsu mask of ``cleaned`` and an annotated foreground mask."""
    pred = cleaned > threshold_otsu(np.asarray(cleaned, dtype=np.float32))
    return dice(pred, np.asarray(gt_mask) > 0)


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return ``(statistic, low, high)`` with a percentile-bootstrap (1 - alpha) CI.

    Non-finite values are dropped first. Returns NaNs for an empty input.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    boots = np.apply_along_axis(stat, 1, samples)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(arr)), float(lo), float(hi)


def compute_flops(model, input_size: Tuple[int, int, int, int] = (1, 1, 512, 512)):
    """Return ``(flops, params)`` for ``model`` on a dummy input.

    ``thop`` counts multiply-accumulates (MACs); one MAC is two floating-point
    operations, so the MAC count is doubled to report true FLOPs. The dummy
    tensor is created on the model's own device (a CPU tensor fed to a CUDA
    model used to crash the report on GPU). Returns ``(nan, nan)`` if
    ``thop`` is not installed.
    """
    try:
        import copy

        import torch
        from thop import profile
    except Exception as exc:  # pragma: no cover - optional dependency path
        log.warning("FLOPs unavailable (%s).", exc)
        return float("nan"), float("nan")

    # thop registers hooks and buffers on the modules; profile a copy so the
    # live model (and its state_dict) is left untouched.
    probe = copy.deepcopy(model).eval()
    device = next(probe.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        macs, _ = profile(probe, inputs=(dummy,), verbose=False)
    # thop's own parameter count skips GroupNorm's affine weights; count directly.
    params = sum(p.numel() for p in model.parameters())
    return 2.0 * float(macs), float(params)


def cv_path_flops(preprocessor, height: int = 512, width: int = 512) -> float:
    """Estimate the floating-point operations of the classical path on an image.

    Counted per pixel from the separable filters OpenCV runs: each Gaussian blur
    is two 1-D passes of ``ksize`` multiply-adds, the opening is two 1-D min and
    two 1-D max passes, and CLAHE / normalisation are a handful of operations.
    An estimate, not a profile, but derived from the actual kernel sizes rather
    than guessed.
    """

    def ksize(sigma: float) -> int:
        return int(round(sigma * 8 + 1)) | 1

    per_pixel = 0.0
    p = preprocessor
    if p.invert:
        per_pixel += 1
    if p.background_radius > 0:
        side = 2 * p.background_radius + 1
        per_pixel += 4 * side  # erode + dilate, each separable into two 1-D passes
        per_pixel += 2 * 2 * ksize(p.background_radius / 2.0) + 1
    for sigma in (p.dog_sigma1, p.dog_sigma2):
        per_pixel += 2 * 2 * ksize(sigma)
    per_pixel += 3  # DoG subtract and scale
    per_pixel += 4 + 10  # quantise, LUT bilinear interpolation
    per_pixel += 3  # normalisation
    return per_pixel * height * width


def human_readable(n: float) -> str:
    """Format a large number with K/M/G/T suffixes."""
    if n != n:  # NaN
        return "n/a"
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000.0:
            return f"{n:.2f} {unit}".strip()
        n /= 1000.0
    return f"{n:.2f} P"
