"""Image-quality metrics (PSNR, SSIM) and a model complexity counter (FLOPs)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity as _sk_ssim


def _match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cast both to float32 in [0,1] and assert matching shapes."""
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    b = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for metric: {a.shape} vs {b.shape}")
    return a, b


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB), higher is better."""
    pred, target = _match(pred, target)
    if np.array_equal(pred, target):
        return float("inf")
    return float(_sk_psnr(target, pred, data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Structural Similarity Index (0..1), higher is better."""
    pred, target = _match(pred, target)
    # win_size must be odd and <= smallest image dimension.
    min_side = min(pred.shape)
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win = max(3, win)
    return float(_sk_ssim(target, pred, data_range=1.0, win_size=win))


def compute_flops(model, input_size: Tuple[int, int, int, int] = (1, 1, 512, 512)):
    """Return (flops, params) for model on a dummy input; (nan, nan) if thop/torch missing."""
    try:
        import torch
        from thop import profile
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"[metrics] FLOPs unavailable ({exc}).")
        return float("nan"), float("nan")

    model = model.eval()
    dummy = torch.randn(*input_size)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)
    return float(flops), float(params)


def human_readable(n: float) -> str:
    """Format a large number with K/M/G/T suffixes."""
    if n != n:  # NaN
        return "n/a"
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000.0:
            return f"{n:.2f} {unit}".strip()
        n /= 1000.0
    return f"{n:.2f} P"


if __name__ == "__main__":
    a = np.random.rand(64, 64).astype(np.float32)
    b = a + 0.02 * np.random.rand(64, 64).astype(np.float32)
    print(f"[metrics] PSNR={psnr(b, a):.2f} dB  SSIM={ssim(b, a):.4f}")
