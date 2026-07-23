"""Generate a synthetic paired dataset (clean brightfield + haloed phase-contrast)."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def _draw_cells(size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random field of soft elliptical cells as a float32 mask in [0, 1]."""
    canvas = np.zeros((size, size), dtype=np.float32)
    n_cells = int(rng.integers(8, 22))
    for _ in range(n_cells):
        cx, cy = rng.integers(0, size, size=2)
        ax = int(rng.integers(size // 30, size // 10))
        ay = int(rng.integers(size // 30, size // 10))
        angle = float(rng.uniform(0, 180))
        intensity = float(rng.uniform(0.5, 1.0))
        cv2.ellipse(
            canvas, (int(cx), int(cy)), (ax, ay), angle, 0, 360,
            color=intensity, thickness=-1,
        )
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.5)
    return np.clip(canvas, 0.0, 1.0)


def _add_halo_and_noise(
    clean: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Corrupt a clean image with a simulated phase-contrast halo, shade-off, and noise."""
    size = clean.shape[0]

    halo_sigma_large = float(rng.uniform(size / 40.0, size / 20.0))
    halo_sigma_small = halo_sigma_large * 0.5
    blur_large = cv2.GaussianBlur(clean, (0, 0), sigmaX=halo_sigma_large)
    blur_small = cv2.GaussianBlur(clean, (0, 0), sigmaX=halo_sigma_small)
    halo = blur_small - blur_large
    halo_strength = float(rng.uniform(1.5, 3.0))
    haloed = clean + halo_strength * halo

    # Strong low-frequency gradient: this is what breaks a GLOBAL Otsu threshold,
    # so removing it is what restores downstream segmentability.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    shade_amp = float(rng.uniform(0.6, 1.0))
    shade = shade_amp * (gx * xx + gy * yy)
    haloed = haloed + shade

    noise_sigma = float(rng.uniform(0.01, 0.04))
    haloed = haloed + rng.normal(0.0, noise_sigma, size=haloed.shape).astype(np.float32)

    return haloed


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max normalise to uint8 for saving as PNG."""
    lo, hi = float(image.min()), float(image.max())
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - lo) / rng * 255.0).astype(np.uint8)


def generate(
    num: int,
    size: int,
    out_dir: str,
    seed: int = 42,
) -> Tuple[str, str]:
    """Generate num paired (clean, noisy) images; return (clean_dir, noisy_dir)."""
    clean_dir = os.path.join(out_dir, "ground_truth")
    noisy_dir = os.path.join(out_dir, "noisy_input")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(num):
        clean = _draw_cells(size, rng)
        noisy = _add_halo_and_noise(clean, rng)

        cv2.imwrite(os.path.join(clean_dir, f"img_{i:04d}.png"), _to_uint8(clean))
        cv2.imwrite(os.path.join(noisy_dir, f"img_{i:04d}.png"), _to_uint8(noisy))

    print(f"[generate_synthetic] wrote {num} pairs "
          f"({size}x{size}) -> {clean_dir} , {noisy_dir}")
    return clean_dir, noisy_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic paired dataset.")
    parser.add_argument("--num", type=int, default=100, help="number of pairs")
    parser.add_argument("--size", type=int, default=512, help="image side length")
    parser.add_argument("--out_dir", type=str, default="data", help="base output dir")
    parser.add_argument("--seed", type=int, default=42, help="rng seed")
    args = parser.parse_args()
    generate(args.num, args.size, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
