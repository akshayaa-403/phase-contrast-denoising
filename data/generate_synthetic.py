"""Generate a synthetic paired dataset from a physical model of phase-contrast optics.

    python data/generate_synthetic.py                 # 100 train/val pairs + 30 test pairs

Why a physical model
--------------------
The first version of this script built its halo as a Difference of Gaussians,
the same operation the classical path uses to remove it, so the evaluation was
partly circular. This version follows the linear imaging model of

    Z. Yin, T. Kanade, M. Chen. "Understanding the phase contrast optics to
    restore artifact-free microscopy images for segmentation."
    Medical Image Analysis 16(5), 2012.

A thin specimen with optical path difference ``f(x)`` imaged through a phase
ring gives approximately

    g(x) = C - beta * (f(x) - (k * f)(x))

where ``k`` is the (normalised) point-spread function of the phase ring, an
*obscured Airy pattern*

    k(r) ∝ R * J1(2 pi R r) / r  -  (R - W) * J1(2 pi (R - W) r) / r

with ``R`` the outer radius and ``W`` the width of the ring in the back focal
plane. Both artifacts fall out of that one equation: just outside a cell
``f = 0`` but ``k * f > 0``, so the background brightens (the **halo**); deep
inside a large flat cell ``k * f ~ f``, so the cell fades back towards the
background (**shade-off**). Cells appear dark on a brighter background, as in
positive phase contrast. The target is ``f`` itself: the artifact-free image
Yin et al. restore, with cells bright on a dark background.

On top of the optics this adds what a real camera adds: multiplicative uneven
illumination (a tilted plane plus vignetting) and Poisson-Gaussian noise.

Noisy inputs are saved as 16-bit PNG (as a scientific camera would record
them) and targets as 8-bit PNG.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
from scipy.special import j1


def obscured_airy_kernel(ring_radius: float, ring_width: float, size: int) -> np.ndarray:
    """The phase-ring PSF of Yin et al. (2012), normalised to sum to 1.

    ``ring_radius`` and ``ring_width`` are in cycles per pixel (spatial
    frequency); the main lobe has radius ~0.61 / ring_radius pixels.
    """
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r = np.hypot(xx, yy)
    r[half, half] = 1e-9  # J1(a r) / r -> a / 2 as r -> 0
    inner = ring_radius - ring_width

    def jinc(a: float) -> np.ndarray:
        return a * j1(2 * np.pi * a * r) / r

    kernel = jinc(ring_radius) - jinc(inner)
    # Apodise: a real condenser annulus has finite width and partial coherence,
    # which damps the Airy side lobes. Without this the model draws concentric
    # rings that real phase-contrast images do not show.
    lobe = 0.61 / ring_radius
    kernel *= np.exp(-(r * r) / (2.0 * (1.2 * lobe) ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def draw_phase_map(size: int, rng: np.random.Generator) -> np.ndarray:
    """A field of cells as optical-path-difference, float32 in [0, 1].

    Each cell is a dome (thicker in the middle) with a denser nucleus and a few
    granules, so the target has texture to preserve, not just flat blobs.
    """
    phase = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    n_cells = int(rng.integers(6, 16))
    for _ in range(n_cells):
        cx, cy = rng.uniform(0, size, size=2)
        ax = rng.uniform(size / 30, size / 10)
        ay = ax * rng.uniform(0.5, 1.0)
        theta = rng.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        u = ((xx - cx) * c + (yy - cy) * s) / ax
        v = (-(xx - cx) * s + (yy - cy) * c) / ay
        rho2 = u * u + v * v
        dome = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
        height = rng.uniform(0.5, 0.9)
        cell = height * dome

        # Nucleus: a smaller, denser dome offset from the centre.
        nu = (u - rng.uniform(-0.25, 0.25)) / 0.45
        nv = (v - rng.uniform(-0.25, 0.25)) / 0.45
        nucleus = np.sqrt(np.clip(1.0 - (nu * nu + nv * nv), 0.0, None))
        cell = cell + 0.25 * height * nucleus

        # Granules inside the cytoplasm.
        for _ in range(int(rng.integers(2, 7))):
            gu, gv = rng.uniform(-0.6, 0.6, size=2)
            gr = rng.uniform(0.06, 0.14)
            g = np.exp(-((u - gu) ** 2 + (v - gv) ** 2) / (2 * gr * gr))
            cell = cell + 0.12 * height * g * (rho2 < 1.0)

        phase = np.maximum(phase, cell.astype(np.float32))
    phase = cv2.GaussianBlur(phase, (0, 0), sigmaX=0.8)
    return np.clip(phase, 0.0, 1.0)


def illumination_field(size: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative uneven illumination: a tilted plane plus off-centre vignetting."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    gx, gy = rng.uniform(-1, 1, size=2)
    tilt = rng.uniform(0.15, 0.45) * (gx * xx + gy * yy)
    vx, vy = rng.uniform(-0.3, 0.3, size=2)
    vignette = -rng.uniform(0.1, 0.4) * ((xx - vx) ** 2 + (yy - vy) ** 2)
    return (1.0 + tilt + vignette).astype(np.float32)


def phase_contrast_image(
    phase: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, dict]:
    """Image ``phase`` through the phase-contrast model and a noisy camera."""
    size = phase.shape[0]
    lobe = rng.uniform(2.5, 7.0)  # main-lobe radius in px, i.e. the halo width
    ring_radius = 0.61 / lobe
    ring_width = ring_radius * rng.uniform(0.25, 0.45)
    ksize = int(min(size - 1, 2 * round(6 * lobe) + 1))
    kernel = obscured_airy_kernel(ring_radius, ring_width, ksize)

    blurred = cv2.filter2D(phase, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    beta = rng.uniform(0.35, 0.6)
    background = 0.5
    optics = background - beta * (phase - blurred)

    illum = illumination_field(size, rng)
    signal = np.clip(optics * illum, 1e-4, None)

    photons = rng.uniform(300.0, 3000.0)  # expected photons at background level
    counts = rng.poisson(signal / background * photons).astype(np.float32)
    read_noise = rng.uniform(1.0, 6.0)
    counts += rng.normal(0.0, read_noise, size=counts.shape).astype(np.float32)
    image = counts / photons * background

    params = {"halo_px": lobe, "beta": beta, "photons": photons}
    return image.astype(np.float32), params


def _to_uint16(image: np.ndarray) -> np.ndarray:
    """Min-max scale to the full 16-bit range, as a camera's auto-scaling would."""
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.uint16)
    return np.round((image - lo) / (hi - lo) * 65535.0).astype(np.uint16)


def generate(num: int, size: int, out_dir: str, seed: int = 42) -> Tuple[str, str]:
    """Write ``num`` (noisy, clean) pairs under ``out_dir``; return (clean_dir, noisy_dir)."""
    clean_dir = os.path.join(out_dir, "ground_truth")
    noisy_dir = os.path.join(out_dir, "noisy_input")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(num):
        phase = draw_phase_map(size, rng)
        noisy, _ = phase_contrast_image(phase, rng)
        name = f"img_{i:04d}.png"
        cv2.imwrite(os.path.join(clean_dir, name), np.round(phase * 255.0).astype(np.uint8))
        cv2.imwrite(os.path.join(noisy_dir, name), _to_uint16(noisy))

    print(f"[generate_synthetic] wrote {num} pairs ({size}x{size}) -> {clean_dir} , {noisy_dir}")
    return clean_dir, noisy_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic paired dataset.")
    parser.add_argument("--num", type=int, default=100, help="train/val pairs")
    parser.add_argument("--num_test", type=int, default=30, help="held-out test pairs (0 to skip)")
    parser.add_argument("--size", type=int, default=256, help="image side length")
    parser.add_argument("--out_dir", type=str, default="data", help="base output dir")
    parser.add_argument("--seed", type=int, default=42, help="rng seed (test set uses seed + 1000)")
    args = parser.parse_args()
    generate(args.num, args.size, args.out_dir, args.seed)
    if args.num_test > 0:
        generate(args.num_test, args.size, os.path.join(args.out_dir, "test"), args.seed + 1000)


if __name__ == "__main__":
    main()
