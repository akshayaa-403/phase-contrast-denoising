"""Classical OpenCV pre-processing (halo suppression + contrast) for phase-contrast microscopy."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


class Preprocessor:
    """Pure-OpenCV halo suppression and contrast enhancement for grayscale images."""

    def __init__(
        self,
        dog_sigma1: float = 1.0,
        dog_sigma2: float = 1.6,
        dog_alpha: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        clahe_dtype: str = "uint16",
    ) -> None:
        if dog_sigma2 <= dog_sigma1:
            raise ValueError(
                f"dog_sigma2 ({dog_sigma2}) must be strictly greater than "
                f"dog_sigma1 ({dog_sigma1}); the larger sigma must capture the "
                "broad halo."
            )
        self.dog_sigma1 = float(dog_sigma1)
        self.dog_sigma2 = float(dog_sigma2)
        self.dog_alpha = float(dog_alpha)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = (int(clahe_grid_size[0]), int(clahe_grid_size[1]))
        # Opt-in only. "uint16" is the shipped behaviour and what the U-Net was
        # trained on. With 65536 histogram bins OpenCV's per-bin clip ceiling is
        # max(int(clip * tileArea / 65536), 1), which is 1 for small tiles, so
        # the clip limit does nothing there; "uint8" (256 bins) makes it bite.
        if clahe_dtype not in ("uint16", "uint8"):
            raise ValueError(f"clahe_dtype must be 'uint16' or 'uint8', got {clahe_dtype!r}.")
        self.clahe_dtype = clahe_dtype

    @staticmethod
    def _to_float32(image: np.ndarray) -> np.ndarray:
        """Return a contiguous float32 single-channel copy of image."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(image, dtype=np.float32)

    def halo_suppression_dog(
        self,
        image: np.ndarray,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Suppress the low-frequency halo via Difference-of-Gaussians."""
        s1 = self.dog_sigma1 if sigma1 is None else float(sigma1)
        s2 = self.dog_sigma2 if sigma2 is None else float(sigma2)
        a = self.dog_alpha if alpha is None else float(alpha)
        if s2 <= s1:
            raise ValueError(f"sigma2 ({s2}) must be greater than sigma1 ({s1}).")

        img = self._to_float32(image)

        blur_small = cv2.GaussianBlur(img, (0, 0), sigmaX=s1, sigmaY=s1)
        blur_large = cv2.GaussianBlur(img, (0, 0), sigmaX=s2, sigmaY=s2)

        dog_response = blur_small - blur_large
        enhanced = img - a * dog_response
        return enhanced

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Locally enhance contrast with CLAHE."""
        clip = self.clahe_clip_limit if clip_limit is None else float(clip_limit)
        grid = self.clahe_grid_size if grid_size is None else (
            int(grid_size[0]), int(grid_size[1])
        )

        img = self._to_float32(image)
        lo, hi = float(img.min()), float(img.max())
        rng = hi - lo

        if rng < 1e-8:
            return img.copy()

        # CLAHE requires an integer image; uint16 keeps more precision than uint8.
        if self.clahe_dtype == "uint8":
            dtype, top = np.uint8, 255.0
        else:
            dtype, top = np.uint16, 65535.0
        img_int = ((img - lo) / rng * top).astype(dtype)

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        eq_int = clahe.apply(img_int)

        eq = eq_int.astype(np.float32) / top * rng + lo
        return eq

    @staticmethod
    def normalize_01(image: np.ndarray) -> np.ndarray:
        """Min-max scale an image to [0, 1] (constant image -> all zeros)."""
        img = np.ascontiguousarray(image, dtype=np.float32)
        lo, hi = float(img.min()), float(img.max())
        rng = hi - lo
        if rng < 1e-8:
            return np.zeros_like(img)
        return (img - lo) / rng

    def save_debug_overlay(
        self,
        original: np.ndarray,
        dog_result: np.ndarray,
        clahe_result: np.ndarray,
        step: int,
        out_dir: str = "data/intermediate",
    ) -> str:
        """Save a side-by-side [original | DoG | CLAHE] panel; return the PNG path."""
        os.makedirs(out_dir, exist_ok=True)

        def _to_display(arr: np.ndarray) -> np.ndarray:
            disp = self.normalize_01(arr) * 255.0
            return disp.astype(np.uint8)

        panels = [_to_display(original), _to_display(dog_result), _to_display(clahe_result)]

        h = max(p.shape[0] for p in panels)
        gutter = np.full((h, 4), 255, dtype=np.uint8)
        row = []
        for i, p in enumerate(panels):
            if p.shape[0] != h:
                p = cv2.copyMakeBorder(
                    p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0
                )
            row.append(p)
            if i < len(panels) - 1:
                row.append(gutter)
        overlay = np.hstack(row)

        out_path = os.path.join(out_dir, f"debug_overlay_step{step:04d}.png")
        cv2.imwrite(out_path, overlay)
        return out_path


if __name__ == "__main__":
    dummy = np.random.rand(256, 256).astype(np.float32)
    pre = Preprocessor()
    dog = pre.halo_suppression_dog(dummy)
    clahe = pre.apply_clahe(dog)
    norm = pre.normalize_01(clahe)
    assert dog.shape == dummy.shape
    assert clahe.shape == dummy.shape
    assert not np.isnan(clahe).any(), "CLAHE produced NaNs"
    assert 0.0 <= norm.min() and norm.max() <= 1.0
    path = pre.save_debug_overlay(dummy, dog, clahe, step=0)
    print(f"[preprocessor] smoke test passed. Debug overlay -> {path}")
