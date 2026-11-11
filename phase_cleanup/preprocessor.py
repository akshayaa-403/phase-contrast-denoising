"""Classical OpenCV pre-processing for phase-contrast microscopy.

No deep learning here: every step is a few lines of classical image processing
a biomedical engineer can inspect and tune. ``Preprocessor.process`` runs them
in order, and is the one definition of "the classical path" that inference,
training and the report all share:

1. **Polarity** (optional): positive phase contrast shows cells *darker* than
   the background. Inverting makes cells bright, which is what thresholding and
   the training targets expect.
2. **Background flattening**: uneven illumination is a slow, smooth change in
   brightness across the field. It breaks a single global threshold (one side
   of the image is brighter than cells on the other side), and the DoG step
   cannot remove it, because a DoG responds to a linear ramp with ~0. We
   estimate the background with a grey-level *opening* (removes every bright
   object narrower than the structuring element, leaving the background), smooth
   it, and subtract it -- the rolling-ball idea from ImageJ.
3. **Halo suppression (DoG)**: subtract a scaled Difference-of-Gaussians band.
4. **CLAHE**: local contrast enhancement.
5. **Normalisation** to [0, 1], by percentiles so a single hot pixel cannot set
   the scale for the whole image.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


class Preprocessor:
    """Pure-OpenCV halo suppression and contrast enhancement for grayscale images.

    Parameters mirror the ``preprocessor:`` section of the YAML config; see
    ``configs/default_config.yaml`` for what each one does and how to tune it.
    """

    def __init__(
        self,
        dog_sigma1: float = 1.0,
        dog_sigma2: float = 1.6,
        dog_alpha: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        clahe_dtype: str = "uint8",
        invert: bool = False,
        background_radius: int = 0,
        norm_percentiles: Optional[Sequence[float]] = None,
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
        # "uint8" (256 histogram bins) is the default because the clip limit
        # only works there. OpenCV turns it into a per-bin ceiling of
        # max(int(clip * tileArea / bins), 1); with 65536 bins ("uint16") that
        # is 1 for any tile under ~32k pixels, so clip does nothing.
        if clahe_dtype not in ("uint16", "uint8"):
            raise ValueError(f"clahe_dtype must be 'uint16' or 'uint8', got {clahe_dtype!r}.")
        self.clahe_dtype = clahe_dtype
        self.invert = bool(invert)
        self.background_radius = int(background_radius)
        if self.background_radius < 0:
            raise ValueError("background_radius must be >= 0 (0 disables flattening).")
        if norm_percentiles is not None:
            lo, hi = (float(norm_percentiles[0]), float(norm_percentiles[1]))
            if not 0.0 <= lo < hi <= 100.0:
                raise ValueError(f"norm_percentiles must satisfy 0 <= low < high <= 100, got {norm_percentiles}.")
            self.norm_percentiles: Optional[Tuple[float, float]] = (lo, hi)
        else:
            self.norm_percentiles = None

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]], **overrides: Any) -> Preprocessor:
        """Build from the ``preprocessor:`` section of the YAML config.

        ``overrides`` (e.g. ``dog_alpha=0.8``) replace individual settings, which
        is how the Gradio demo applies its sliders without touching shared state.
        """
        cfg = dict(cfg or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        pct = cfg.get("norm_percentiles")
        return cls(
            dog_sigma1=cfg.get("dog_sigma1", 1.0),
            dog_sigma2=cfg.get("dog_sigma2", 1.6),
            dog_alpha=cfg.get("dog_alpha", 0.5),
            clahe_clip_limit=cfg.get("clahe_clip_limit", 2.0),
            clahe_grid_size=tuple(cfg.get("clahe_grid_size", (8, 8))),
            clahe_dtype=cfg.get("clahe_dtype", "uint8"),
            invert=cfg.get("invert", False),
            background_radius=cfg.get("background_radius", 0),
            norm_percentiles=tuple(pct) if pct else None,
        )

    def to_config(self) -> dict:
        """The settings as a plain dict, in the YAML config's vocabulary."""
        return {
            "dog_sigma1": self.dog_sigma1,
            "dog_sigma2": self.dog_sigma2,
            "dog_alpha": self.dog_alpha,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": list(self.clahe_grid_size),
            "clahe_dtype": self.clahe_dtype,
            "invert": self.invert,
            "background_radius": self.background_radius,
            "norm_percentiles": list(self.norm_percentiles) if self.norm_percentiles else None,
        }

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _to_float32(image: np.ndarray) -> np.ndarray:
        """Return a contiguous float32 single-channel copy of image."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(image, dtype=np.float32)

    def match_polarity(self, image: np.ndarray) -> np.ndarray:
        """Return the image with cells bright: ``1 - image`` if ``invert`` is set.

        Expects an image in [0, 1]. This is also the fair "raw" baseline to
        score against a bright-cell target.
        """
        img = self._to_float32(image)
        return (1.0 - img) if self.invert else img

    def flatten_background(self, image: np.ndarray, radius: Optional[int] = None) -> np.ndarray:
        """Subtract a smooth estimate of the illumination background.

        The background is a grey-level opening with a ``(2r+1)``-pixel square
        (removes bright objects narrower than that, i.e. the cells), smoothed
        with a Gaussian of sigma ``r / 2`` so the square leaves no blocks. Pick
        ``r`` a little larger than the largest cell radius. ``r = 0`` returns
        the image unchanged.
        """
        r = self.background_radius if radius is None else int(radius)
        img = self._to_float32(image)
        if r <= 0:
            return img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        background = cv2.GaussianBlur(opened, (0, 0), sigmaX=r / 2.0, sigmaY=r / 2.0)
        return img - background

    def halo_suppression_dog(
        self,
        image: np.ndarray,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Suppress the halo by subtracting a scaled Difference-of-Gaussians band.

        ``enhanced = image - alpha * (blur(sigma1) - blur(sigma2))``. The band
        between the two blur scales holds the halo ring; ``alpha`` sets how much
        of it to remove (too high starts eroding cell edges).
        """
        s1 = self.dog_sigma1 if sigma1 is None else float(sigma1)
        s2 = self.dog_sigma2 if sigma2 is None else float(sigma2)
        a = self.dog_alpha if alpha is None else float(alpha)
        if s2 <= s1:
            raise ValueError(f"sigma2 ({s2}) must be greater than sigma1 ({s1}).")

        img = self._to_float32(image)
        blur_small = cv2.GaussianBlur(img, (0, 0), sigmaX=s1, sigmaY=s1)
        blur_large = cv2.GaussianBlur(img, (0, 0), sigmaX=s2, sigmaY=s2)
        dog_response = blur_small - blur_large
        return img - a * dog_response

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Locally enhance contrast with CLAHE; returns float32 in the input's range."""
        clip = self.clahe_clip_limit if clip_limit is None else float(clip_limit)
        grid = self.clahe_grid_size if grid_size is None else (int(grid_size[0]), int(grid_size[1]))

        img = self._to_float32(image)
        lo, hi = float(img.min()), float(img.max())
        rng = hi - lo
        if rng < 1e-8:
            return img.copy()

        # CLAHE requires an integer image.
        if self.clahe_dtype == "uint8":
            dtype, top = np.uint8, 255.0
        else:
            dtype, top = np.uint16, 65535.0
        img_int = ((img - lo) / rng * top).astype(dtype)

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        eq_int = clahe.apply(img_int)
        return eq_int.astype(np.float32) / top * rng + lo

    @staticmethod
    def normalize_01(image: np.ndarray) -> np.ndarray:
        """Min-max scale an image to [0, 1] (constant image -> all zeros)."""
        img = np.ascontiguousarray(image, dtype=np.float32)
        lo, hi = float(img.min()), float(img.max())
        rng = hi - lo
        if rng < 1e-8:
            return np.zeros_like(img)
        return (img - lo) / rng

    @staticmethod
    def normalize_percentile(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
        """Scale so the ``low``/``high`` percentiles map to 0/1, then clip.

        Robust to outliers: a hot pixel or a speck of dust no longer squeezes
        the rest of the image into a narrow grey band (the CSBDeep/CARE default
        is 1 / 99.8).
        """
        img = np.ascontiguousarray(image, dtype=np.float32)
        lo, hi = np.percentile(img, [low, high])
        rng = float(hi) - float(lo)
        if rng < 1e-8:
            return Preprocessor.normalize_01(img)
        return np.clip((img - np.float32(lo)) / np.float32(rng), 0.0, 1.0)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """The configured normalisation: percentile if ``norm_percentiles`` is set, else min-max."""
        if self.norm_percentiles is None:
            return self.normalize_01(image)
        return self.normalize_percentile(image, *self.norm_percentiles)

    # ------------------------------------------------------------------ pipeline
    def process(self, image01: np.ndarray, return_stages: bool = False):
        """Run the whole classical path on an image in [0, 1].

        Returns the [0, 1] result, or with ``return_stages=True`` a dict of
        every intermediate (``polarity``, ``flat``, ``dog``, ``clahe``, ``out``).
        """
        polarity = self.match_polarity(image01)
        flat = self.flatten_background(polarity)
        dog = self.halo_suppression_dog(flat)
        clahe = self.apply_clahe(dog)
        out = self.normalize(clahe)
        if return_stages:
            return {"polarity": polarity, "flat": flat, "dog": dog, "clahe": clahe, "out": out}
        return out

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
            return (self.normalize_01(arr) * 255.0).astype(np.uint8)

        panels = [_to_display(original), _to_display(dog_result), _to_display(clahe_result)]
        h = max(p.shape[0] for p in panels)
        gutter = np.full((h, 4), 255, dtype=np.uint8)
        row = []
        for i, p in enumerate(panels):
            if p.shape[0] != h:
                p = cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            row.append(p)
            if i < len(panels) - 1:
                row.append(gutter)
        out_path = os.path.join(out_dir, f"debug_overlay_step{step:04d}.png")
        cv2.imwrite(out_path, np.hstack(row))
        return out_path
