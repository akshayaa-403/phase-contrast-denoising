"""Orchestration layer: a single pipeline class with a CV-only / Hybrid toggle."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import yaml

from .preprocessor import Preprocessor
from .residual_unet import ResidualUNet


def load_yaml(path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class CleanupPipeline:
    """Halo-suppression pipeline with a strict CV-only / Hybrid toggle."""

    VALID_MODES = ("cv_only", "hybrid")

    def __init__(
        self,
        config_path: str,
        model_weights: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = load_yaml(config_path)
        pcfg = self.config.get("preprocessor", {})

        self.cv_ops = Preprocessor(
            dog_sigma1=pcfg.get("dog_sigma1", 1.0),
            dog_sigma2=pcfg.get("dog_sigma2", 1.6),
            dog_alpha=pcfg.get("dog_alpha", 0.5),
            clahe_clip_limit=pcfg.get("clahe_clip_limit", 2.0),
            clahe_grid_size=tuple(pcfg.get("clahe_grid_size", (8, 8))),
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        mcfg = self.config.get("model", {})
        self.model = ResidualUNet(
            in_channels=mcfg.get("in_channels", 1),
            out_channels=mcfg.get("out_channels", 1),
            base_channels=mcfg.get("base_channels", 64),
        ).to(self.device)
        self.weights_loaded = False
        if model_weights:
            self.load_weights(model_weights)

    def load_weights(self, model_weights: str) -> None:
        """Load a checkpoint (raw state_dict or trainer wrapper) into the model."""
        if not os.path.isfile(model_weights):
            raise FileNotFoundError(f"Model weights not found: {model_weights}")
        # weights_only=True refuses to unpickle arbitrary objects, so a tampered
        # checkpoint cannot execute code on load. Our checkpoints only hold
        # tensors + primitives, so this is always sufficient.
        ckpt = torch.load(model_weights, map_location=self.device, weights_only=True)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        self.weights_loaded = True
        print(f"[pipeline] loaded weights from {model_weights}")

    def _cv_baseline(self, image: np.ndarray) -> np.ndarray:
        """Run DoG halo suppression then CLAHE, returning a float32 image in [0, 1]."""
        img = np.asarray(image, dtype=np.float32)
        img = self.cv_ops.halo_suppression_dog(img)
        img = self.cv_ops.apply_clahe(img)
        img = self.cv_ops.normalize_01(img)
        return img

    def run(self, image: np.ndarray, mode: str = "hybrid") -> np.ndarray:
        """Clean a single grayscale image; returns float32 in [0, 1], same H,W."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}.")

        img = self._cv_baseline(image)

        if mode == "cv_only":
            return img

        if not self.weights_loaded:
            raise RuntimeError(
                "mode='hybrid' requires model weights. Pass model_weights= to "
                "the constructor or call load_weights()."
            )

        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
            residual = self.model.predict_residual(tensor)
            cleaned = tensor - residual
            cleaned = torch.clamp(cleaned, 0.0, 1.0)
        return cleaned.squeeze().cpu().numpy().astype(np.float32)

    def run_staged(self, image: np.ndarray) -> dict:
        """Return every intermediate stage (raw, dog, clahe, cv_only, hybrid) for reports."""
        raw = self.cv_ops.normalize_01(np.asarray(image, dtype=np.float32))
        dog = self.cv_ops.halo_suppression_dog(np.asarray(image, dtype=np.float32))
        clahe = self.cv_ops.apply_clahe(dog)
        cv_only = self.cv_ops.normalize_01(clahe)

        stages = {
            "raw": raw,
            "dog": self.cv_ops.normalize_01(dog),
            "clahe": cv_only,
            "cv_only": cv_only,
        }
        if self.weights_loaded:
            stages["hybrid"] = self.run(image, mode="hybrid")
        return stages
