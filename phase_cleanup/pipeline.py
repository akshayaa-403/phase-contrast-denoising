"""Orchestration layer: a single pipeline class with a CV-only / Hybrid toggle."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .preprocessor import Preprocessor
from .residual_unet import ResidualUNet

log = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    """Load a YAML config file into a plain dict (empty sections become {})."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for section in ("preprocessor", "model", "data", "training", "paths", "inference"):
        if cfg.get(section) is None:
            cfg[section] = {}
    return cfg


def model_from_config(mcfg: Optional[Dict[str, Any]]) -> ResidualUNet:
    """Build a ResidualUNet from the ``model:`` config section."""
    mcfg = mcfg or {}
    return ResidualUNet(
        in_channels=int(mcfg.get("in_channels", 1)),
        out_channels=int(mcfg.get("out_channels", 1)),
        base_channels=int(mcfg.get("base_channels", 32)),
        residual=bool(mcfg.get("residual", True)),
    )


def git_sha() -> Optional[str]:
    """The current commit (with ``-dirty`` if there are local changes), or None."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=here,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha or None
    except Exception:
        return None


def _blend_window(size: int, overlap: int) -> np.ndarray:
    """1-D weights for stitching tiles: linear ramps over the overlap, 1 elsewhere."""
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 1.0) / (overlap + 1.0)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


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
        self.cv_ops = Preprocessor.from_config(self.config["preprocessor"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        icfg = self.config["inference"]
        self.tile_size = int(icfg.get("tile_size", 1024))
        self.tile_overlap = int(icfg.get("tile_overlap", 64))
        self.half_precision = bool(icfg.get("half_precision", True))

        self.model = model_from_config(self.config["model"]).to(self.device).eval()
        self.weights_loaded = False
        self.checkpoint_meta: Dict[str, Any] = {}
        if model_weights:
            self.load_weights(model_weights)

    # ------------------------------------------------------------------ weights
    def load_weights(self, model_weights: str) -> None:
        """Load a checkpoint (raw state_dict or trainer wrapper) into the model.

        Checkpoints written by the trainer carry the model and preprocessing
        settings they were trained with. The model is rebuilt from the
        checkpoint's own architecture, and a warning is logged if the
        preprocessing differs from the config: the U-Net only knows the inputs
        it was trained on.
        """
        if not os.path.isfile(model_weights):
            raise FileNotFoundError(
                f"Model weights not found: {model_weights}\n"
                "Download the pre-trained model with `python scripts/download_weights.py` "
                "or train one with `python main.py --train`."
            )
        # weights_only=True refuses to unpickle arbitrary objects, so a tampered
        # checkpoint cannot execute code on load.
        ckpt = torch.load(model_weights, map_location="cpu", weights_only=True)
        is_wrapped = isinstance(ckpt, dict) and "model_state" in ckpt
        state = ckpt["model_state"] if is_wrapped else ckpt
        meta = {k: v for k, v in ckpt.items() if k != "model_state"} if is_wrapped else {}

        ckpt_model = meta.get("model_config")
        if not ckpt_model and "enc1.block.0.weight" in state:
            # Older checkpoints carry no settings: read the width off the weights.
            ckpt_model = {"base_channels": int(state["enc1.block.0.weight"].shape[0])}
        if ckpt_model and ckpt_model != {k: self.config["model"].get(k) for k in ckpt_model}:
            log.info("Using the checkpoint's model settings %s (config had %s).",
                     ckpt_model, self.config["model"])
            self.model = model_from_config(ckpt_model).to(self.device)

        ckpt_pre = meta.get("preprocessor")
        if ckpt_pre and ckpt_pre != self.cv_ops.to_config():
            diffs = {k: (ckpt_pre.get(k), v) for k, v in self.cv_ops.to_config().items()
                     if ckpt_pre.get(k) != v}
            log.warning(
                "Preprocessing differs from what %s was trained with (trained, now): %s. "
                "Hybrid output may degrade.", model_weights, diffs,
            )

        self.model.load_state_dict(state)
        self.model.eval()
        self.weights_loaded = True
        self.checkpoint_meta = meta
        log.info("Loaded weights from %s (%s)", model_weights,
                 ", ".join(f"{k}={meta[k]}" for k in ("epoch", "val_psnr", "git_sha") if k in meta))

    # ------------------------------------------------------------------ inference
    def _infer_residual(self, img01: np.ndarray) -> np.ndarray:
        """U-Net residual for a [0, 1] image, tiled if it is larger than ``tile_size``."""
        h, w = img01.shape
        ts, ov = self.tile_size, self.tile_overlap
        if h <= ts and w <= ts:
            return self._residual_tensor(img01)

        out = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        step = ts - 2 * ov
        ys = list(range(0, max(h - ts, 0) + 1, step))
        xs = list(range(0, max(w - ts, 0) + 1, step))
        if ys[-1] + ts < h:
            ys.append(h - ts)
        if xs[-1] + ts < w:
            xs.append(w - ts)
        for y in ys:
            for x in xs:
                tile = img01[y:y + ts, x:x + ts]
                res = self._residual_tensor(tile)
                wy = _blend_window(tile.shape[0], ov if h > ts else 0)
                wx = _blend_window(tile.shape[1], ov if w > ts else 0)
                win = np.outer(wy, wx)
                out[y:y + ts, x:x + ts] += res * win
                weight[y:y + ts, x:x + ts] += win
        return out / np.maximum(weight, 1e-8)

    def _residual_tensor(self, img01: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(img01)).float()[None, None].to(self.device)
        use_amp = self.half_precision and self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            residual = self.model.predict_residual(tensor)
        return residual.float().squeeze(0).squeeze(0).cpu().numpy()

    def run(
        self,
        image: np.ndarray,
        mode: str = "hybrid",
        preprocessor: Optional[Preprocessor] = None,
    ) -> np.ndarray:
        """Clean a single grayscale image in [0, 1]; returns float32 in [0, 1], same H, W.

        ``preprocessor`` overrides the configured classical settings for this
        call only (the shared pipeline is never modified, so concurrent callers
        such as web-demo users cannot see each other's settings).
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}.")
        if mode == "hybrid" and not self.weights_loaded:
            raise RuntimeError(
                "mode='hybrid' requires model weights. Pass model_weights= to "
                "the constructor or call load_weights()."
            )
        cv_ops = preprocessor or self.cv_ops
        img = cv_ops.process(np.asarray(image, dtype=np.float32))
        if mode == "cv_only":
            return img
        cleaned = img - self._infer_residual(img)
        return np.clip(cleaned, 0.0, 1.0).astype(np.float32)

    def run_staged(self, image: np.ndarray, preprocessor: Optional[Preprocessor] = None) -> dict:
        """Every intermediate stage, each in [0, 1], computing each step once.

        Keys: ``raw`` (polarity-matched input, the fair baseline), ``flat``,
        ``dog``, ``clahe`` / ``cv_only`` (the classical output), and, when weights
        are loaded, ``residual`` (signed) and ``hybrid``.
        """
        cv_ops = preprocessor or self.cv_ops
        st = cv_ops.process(np.asarray(image, dtype=np.float32), return_stages=True)
        stages = {
            "raw": Preprocessor.normalize_01(st["polarity"]),
            "flat": Preprocessor.normalize_01(st["flat"]),
            "dog": Preprocessor.normalize_01(st["dog"]),
            "clahe": st["out"],
            "cv_only": st["out"],
        }
        if self.weights_loaded:
            residual = self._infer_residual(st["out"])
            stages["residual"] = residual
            stages["hybrid"] = np.clip(st["out"] - residual, 0.0, 1.0).astype(np.float32)
        return stages
