"""Pipeline tests: toggle, tiling, checkpoint loading and the fp16 round trip."""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import torch

from phase_cleanup.pipeline import CleanupPipeline, model_from_config
from phase_cleanup.preprocessor import Preprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "default_config.yaml")


@pytest.fixture()
def small_ckpt(tmp_path):
    """A tiny random checkpoint in the trainer's format (base_channels 8)."""
    torch.manual_seed(0)
    model = model_from_config({"base_channels": 8})
    pre = CleanupPipeline(CFG).cv_ops.to_config()
    path = tmp_path / "tiny.pth"
    torch.save({"model_state": model.state_dict(), "epoch": 3, "val_psnr": 20.0,
                "model_config": {"base_channels": 8}, "preprocessor": pre}, path)
    return str(path)


def test_hybrid_requires_weights():
    with pytest.raises(RuntimeError):
        CleanupPipeline(CFG).run(np.random.rand(32, 32).astype(np.float32), mode="hybrid")


def test_invalid_mode():
    with pytest.raises(ValueError):
        CleanupPipeline(CFG).run(np.zeros((16, 16), np.float32), mode="fast")


def test_missing_weights_message():
    with pytest.raises(FileNotFoundError, match="download_weights"):
        CleanupPipeline(CFG, model_weights="does/not/exist.pth")


def test_checkpoint_architecture_wins_and_runs(small_ckpt):
    """The model is rebuilt from the checkpoint's own settings (8 channels, not 32)."""
    pipe = CleanupPipeline(CFG, model_weights=small_ckpt, device="cpu")
    assert pipe.model.enc1.block[0].out_channels == 8
    out = pipe.run(np.random.rand(70, 90).astype(np.float32), mode="hybrid")
    assert out.shape == (70, 90) and 0.0 <= out.min() and out.max() <= 1.0
    assert pipe.checkpoint_meta["epoch"] == 3


def test_preprocessing_mismatch_warns(small_ckpt, tmp_path, caplog):
    ckpt = torch.load(small_ckpt, weights_only=True)
    ckpt["preprocessor"]["dog_alpha"] = 0.123
    other = tmp_path / "other.pth"
    torch.save(ckpt, other)
    with caplog.at_level(logging.WARNING):
        CleanupPipeline(CFG, model_weights=str(other), device="cpu")
    assert "differs" in caplog.text


def test_fp16_checkpoint_round_trip(small_ckpt, tmp_path):
    """compress_checkpoint's fp16 output loads into the fp32 model and agrees closely."""
    from scripts.compress_checkpoint import compress

    half = tmp_path / "half.pth"
    compress(small_ckpt, str(half))
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    full = CleanupPipeline(CFG, model_weights=small_ckpt, device="cpu").run(img, "hybrid")
    fp16 = CleanupPipeline(CFG, model_weights=str(half), device="cpu").run(img, "hybrid")
    assert np.abs(full - fp16).max() < 5e-3


def test_tiled_matches_untiled_closely(small_ckpt):
    pipe = CleanupPipeline(CFG, model_weights=small_ckpt, device="cpu")
    img = np.random.default_rng(1).random((150, 200)).astype(np.float32)
    whole = pipe.run(img, "hybrid")
    pipe.tile_size, pipe.tile_overlap = 96, 16
    tiled = pipe.run(img, "hybrid")
    assert tiled.shape == whole.shape
    # Tiles see less context, so not identical, but close.
    assert np.mean(np.abs(tiled - whole)) < 0.05


def test_per_call_preprocessor_does_not_mutate_shared_state():
    pipe = CleanupPipeline(CFG)
    before = pipe.cv_ops.to_config()
    img = np.random.rand(64, 64).astype(np.float32)
    custom = Preprocessor.from_config(before, dog_alpha=0.1)
    a = pipe.run(img, "cv_only", preprocessor=custom)
    b = pipe.run(img, "cv_only")
    assert pipe.cv_ops.to_config() == before
    assert not np.array_equal(a, b)


def test_run_staged_consistent_with_run():
    pipe = CleanupPipeline(CFG)
    img = np.random.rand(64, 64).astype(np.float32)
    np.testing.assert_array_equal(pipe.run_staged(img)["cv_only"], pipe.run(img, "cv_only"))
