"""Compress a trained checkpoint to float16 to roughly halve its file size.

    python scripts/compress_checkpoint.py --src runs/best.pth --dst models/pretrained_fp16.pth

Keeps every metadata key the trainer wrote (model and preprocessing settings,
epoch, validation PSNR, git commit) so the compressed file is just as
traceable; drops optimizer/scheduler state, which only matters for resuming.
Prints the SHA-256 of the output for scripts/download_weights.py.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import torch

TRAINING_ONLY = {"optimizer_state", "scheduler_state", "scaler_state", "best_psnr",
                 "epochs_without_improvement"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compress(src: str, dst: str) -> str:
    """Cast float tensors to fp16, keep metadata, save to ``dst``; return its SHA-256."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    wrapped = isinstance(ckpt, dict) and "model_state" in ckpt
    state = ckpt["model_state"] if wrapped else ckpt

    half_state = {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in state.items()
    }
    out = {"model_state": half_state, "dtype": "float16"}
    if wrapped:
        out.update({k: v for k, v in ckpt.items() if k != "model_state" and k not in TRAINING_ONLY})

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(out, dst)

    digest = sha256(dst)
    print(f"[compress] {src} ({os.path.getsize(src) / 1e6:.1f} MB) -> "
          f"{dst} ({os.path.getsize(dst) / 1e6:.1f} MB)\n[compress] sha256 {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a checkpoint to float16.")
    parser.add_argument("--src", default="runs/best.pth", help="source fp32 checkpoint")
    parser.add_argument("--dst", default="models/pretrained_fp16.pth", help="output fp16 checkpoint")
    args = parser.parse_args()
    compress(args.src, args.dst)


if __name__ == "__main__":
    main()
