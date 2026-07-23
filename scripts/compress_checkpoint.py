"""Compress a trained checkpoint to float16 to roughly halve its file size."""

from __future__ import annotations

import argparse
import os

import torch


def compress(src: str, dst: str) -> None:
    """Load an fp32 checkpoint, cast float tensors to fp16, and save to dst."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    half_state = {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in state.items()
    }

    out = {"model_state": half_state, "dtype": "float16"}
    if isinstance(ckpt, dict):
        for meta_key in ("epoch", "val_psnr"):
            if meta_key in ckpt:
                out[meta_key] = ckpt[meta_key]

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(out, dst)

    src_mb = os.path.getsize(src) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print(f"[compress] {src} ({src_mb:.1f} MB) -> {dst} ({dst_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a checkpoint to float16.")
    parser.add_argument("--src", default="runs/best.pth", help="source fp32 checkpoint")
    parser.add_argument("--dst", default="models/pretrained_fp16.pth", help="output fp16 checkpoint")
    args = parser.parse_args()
    compress(args.src, args.dst)


if __name__ == "__main__":
    main()
