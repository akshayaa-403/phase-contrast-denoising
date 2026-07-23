"""CLI entry point for the Phase-Contrast Clean-Up pipeline (inference and training)."""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import cv2
import numpy as np
import torch

from src.pipeline import CleanupPipeline, load_yaml
from src.preprocessor import Preprocessor
from src.report import generate_report


IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def _list_images(folder: str) -> List[str]:
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def _load_gray01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    maxv = 65535.0 if img.max() > 255 else 255.0
    return np.clip(img / maxv, 0.0, 1.0)


def run_inference(args: argparse.Namespace) -> None:
    """Clean every image in --input_dir and write results + a report."""
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = CleanupPipeline(
        config_path=args.config,
        model_weights=args.weights if args.mode == "hybrid" else None,
    )
    print(f"[main] device={pipeline.device}, mode={args.mode}")

    names = _list_images(args.input_dir)
    if not names:
        raise SystemExit(f"No images found in {args.input_dir}")

    times = []
    for name in names:
        img = _load_gray01(os.path.join(args.input_dir, name))
        t0 = time.perf_counter()
        cleaned = pipeline.run(img, mode=args.mode)
        times.append((time.perf_counter() - t0) * 1000.0)

        out_u8 = (np.clip(cleaned, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, name), out_u8)

    print(f"[main] cleaned {len(names)} images -> {args.output_dir} "
          f"(avg {np.mean(times):.1f} ms/image, mode={args.mode})")

    if not args.no_report:
        generate_report(
            pipeline,
            noisy_dir=args.input_dir,
            clean_dir=args.clean_dir,
            out_dir=args.output_dir,
        )


def run_training(args: argparse.Namespace) -> None:
    """Train the residual U-Net from paired data under --data_dir."""
    # Imported here so a CV-only user never needs the training stack.
    from src.dataset import build_dataloaders
    from src.residual_unet import ResidualUNet
    from src.trainer import Trainer

    config = load_yaml(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    seed = int(config.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    pcfg = config.get("preprocessor", {})
    pre = Preprocessor(
        dog_sigma1=pcfg.get("dog_sigma1", 1.0),
        dog_sigma2=pcfg.get("dog_sigma2", 1.6),
        dog_alpha=pcfg.get("dog_alpha", 0.5),
        clahe_clip_limit=pcfg.get("clahe_clip_limit", 2.0),
        clahe_grid_size=tuple(pcfg.get("clahe_grid_size", (8, 8))),
    )

    dcfg = config.get("data", {})
    tcfg = config.get("training", {})
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        preprocessor=pre,
        patch_size=dcfg.get("patch_size", 256),
        batch_size=tcfg.get("batch_size", 16),
        val_split=dcfg.get("val_split", 0.2),
        num_workers=dcfg.get("num_workers", 0),
        seed=seed,
    )

    mcfg = config.get("model", {})
    model = ResidualUNet(
        in_channels=mcfg.get("in_channels", 1),
        out_channels=mcfg.get("out_channels", 1),
        base_channels=mcfg.get("base_channels", 64),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-Contrast Microscopy Clean-Up Pipeline (CV + DL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/default_config.yaml",
                   help="path to YAML config")

    p.add_argument("--train", action="store_true", help="run training instead of inference")

    p.add_argument("--input_dir", help="folder of noisy input images (inference)")
    p.add_argument("--output_dir", default="results", help="where to write cleaned images/report")
    p.add_argument("--mode", choices=["cv_only", "hybrid"], default="hybrid",
                   help="pipeline toggle")
    p.add_argument("--weights", default="runs/best.pth",
                   help="model checkpoint (required for hybrid)")
    p.add_argument("--clean_dir", default=None,
                   help="optional ground-truth folder to enable PSNR/SSIM in the report")
    p.add_argument("--no_report", action="store_true", help="skip report.md generation")

    p.add_argument("--data_dir", default="data", help="dataset root (training)")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--batch_size", type=int, default=None, help="override config batch size")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.train:
        run_training(args)
    else:
        if not args.input_dir:
            raise SystemExit("--input_dir is required for inference "
                             "(or pass --train to train).")
        run_inference(args)


if __name__ == "__main__":
    main()
