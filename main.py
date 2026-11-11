"""CLI entry point for the Phase-Contrast Clean-Up pipeline (inference and training)."""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from phase_cleanup.image_io import list_images, load_gray01, output_path, save_image01
from phase_cleanup.pipeline import CleanupPipeline, load_yaml

log = logging.getLogger("main")


def _held_out_names(args: argparse.Namespace, config: dict):
    """If --input_dir is the training folder, return its validation file names.

    Scoring images the model trained on inflates every number, so the report
    is restricted to the held-out split the trainer defines.
    """
    from phase_cleanup.dataset import _list_pairs, split_pairs

    train_dir = os.path.abspath(os.path.join(args.data_dir, "noisy_input"))
    if os.path.abspath(args.input_dir) != train_dir or not args.clean_dir:
        return None, ""
    pairs = _list_pairs(args.input_dir, args.clean_dir)
    seed = int(config["training"].get("seed", 42))
    val_split = float(config["data"].get("val_split", 0.2))
    _, val = split_pairs(pairs, val_split, seed)
    names = [os.path.basename(n) for n, _ in val]
    log.warning("%s is the training folder: the report scores only its %d held-out "
                "validation images. Use data/test for the final numbers.", args.input_dir, len(names))
    return names, f"validation split, {len(names)} of {len(pairs)}"


def run_inference(args: argparse.Namespace) -> None:
    """Clean every image in --input_dir and write results + a report."""
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = CleanupPipeline(
        config_path=args.config,
        model_weights=args.weights if args.mode == "hybrid" else None,
    )
    log.info("device=%s, mode=%s", pipeline.device, args.mode)

    names = list_images(args.input_dir)
    if not names:
        raise SystemExit(f"No images found in {args.input_dir}")

    times = []
    for i, name in enumerate(names):
        img, depth = load_gray01(os.path.join(args.input_dir, name), return_depth=True)
        if i == 0:  # warm-up: the first call pays for CUDA/cuDNN initialisation
            pipeline.run(img, mode=args.mode)
        if pipeline.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cleaned = pipeline.run(img, mode=args.mode)
        if pipeline.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        # Write back at the input's precision: a 16-bit frame stays 16-bit.
        save_image01(output_path(args.output_dir, name), cleaned, bit_depth=depth)

    log.info("Cleaned %d images -> %s (median %.1f ms/image, mode=%s)",
             len(names), args.output_dir, float(np.median(times)), args.mode)

    if not args.no_report:
        from phase_cleanup.report import generate_report

        ablation = None
        if args.ablation_weights:
            ablation = CleanupPipeline(args.config, model_weights=args.ablation_weights)
        eval_names, note = _held_out_names(args, pipeline.config)
        generate_report(
            pipeline,
            noisy_dir=args.input_dir,
            clean_dir=args.clean_dir,
            out_dir=args.output_dir,
            names=eval_names,
            ablation=ablation,
            split_note=note,
        )


def run_training(args: argparse.Namespace) -> None:
    """Train the residual U-Net from paired data under --data_dir."""
    # Imported here so a CV-only user never needs the training stack.
    from phase_cleanup.dataset import build_dataloaders
    from phase_cleanup.pipeline import model_from_config
    from phase_cleanup.preprocessor import Preprocessor
    from phase_cleanup.trainer import Trainer

    config = load_yaml(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.base_channels is not None:
        config["model"]["base_channels"] = args.base_channels
    if args.plain_unet:
        config["model"]["residual"] = False
    if args.runs_dir:
        config["paths"]["runs_dir"] = args.runs_dir
        config["paths"]["best_checkpoint"] = os.path.join(args.runs_dir, "best.pth")
        config["paths"]["last_checkpoint"] = os.path.join(args.runs_dir, "last.pth")

    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    pre = Preprocessor.from_config(config["preprocessor"])
    dcfg, tcfg = config["data"], config["training"]
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        preprocessor=pre,
        patch_size=dcfg.get("patch_size", 256),
        batch_size=tcfg.get("batch_size", 16),
        val_split=dcfg.get("val_split", 0.2),
        num_workers=dcfg.get("num_workers", 0),
        seed=seed,
    )

    model = model_from_config(config["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, train_loader, val_loader, config, device=device,
                      preprocessor_config=pre.to_config())
    if args.resume:
        trainer.resume()
    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-Contrast Microscopy Clean-Up Pipeline (CV + DL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/default_config.yaml", help="path to YAML config")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    p.add_argument("--train", action="store_true", help="run training instead of inference")

    p.add_argument("--input_dir", help="folder of noisy input images (inference)")
    p.add_argument("--output_dir", default="results", help="where to write cleaned images/report")
    p.add_argument("--mode", choices=["cv_only", "hybrid"], default="hybrid", help="pipeline toggle")
    p.add_argument("--weights", default=None,
                   help="model checkpoint for hybrid mode (default: paths.pretrained in the config)")
    p.add_argument("--clean_dir", default=None,
                   help="optional ground-truth folder to enable PSNR/SSIM in the report")
    p.add_argument("--ablation_weights", default=None,
                   help="plain-U-Net checkpoint to include as an ablation row in the report")
    p.add_argument("--no_report", action="store_true", help="skip report.md generation")

    p.add_argument("--data_dir", default="data", help="dataset root (training)")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--batch_size", type=int, default=None, help="override config batch size")
    p.add_argument("--base_channels", type=int, default=None, help="override model width")
    p.add_argument("--plain_unet", action="store_true", help="train the non-residual ablation")
    p.add_argument("--runs_dir", default=None, help="override output folder for checkpoints/logs")
    p.add_argument("--resume", action="store_true", help="resume training from last.pth")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(name)s] %(message)s")
    if args.train:
        run_training(args)
        return
    if not args.input_dir:
        raise SystemExit("--input_dir is required for inference (or pass --train to train).")
    if args.weights is None:
        args.weights = load_yaml(args.config)["paths"].get("pretrained", "models/pretrained_fp16.pth")
    run_inference(args)


if __name__ == "__main__":
    main()
