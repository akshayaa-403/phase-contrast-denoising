"""Build the asset bundle the GitHub Pages demo serves.

    python scripts/make_demo_assets.py

The samples are the frames the U-Net never trained on: the held-out split that
``phase_cleanup.dataset.split_pairs`` defines (seed 42, 20% of 100 frames = 20
frames), the same function ``build_dataloaders`` uses, so the demo and the
trainer cannot disagree about which frames are fair to show.

For each held-out frame this writes, into ``docs/samples/``:

    <stem>_noisy.png     the phase-contrast input as recorded (16-bit PNG)
    <stem>_truth.png     the clean target, for the metric read-outs
    <stem>_hybrid.png    the residual U-Net's output, precomputed
    <stem>_residual.png  what the U-Net subtracted, mid-grey = 0 (see
                         ``residual_scale`` in the manifest)
    <stem>_thumb.png     a 64 px area-averaged thumbnail

plus ``manifest.json`` holding what this Python code measures: PSNR (plain and
affine-matched), SSIM and Otsu Dice per frame for the polarity-matched raw
input, the classical path at both CLAHE precisions (8-bit as shipped, 16-bit
opt-in), and the hybrid path, and a held-out summary the page's headline
reads. The page recomputes the classical path live in JavaScript and displays
its own numbers next to these, so drift between the two implementations is
visible rather than hidden (see also scripts/verify_web_pipeline.py).

``manifest.model`` points the page at ``models/phase_unet.onnx`` (relative to
docs/) for running the U-Net in the browser on uploads. That file is not in git:
the Pages workflow downloads it from the release, and for a local preview copy
the output of scripts/export_onnx.py there. The page hides the button if the
file is missing.

Running it twice gives byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_cleanup.dataset import _list_pairs, split_pairs  # noqa: E402
from phase_cleanup.image_io import load_gray01  # noqa: E402
from phase_cleanup.metrics import otsu_dice, psnr, psnr_affine, ssim  # noqa: E402
from phase_cleanup.pipeline import CleanupPipeline  # noqa: E402
from phase_cleanup.preprocessor import Preprocessor  # noqa: E402

THUMB_SIZE = 64
PATHS = ("raw", "cv_only_uint8", "cv_only_uint16", "hybrid")


def save_unit(image: np.ndarray, path: Path, bits: int = 8) -> None:
    """Write a [0, 1] float image as an 8- or 16-bit PNG."""
    top, dtype = (65535, np.uint16) if bits == 16 else (255, np.uint8)
    cv2.imwrite(str(path), (np.clip(image, 0, 1) * top).round().astype(dtype))


def save_signed(residual: np.ndarray, scale: float, path: Path) -> None:
    """Write a signed image as 8-bit PNG: 128 = 0, 1 = -scale, 255 = +scale."""
    encoded = np.clip(np.round(128 + 127 * residual / scale), 1, 255).astype(np.uint8)
    cv2.imwrite(str(path), encoded)


def legacy_split_names(pairs: list, val_split: float, seed: int) -> list:
    """The split exactly as build_dataloaders wrote it before split_pairs existed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    return [pairs[i] for i in idx[:n_val]]


def scores(image: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "psnr": round(float(psnr(image, truth)), 3),
        "psnr_affine": round(float(psnr_affine(image, truth)), 3),
        "ssim": round(float(ssim(image, truth)), 4),
        "dice": round(float(otsu_dice(image, truth)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "pretrained_fp16.pth")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "samples")
    args = parser.parse_args()

    noisy_dir = args.data_dir / "noisy_input"
    truth_dir = args.data_dir / "ground_truth"
    if not noisy_dir.is_dir() or not any(noisy_dir.iterdir()):
        raise SystemExit(f"No images in {noisy_dir}. Generate them first:\n  python data/generate_synthetic.py")

    pipeline = CleanupPipeline(str(args.config))
    config = pipeline.config
    val_split = float(config["data"].get("val_split", 0.2))
    seed = int(config["training"].get("seed", 42))

    # The same pairing and split the trainer uses.
    pairs = _list_pairs(str(noisy_dir), str(truth_dir))
    _, held_out = split_pairs(pairs, val_split, seed)
    assert held_out == legacy_split_names(pairs, val_split, seed), (
        "split_pairs no longer reproduces build_dataloaders' original held-out indices"
    )
    held_out = sorted(held_out)

    use_model = args.weights.is_file()
    if use_model:
        pipeline.load_weights(str(args.weights))
    else:
        print(f"[warn] {args.weights} not found; writing classical results only.")
    torch.set_num_threads(1)  # one thread keeps float summation order, and the output, fixed

    cv_ops = pipeline.cv_ops
    cv_ops_16 = Preprocessor.from_config(cv_ops.to_config(), clahe_dtype="uint16")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.png"):
        stale.unlink()

    # First pass: run everything, keep the residuals to pick one shared scale.
    frames = []
    for noisy_path, truth_path in held_out:
        noisy = load_gray01(noisy_path)
        truth = load_gray01(truth_path)
        stages = pipeline.run_staged(noisy)
        cv_16 = cv_ops_16.process(noisy)
        frames.append((Path(noisy_path).stem, noisy, truth, stages, cv_16))

    residual_scale = None
    if use_model:
        everything = np.concatenate([np.abs(f[3]["residual"]).ravel() for f in frames])
        # 99.9th percentile, rounded up to 0.01, so a few outliers cannot wash
        # out the display; values beyond it are clipped and the page says so.
        residual_scale = math.ceil(float(np.percentile(everything, 99.9)) * 100) / 100

    entries = []
    per_path: dict[str, dict[str, list]] = {p: {"psnr": [], "psnr_affine": [], "ssim": [], "dice": []}
                                           for p in PATHS}
    for stem, noisy, truth, stages, cv_16 in frames:
        save_unit(Preprocessor.normalize_01(noisy), args.out_dir / f"{stem}_noisy.png", bits=16)
        save_unit(truth, args.out_dir / f"{stem}_truth.png")
        thumb = cv2.resize(Preprocessor.normalize_01(noisy), (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        save_unit(thumb, args.out_dir / f"{stem}_thumb.png")

        metrics = {
            "raw": scores(stages["raw"], truth),
            "cv_only": {"uint8": scores(stages["cv_only"], truth), "uint16": scores(cv_16, truth)},
        }
        entry: dict = {"id": stem, "noisy": f"{stem}_noisy.png", "truth": f"{stem}_truth.png",
                       "thumb": f"{stem}_thumb.png"}
        if use_model:
            save_unit(stages["hybrid"], args.out_dir / f"{stem}_hybrid.png")
            save_signed(stages["residual"], residual_scale, args.out_dir / f"{stem}_residual.png")
            entry["hybrid"] = f"{stem}_hybrid.png"
            entry["residual"] = f"{stem}_residual.png"
            metrics["hybrid"] = scores(stages["hybrid"], truth)
        entry["metrics"] = metrics
        entries.append(entry)

        flat = {
            "raw": metrics["raw"],
            "cv_only_uint8": metrics["cv_only"]["uint8"],
            "cv_only_uint16": metrics["cv_only"]["uint16"],
            "hybrid": metrics.get("hybrid"),
        }
        for path, values in flat.items():
            if values:
                for key in per_path[path]:
                    per_path[path][key].append((values[key], stem))
        detail = "  ".join(f"{k}: {v['psnr']:.2f}dB/{v['dice']:.3f}" for k, v in flat.items() if v)
        print(f"{stem}  {detail}")

    def summarise(values: list) -> dict | None:
        if not values:
            return None
        numbers = [v for v, _ in values]
        lowest = min(values)
        return {"mean": round(float(np.mean(numbers)), 4), "min": lowest[0], "min_id": lowest[1]}

    summary = {
        "frames": len(entries),
        "of": len(pairs),
        "val_split": val_split,
        "seed": seed,
        "paths": {
            path: {key: summarise(per_path[path][key]) for key in per_path[path]}
            for path in PATHS
            if per_path[path]["psnr"]
        },
    }

    pct = cv_ops.norm_percentiles
    manifest = {
        "defaults": {
            "sigma1": cv_ops.dog_sigma1,
            "sigma2": cv_ops.dog_sigma2,
            "alpha": cv_ops.dog_alpha,
            "clipLimit": cv_ops.clahe_clip_limit,
            "tilesX": cv_ops.clahe_grid_size[0],
            "tilesY": cv_ops.clahe_grid_size[1],
            "bits": 8 if cv_ops.clahe_dtype == "uint8" else 16,
            "invert": cv_ops.invert,
            "backgroundRadius": cv_ops.background_radius,
            "pLow": pct[0] if pct else None,
            "pHigh": pct[1] if pct else None,
        },
        "has_model_outputs": use_model,
        "residual_scale": residual_scale,
        "model": {"onnx": "models/phase_unet.onnx", "multiple": 16},
        "held_out": summary,
        "samples": entries,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(entries)} held-out frame(s) of {len(pairs)} -> {args.out_dir}")
    for path, values in summary["paths"].items():
        print(f"  {path:<16} PSNR mean {values['psnr']['mean']:.2f}  affine {values['psnr_affine']['mean']:.2f}"
              f"   Dice mean {values['dice']['mean']:.3f}  min {values['dice']['min']:.3f}")
    print("Preview:  python -m http.server -d docs 8000")
    print("Verify the JS port matches OpenCV:  python scripts/verify_web_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
