"""Build the asset bundle the GitHub Pages demo serves.

    python scripts/make_demo_assets.py --count 6

For each selected image this writes, into ``docs/samples/``:

    <stem>_noisy.png    the phase-contrast input, as the demo receives it
    <stem>_truth.png    the clean target, for the metric read-outs
    <stem>_hybrid.png   the residual U-Net's output, precomputed

plus a ``manifest.json`` holding the PSNR/SSIM this Python code measures. The
page recomputes the classical path live in JavaScript and displays its own
numbers next to these, so any drift between the two implementations is visible
rather than hidden (see also scripts/verify_web_pipeline.py).

The U-Net output is precomputed rather than run in the browser on purpose: at
31M parameters it is a ~124MB float32 download and far too slow for WebAssembly,
while the classical front-end is a few hundred lines of arithmetic that runs in
milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import psnr, ssim  # noqa: E402
from src.pipeline import CleanupPipeline  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    """Load an image as float32 in 0-255, matching main.py."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    if image.max() > 255:
        image = image / 65535.0 * 255.0
    return image


def save_unit(image: np.ndarray, path: Path) -> None:
    """Write a [0, 1] float image as 8-bit PNG."""
    cv2.imwrite(str(path), (np.clip(image, 0, 1) * 255).round().astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--noisy-dir", type=Path, default=REPO_ROOT / "data" / "noisy_input")
    parser.add_argument("--truth-dir", type=Path, default=REPO_ROOT / "data" / "ground_truth")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "pretrained_fp16.pth")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "samples")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    noisy_files = sorted(args.noisy_dir.glob("*.png"))
    if not noisy_files:
        raise SystemExit(
            f"No images in {args.noisy_dir}. Generate them first:\n"
            "  python data/generate_synthetic.py --num 100 --size 256"
        )
    noisy_files = noisy_files[: args.count]

    use_model = args.weights.is_file()
    if not use_model:
        print(f"[warn] {args.weights} not found; writing classical results only.")

    pipeline = CleanupPipeline(str(args.config), model_weights=str(args.weights) if use_model else None)
    cv_ops = pipeline.cv_ops

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.png"):
        stale.unlink()

    entries = []
    for path in noisy_files:
        noisy = load_gray(path)
        truth_path = args.truth_dir / path.name
        truth = load_gray(truth_path) / 255.0 if truth_path.is_file() else None

        stages = pipeline.run_staged(noisy)
        stem = path.stem

        save_unit(stages["raw"], args.out_dir / f"{stem}_noisy.png")
        entry: dict = {"id": stem, "noisy": f"{stem}_noisy.png"}

        if truth is not None:
            save_unit(truth, args.out_dir / f"{stem}_truth.png")
            entry["truth"] = f"{stem}_truth.png"
            entry["metrics"] = {
                "raw": {"psnr": round(psnr(stages["raw"], truth), 3), "ssim": round(ssim(stages["raw"], truth), 4)},
                "cv_only": {
                    "psnr": round(psnr(stages["cv_only"], truth), 3),
                    "ssim": round(ssim(stages["cv_only"], truth), 4),
                },
            }

        if "hybrid" in stages:
            save_unit(stages["hybrid"], args.out_dir / f"{stem}_hybrid.png")
            entry["hybrid"] = f"{stem}_hybrid.png"
            if truth is not None:
                entry["metrics"]["hybrid"] = {
                    "psnr": round(psnr(stages["hybrid"], truth), 3),
                    "ssim": round(ssim(stages["hybrid"], truth), 4),
                }

        entries.append(entry)
        summary = entry.get("metrics", {})
        detail = "  ".join(f"{k}: {v['psnr']:.2f}dB/{v['ssim']:.3f}" for k, v in summary.items())
        print(f"{stem}  {detail}")

    manifest = {
        "defaults": {
            "sigma1": cv_ops.dog_sigma1,
            "sigma2": cv_ops.dog_sigma2,
            "alpha": cv_ops.dog_alpha,
            "clipLimit": cv_ops.clahe_clip_limit,
            "tilesX": cv_ops.clahe_grid_size[0],
            "tilesY": cv_ops.clahe_grid_size[1],
        },
        "has_model_outputs": use_model,
        "samples": entries,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(entries)} sample(s) -> {args.out_dir}")
    print("Preview:  python -m http.server -d docs 8000")
    print("Verify the JS port matches OpenCV:  python scripts/verify_web_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
