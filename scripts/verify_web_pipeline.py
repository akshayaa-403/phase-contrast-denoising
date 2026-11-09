"""Measure how closely docs/cv.js reproduces the OpenCV pipeline.

The demo page re-implements `src/preprocessor.py` in JavaScript so it can run
without a server. That port is only trustworthy if someone checks it, so this
script runs both implementations on the same images and reports the difference,
at both CLAHE precisions the page offers (16-bit as shipped, and the opt-in
8-bit `Preprocessor(clahe_dtype="uint8")`). It also checks the page's own PSNR,
SSIM and Otsu Dice against scikit-image and `src/report.py::_otsu_dice`.

    python scripts/verify_web_pipeline.py               # all sample images
    python scripts/verify_web_pipeline.py --images 3

Requires Node.js on PATH. Exits non-zero if agreement falls below --min-psnr,
or any metric disagrees, so it can run in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import psnr, ssim  # noqa: E402
from src.pipeline import CleanupPipeline  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.report import _otsu_dice  # noqa: E402

# Runs the JS pipeline over raw float32 images piped in as JSON, and -- where a
# clean target is supplied -- its own PSNR/SSIM/Dice, so both the image and the
# metric implementations get checked.
NODE_DRIVER = """
const fs = require('fs');
const PhaseCV = require(process.argv[2]);
const job = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const item of job.images) {
  const src = Float32Array.from(item.pixels);
  const { result } = PhaseCV.runCvPipeline(src, item.width, item.height, job.options);
  const entry = { result: Array.from(result) };
  if (item.truth) {
    const truth = Float32Array.from(item.truth);
    entry.psnr = PhaseCV.psnr(result, truth);
    entry.ssim = PhaseCV.ssim(result, truth, item.width, item.height);
    entry.dice = PhaseCV.otsuDice(result, truth);
    // The Dice port on inputs identical to Python's, so the implementation is
    // checked separately from the pipeline that feeds it.
    entry.diceSame = {};
    for (const [name, pixels] of Object.entries(item.sameInputs || {})) {
      entry.diceSame[name] = PhaseCV.otsuDice(Float32Array.from(pixels), truth);
    }
  }
  out.push(entry);
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def load_gray(path: Path) -> np.ndarray:
    """Load an image as float32, 0-255."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    if image.max() > 255:
        image = image / 65535.0 * 255.0
    return image


def run_javascript(
    images: list[np.ndarray],
    options: dict,
    truths: list[np.ndarray | None] | None = None,
    same_inputs: list[dict[str, np.ndarray]] | None = None,
) -> list[dict]:
    """Execute docs/cv.js on the given images via Node.

    Returns one dict per image with ``result`` (the pipeline output) and, when a
    ground truth was supplied, the JavaScript-computed ``psnr``, ``ssim`` and
    ``dice``, plus ``diceSame``: Dice on each of ``same_inputs``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        driver = tmp_path / "driver.js"
        driver.write_text(NODE_DRIVER, encoding="utf-8")

        payload = []
        for index, img in enumerate(images):
            item = {"width": img.shape[1], "height": img.shape[0], "pixels": img.ravel().tolist()}
            truth = truths[index] if truths else None
            if truth is not None:
                item["truth"] = truth.ravel().tolist()
            if same_inputs:
                item["sameInputs"] = {k: v.ravel().tolist() for k, v in same_inputs[index].items()}
            payload.append(item)
        job = {"options": options, "images": payload}
        job_file = tmp_path / "job.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        out_file = tmp_path / "out.json"

        result = subprocess.run(
            ["node", str(driver), str(REPO_ROOT / "docs" / "cv.js"), str(job_file), str(out_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"Node failed:\n{result.stderr}")

        raw = json.loads(out_file.read_text(encoding="utf-8"))

    out = []
    for entry, img in zip(raw, images, strict=True):
        entry["result"] = np.asarray(entry["result"], dtype=np.float32).reshape(img.shape)
        out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "docs" / "samples")
    parser.add_argument("--pattern", type=str, default="*_noisy.png")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--images", type=int, default=None, help="Cap how many to check")
    parser.add_argument(
        "--min-psnr",
        type=float,
        default=40.0,
        help="Fail if any image agrees less closely than this (dB)",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(
            f"No images matching {args.pattern} under {args.input_dir}. "
            "Run scripts/make_demo_assets.py first."
        )
    if args.images:
        files = files[: args.images]

    pipeline = CleanupPipeline(str(args.config))
    base = pipeline.cv_ops

    images = [load_gray(path) for path in files]
    truths: list[np.ndarray | None] = []
    hybrids: list[np.ndarray | None] = []
    for path in files:
        truth_path = path.with_name(path.name.replace("_noisy", "_truth"))
        hybrid_path = path.with_name(path.name.replace("_noisy", "_hybrid"))
        truths.append(load_gray(truth_path) / 255.0 if truth_path.is_file() else None)
        hybrids.append(load_gray(hybrid_path) / 255.0 if hybrid_path.is_file() else None)

    failures: list[str] = []
    for bits, dtype in ((16, "uint16"), (8, "uint8")):
        cv_ops = Preprocessor(
            dog_sigma1=base.dog_sigma1,
            dog_sigma2=base.dog_sigma2,
            dog_alpha=base.dog_alpha,
            clahe_clip_limit=base.clahe_clip_limit,
            clahe_grid_size=base.clahe_grid_size,
            clahe_dtype=dtype,
        )
        pipeline.cv_ops = cv_ops
        options = {
            "sigma1": cv_ops.dog_sigma1,
            "sigma2": cv_ops.dog_sigma2,
            "alpha": cv_ops.dog_alpha,
            "clipLimit": cv_ops.clahe_clip_limit,
            "tilesX": cv_ops.clahe_grid_size[0],
            "tilesY": cv_ops.clahe_grid_size[1],
            "bits": bits,
        }
        print(f"=== {bits}-bit CLAHE ({dtype}), {len(files)} image(s): {options}\n")

        python_out = [pipeline.run(img, mode="cv_only") for img in images]
        same_inputs = []
        for img, cv_out, hybrid in zip(images, python_out, hybrids, strict=True):
            entry = {"raw": base.normalize_01(img), "cv_only": cv_out}
            if hybrid is not None:
                entry["hybrid"] = hybrid
            same_inputs.append(entry)

        js_out = run_javascript(images, options, truths, same_inputs)

        print("Pipeline output: JavaScript vs OpenCV")
        print(f"{'image':<24}{'PSNR (dB)':>12}{'SSIM':>10}{'max |diff|':>13}")
        print("-" * 59)
        worst = float("inf")
        for path, expected, actual in zip(files, python_out, js_out, strict=True):
            agreement = psnr(actual["result"], expected)
            similarity = ssim(actual["result"], expected)
            max_diff = float(np.abs(actual["result"] - expected).max())
            worst = min(worst, agreement)
            print(f"{path.name:<24}{agreement:>12.2f}{similarity:>10.4f}{max_diff:>13.5f}")
        print("-" * 59)
        print(f"worst agreement: {worst:.2f} dB (threshold {args.min_psnr:.1f} dB)\n")
        if worst < args.min_psnr:
            failures.append(f"{bits}-bit: the JavaScript port has drifted from the OpenCV pipeline")

        rows = [
            (path, truth, actual, same, expected)
            for path, truth, actual, same, expected in zip(
                files, truths, js_out, same_inputs, python_out, strict=True
            )
            if truth is not None
        ]
        if not rows:
            continue

        metric_gap = dice_impl_gap = dice_end_gap = 0.0
        print("Metrics: JavaScript vs scikit-image and src.report._otsu_dice")
        # ASCII only: a Windows console defaulting to cp1252 cannot print a delta.
        print(f"{'image':<24}{'diff PSNR':>11}{'diff SSIM':>11}{'Dice same':>11}{'Dice e2e':>11}")
        print("-" * 68)
        for path, truth, actual, same, expected in rows:
            d_psnr = abs(actual["psnr"] - psnr(actual["result"], truth))
            d_ssim = abs(actual["ssim"] - ssim(actual["result"], truth))
            # Implementation: identical inputs on both sides (raw, Python's
            # cv_only output, the hybrid PNG) must give identical Dice.
            d_dice = max(abs(actual["diceSame"][k] - _otsu_dice(v, truth)) for k, v in same.items())
            # End to end: each side segments its own pipeline output.
            d_end = abs(actual["dice"] - _otsu_dice(expected, truth))
            metric_gap = max(metric_gap, d_psnr, d_ssim)
            dice_impl_gap = max(dice_impl_gap, d_dice)
            dice_end_gap = max(dice_end_gap, d_end)
            print(f"{path.name:<24}{d_psnr:>11.5f}{d_ssim:>11.5f}{d_dice:>11.5f}{d_end:>11.5f}")
        print("-" * 68)
        print(f"largest PSNR/SSIM disagreement:          {metric_gap:.5f} (limit 0.01)")
        print(f"largest Dice disagreement, same inputs:  {dice_impl_gap:.5f} (limit 0)")
        print(f"largest Dice disagreement, end to end:   {dice_end_gap:.5f} (limit 0.0005, i.e. 3 decimals)\n")
        if metric_gap > 0.01:
            failures.append(f"{bits}-bit: the JavaScript PSNR/SSIM no longer match scikit-image")
        if dice_impl_gap > 1e-9:
            failures.append(f"{bits}-bit: the JavaScript Otsu Dice no longer matches _otsu_dice")
        if dice_end_gap >= 0.0005:
            failures.append(f"{bits}-bit: end-to-end Dice differs from Python in the third decimal")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}.")
        return 1
    print("PASS: the demo reproduces the OpenCV pipeline at both precisions, and its PSNR, SSIM and Dice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
