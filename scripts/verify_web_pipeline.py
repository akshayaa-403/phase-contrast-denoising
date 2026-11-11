"""Measure how closely docs/cv.js reproduces the Python classical pipeline.

The demo page re-implements ``Preprocessor.process`` (``phase_cleanup/preprocessor.py``)
in JavaScript so it can run without a server. That port is only trustworthy if
someone checks it, so this script runs both implementations on the same images
and reports the difference, at both CLAHE precisions (8-bit, the default, and
the opt-in 16-bit), with the configured polarity, background flattening and
percentile normalisation. It also checks the page's own PSNR (plain and
affine-matched), SSIM and Otsu Dice against ``phase_cleanup.metrics``, and that
the page's polarity-matched "Before" image equals Python's ``run_staged`` raw.

    python scripts/verify_web_pipeline.py               # all sample images
    python scripts/verify_web_pipeline.py --images 3

Each side reads the sample files itself, exactly as it would in use: Python
with ``phase_cleanup.image_io.load_gray01`` (as ``main.py`` does) and
JavaScript with ``docs/decode.js`` (as the page does). Both produce float32 in
[0, 1] scaled by the file's bit depth, and the script checks the decoded pixels
are bit-identical before comparing anything downstream.

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_cleanup.image_io import load_gray01  # noqa: E402
from phase_cleanup.metrics import otsu_dice, psnr, psnr_affine, ssim  # noqa: E402
from phase_cleanup.pipeline import CleanupPipeline  # noqa: E402
from phase_cleanup.preprocessor import Preprocessor  # noqa: E402

# Runs the JS pipeline over raw float32 images piped in as JSON, and -- where a
# clean target is supplied -- its own metrics, so both the image and the
# metric implementations get checked.
NODE_DRIVER = """
const fs = require('fs');
const zlib = require('zlib');
const PhaseCV = require(process.argv[2]);
const PhaseDecode = require(process.argv[3]);
const job = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const inflate = async (bytes) => new Uint8Array(zlib.inflateSync(bytes));
const decodeFile = async (file) => (await PhaseDecode.decode(new Uint8Array(fs.readFileSync(file)), inflate));
(async () => {
  const out = [];
  for (const item of job.images) {
    const image = await decodeFile(item.file);
    const stages = PhaseCV.runCvPipeline(image.pixels, image.width, image.height, job.options);
    const result = stages.result;
    const entry = {
      width: image.width, height: image.height,
      decoded: Array.from(image.pixels),
      result: Array.from(result),
      raw: Array.from(PhaseCV.normalize01(stages.polarity)),
    };
    if (item.truthFile) {
      const truth = (await decodeFile(item.truthFile)).pixels;
      entry.truthDecoded = Array.from(truth);
      entry.psnr = PhaseCV.psnr(result, truth);
      entry.psnrAffine = PhaseCV.psnrAffine(result, truth);
      entry.ssim = PhaseCV.ssim(result, truth, image.width, image.height);
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
  fs.writeFileSync(process.argv[5], JSON.stringify(out));
})().catch((err) => { console.error(err); process.exit(1); });
"""


def js_options(pre: Preprocessor) -> dict:
    """The cv.js runCvPipeline options equivalent to a Preprocessor."""
    pct = pre.norm_percentiles
    return {
        "sigma1": pre.dog_sigma1,
        "sigma2": pre.dog_sigma2,
        "alpha": pre.dog_alpha,
        "clipLimit": pre.clahe_clip_limit,
        "tilesX": pre.clahe_grid_size[0],
        "tilesY": pre.clahe_grid_size[1],
        "bits": 8 if pre.clahe_dtype == "uint8" else 16,
        "invert": pre.invert,
        "backgroundRadius": pre.background_radius,
        "pLow": pct[0] if pct else None,
        "pHigh": pct[1] if pct else None,
    }


def run_javascript(
    files: list[Path],
    options: dict,
    truth_files: list[Path | None],
    same_inputs: list[dict[str, np.ndarray]] | None = None,
) -> list[dict]:
    """Decode the given files with docs/decode.js and run docs/cv.js on them via Node.

    Returns one dict per file with ``decoded`` (the [0, 1] pixels the page
    would start from), ``result`` (the pipeline output), ``raw`` (the
    polarity-matched, min-max normalised input) and, when a ground truth file
    is given, ``truthDecoded`` and the JavaScript-computed ``psnr``,
    ``psnrAffine``, ``ssim``, ``dice`` and ``diceSame`` (Dice on each of
    ``same_inputs``).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        driver = tmp_path / "driver.js"
        driver.write_text(NODE_DRIVER, encoding="utf-8")

        payload = []
        for index, path in enumerate(files):
            item = {"file": str(path)}
            if truth_files[index] is not None:
                item["truthFile"] = str(truth_files[index])
            if same_inputs:
                item["sameInputs"] = {k: v.ravel().tolist() for k, v in same_inputs[index].items()}
            payload.append(item)
        job_file = tmp_path / "job.json"
        job_file.write_text(json.dumps({"options": options, "images": payload}), encoding="utf-8")
        out_file = tmp_path / "out.json"

        result = subprocess.run(
            ["node", str(driver), str(REPO_ROOT / "docs" / "cv.js"), str(REPO_ROOT / "docs" / "decode.js"),
             str(job_file), str(out_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"Node failed:\n{result.stderr}")
        raw = json.loads(out_file.read_text(encoding="utf-8"))

    for entry in raw:
        shape = (entry["height"], entry["width"])
        for key in ("decoded", "result", "raw", "truthDecoded"):
            if key in entry:
                entry[key] = np.asarray(entry[key], dtype=np.float32).reshape(shape)
    return raw


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
    base_cfg = pipeline.cv_ops.to_config()

    images01 = [load_gray01(str(path)) for path in files]  # exactly as main.py loads them
    truth_files: list[Path | None] = []
    truths: list[np.ndarray | None] = []
    hybrids: list[np.ndarray | None] = []
    for path in files:
        truth_path = path.with_name(path.name.replace("_noisy", "_truth"))
        hybrid_path = path.with_name(path.name.replace("_noisy", "_hybrid"))
        truth_files.append(truth_path if truth_path.is_file() else None)
        truths.append(load_gray01(str(truth_path)) if truth_path.is_file() else None)
        hybrids.append(load_gray01(str(hybrid_path)) if hybrid_path.is_file() else None)

    failures: list[str] = []
    for bits, dtype in ((8, "uint8"), (16, "uint16")):
        cv_ops = Preprocessor.from_config(base_cfg, clahe_dtype=dtype)
        options = js_options(cv_ops)
        print(f"=== {bits}-bit CLAHE ({dtype}), {len(files)} image(s): {options}\n")

        staged = [pipeline.run_staged(img, preprocessor=cv_ops) for img in images01]
        python_out = [s["cv_only"] for s in staged]
        same_inputs = []
        for st, hybrid in zip(staged, hybrids, strict=True):
            entry = {"raw": st["raw"], "cv_only": st["cv_only"]}
            if hybrid is not None:
                entry["hybrid"] = hybrid
            same_inputs.append(entry)

        js_out = run_javascript(files, options, truth_files, same_inputs)

        # The page must start from the very numbers Python does.
        decode_gap = 0.0
        for img, truth, actual in zip(images01, truths, js_out, strict=True):
            decode_gap = max(decode_gap, float(np.abs(actual["decoded"] - img).max()))
            if truth is not None:
                decode_gap = max(decode_gap, float(np.abs(actual["truthDecoded"] - truth).max()))
        print(f"decode.js vs load_gray01, largest pixel difference: {decode_gap:.3g} (limit 0)\n")
        if decode_gap > 0:
            failures.append(f"{bits}-bit: decode.js no longer reproduces load_gray01 bit for bit")

        print("Pipeline output: JavaScript vs Python")
        print(f"{'image':<24}{'PSNR (dB)':>12}{'SSIM':>10}{'max |diff|':>13}{'raw max|diff|':>15}")
        print("-" * 74)
        worst = float("inf")
        raw_gap = 0.0
        for path, expected, st, actual in zip(files, python_out, staged, js_out, strict=True):
            agreement = psnr(actual["result"], expected)
            similarity = ssim(actual["result"], expected)
            max_diff = float(np.abs(actual["result"] - expected).max())
            d_raw = float(np.abs(actual["raw"] - st["raw"]).max())
            worst = min(worst, agreement)
            raw_gap = max(raw_gap, d_raw)
            print(f"{path.name:<24}{agreement:>12.2f}{similarity:>10.4f}{max_diff:>13.5f}{d_raw:>15.6f}")
        print("-" * 74)
        print(f"worst agreement: {worst:.2f} dB (threshold {args.min_psnr:.1f} dB)")
        print(f"largest polarity-matched 'Before' difference: {raw_gap:.6f} (limit 1e-4)\n")
        if worst < args.min_psnr:
            failures.append(f"{bits}-bit: the JavaScript port has drifted from the Python pipeline")
        if raw_gap > 1e-4:
            failures.append(f"{bits}-bit: the page's polarity-matched Before differs from run_staged raw")

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
        print("Metrics: JavaScript vs phase_cleanup.metrics")
        # ASCII only: a Windows console defaulting to cp1252 cannot print a delta.
        print(f"{'image':<24}{'diff PSNR':>11}{'diff aPSNR':>11}{'diff SSIM':>11}{'Dice same':>11}{'Dice e2e':>11}")
        print("-" * 79)
        for path, truth, actual, same, expected in rows:
            d_psnr = abs(actual["psnr"] - psnr(actual["result"], truth))
            d_apsnr = abs(actual["psnrAffine"] - psnr_affine(actual["result"], truth))
            d_ssim = abs(actual["ssim"] - ssim(actual["result"], truth))
            # Implementation: identical inputs on both sides (raw, Python's
            # cv_only output, the hybrid PNG) must give identical Dice.
            d_dice = max(abs(actual["diceSame"][k] - otsu_dice(v, truth)) for k, v in same.items())
            # End to end: each side segments its own pipeline output.
            d_end = abs(actual["dice"] - otsu_dice(expected, truth))
            metric_gap = max(metric_gap, d_psnr, d_apsnr, d_ssim)
            dice_impl_gap = max(dice_impl_gap, d_dice)
            dice_end_gap = max(dice_end_gap, d_end)
            print(f"{path.name:<24}{d_psnr:>11.5f}{d_apsnr:>11.5f}{d_ssim:>11.5f}{d_dice:>11.5f}{d_end:>11.5f}")
        print("-" * 79)
        print(f"largest PSNR/SSIM disagreement:          {metric_gap:.5f} (limit 0.01)")
        print(f"largest Dice disagreement, same inputs:  {dice_impl_gap:.5f} (limit 0)")
        print(f"largest Dice disagreement, end to end:   {dice_end_gap:.5f} (limit 0.0005, i.e. 3 decimals)\n")
        if metric_gap > 0.01:
            failures.append(f"{bits}-bit: the JavaScript PSNR/SSIM no longer match phase_cleanup.metrics")
        if dice_impl_gap > 1e-9:
            failures.append(f"{bits}-bit: the JavaScript Otsu Dice no longer matches otsu_dice")
        if dice_end_gap >= 0.0005:
            failures.append(f"{bits}-bit: end-to-end Dice differs from Python in the third decimal")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}.")
        return 1
    print("PASS: the demo reproduces the Python pipeline at both precisions, and its PSNR, SSIM and Dice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
