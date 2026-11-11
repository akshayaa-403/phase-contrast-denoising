"""Evaluate on real phase-contrast microscopy: Cell Tracking Challenge PhC-C2DH-U373.

    python scripts/eval_real_data.py --download          # fetch (~43 MB) and evaluate
    python scripts/eval_real_data.py --scales 1.0 0.35   # evaluate at several pixel scales

The synthetic data the model trains on has no claim to realism beyond its
physics model, so this script scores the pipeline on real images: U373
glioblastoma cells imaged in phase contrast, from the Cell Tracking Challenge
(Ulman et al., Nature Methods 2017; data: https://celltrackingchallenge.net,
licensed for research use). There is no clean "target image" for real data, so
the score is downstream: the Dice overlap between an **Otsu threshold** of each
output and the **manual segmentation** (the challenge's gold-standard ``SEG``
annotations, 34 frames across the two training sequences).

``--scales`` resizes each frame before processing (and the output back after),
so cells can be brought to the size the model and the classical defaults were
tuned for. U373 cells are ~60-120 px across, the synthetic cells ~15-50 px, so
scale ~0.35 roughly matches them. The report shows every scale side by side;
picking the best one afterwards would be tuning on the test set, so the
recommended reading is the pre-registered scale (0.35, chosen from cell size
alone, before looking at scores).

The dataset is unpacked under ``data/real/`` (git-ignored). The archive is
checked for unsafe paths before extraction.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_cleanup.image_io import _read_raw, load_gray01  # noqa: E402
from phase_cleanup.metrics import bootstrap_ci, otsu_dice_vs_mask  # noqa: E402
from phase_cleanup.pipeline import CleanupPipeline, load_yaml  # noqa: E402

URL = "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"
NAME = "PhC-C2DH-U373"
PREREGISTERED_SCALE = 0.35

log = logging.getLogger("eval_real")


def download(root: Path) -> Path:
    """Download and safely extract the dataset under ``root``; return its folder."""
    target = root / "extracted" / NAME
    if target.is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{NAME}.zip"
    if not archive.is_file():
        log.info("Downloading %s ...", URL)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            resolved = (root / "extracted" / name).resolve()
            if not str(resolved).startswith(str((root / "extracted").resolve())):
                raise SystemExit(f"Refusing unsafe path in archive: {name}")
        zf.extractall(root / "extracted")
    return target


def frames(dataset: Path):
    """Yield (sequence, frame_id, image_path, mask_path) for every gold-standard SEG frame."""
    for seq in ("01", "02"):
        for mask in sorted((dataset / f"{seq}_GT" / "SEG").glob("man_seg*.tif")):
            fid = re.search(r"(\d+)", mask.stem).group(1)
            image = dataset / seq / f"t{fid}.tif"
            if image.is_file():
                yield seq, fid, image, mask


def run_at_scale(pipeline: CleanupPipeline, image: np.ndarray, scale: float) -> dict:
    """Run raw / cv_only / hybrid at ``scale``; return outputs at the original size."""
    h, w = image.shape
    small = image if scale == 1.0 else cv2.resize(
        image, (max(16, round(w * scale)), max(16, round(h * scale))), interpolation=cv2.INTER_AREA)
    stages = pipeline.run_staged(small)
    outs = {k: stages[k] for k in ("raw", "cv_only", "hybrid") if k in stages}
    if scale != 1.0:
        outs = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) for k, v in outs.items()}
    return outs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "real")
    parser.add_argument("--download", action="store_true", help="download the dataset if missing")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, PREREGISTERED_SCALE])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results_real")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    dataset = args.root / "extracted" / NAME
    if not dataset.is_dir():
        if not args.download:
            raise SystemExit(f"{dataset} not found. Re-run with --download.")
        dataset = download(args.root)

    weights = args.weights or REPO_ROOT / load_yaml(str(args.config))["paths"].get(
        "pretrained", "models/pretrained_fp16.pth")
    pipeline = CleanupPipeline(str(args.config), model_weights=str(weights) if Path(weights).is_file() else None)
    if not pipeline.weights_loaded:
        log.warning("No weights at %s: scoring raw and cv_only only.", weights)

    items = list(frames(dataset))
    if not items:
        raise SystemExit(f"No SEG frames found under {dataset}")
    paths = ["raw", "cv_only"] + (["hybrid"] if pipeline.weights_loaded else [])
    scores = {s: {p: [] for p in paths} for s in args.scales}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    example = None
    for i, (seq, fid, image_path, mask_path) in enumerate(items):
        image = load_gray01(str(image_path))
        mask = _read_raw(str(mask_path)) > 0
        for scale in args.scales:
            outs = run_at_scale(pipeline, image, scale)
            for p in paths:
                scores[scale][p].append(otsu_dice_vs_mask(outs[p], mask))
            if i == 0 and scale == PREREGISTERED_SCALE:
                example = (image, mask, outs)
        log.info("%s/t%s  %s", seq, fid, "  ".join(
            f"s{s}: " + " ".join(f"{p}={scores[s][p][-1]:.3f}" for p in paths) for s in args.scales))

    # ------------------------------------------------------------------ report
    labels = {"raw": "Raw (polarity-matched)", "cv_only": "CV-only", "hybrid": "Hybrid"}
    lines = [f"# Real-data evaluation: {NAME}\n",
             f"{len(items)} frames with gold-standard manual segmentation (sequences 01 and 02). "
             "Score: Dice between an Otsu threshold of each output and the manual cell mask; "
             "mean with 95% bootstrap CI. Higher is better.\n",
             "| Path | " + " | ".join(
                 f"scale {s}{' (pre-registered)' if s == PREREGISTERED_SCALE else ''}" for s in args.scales) + " |",
             "| :--- |" + " ---: |" * len(args.scales)]
    for p in paths:
        cells = []
        for s in args.scales:
            m, lo, hi = bootstrap_ci(scores[s][p])
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {labels[p]} | " + " | ".join(cells) + " |")
    lines += ["",
              "The model was trained only on synthetic data. Real U373 frames differ from it in "
              "cell size, texture, halo shape and noise, so this table measures how far the "
              "synthetic training transfers, not what a model trained on real data would reach.\n"]

    if example is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        image, mask, outs = example
        panels = [(image, "Input"), *[(outs[p], labels[p]) for p in paths], (mask, "Manual mask")]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0))
        for ax, (img, title) in zip(axes, panels, strict=True):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{NAME}, scale {PREREGISTERED_SCALE}")
        fig.tight_layout()
        fig.savefig(args.out_dir / "real_example.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        lines.append("![example](real_example.png)\n")

    report = args.out_dir / "report_real.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", report)
    print("\n".join(lines[2:2 + 2 + len(paths)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
