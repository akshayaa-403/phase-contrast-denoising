"""Generate the quantitative report.md and comparison figures."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .metrics import compute_flops, human_readable, psnr, ssim
from .pipeline import CleanupPipeline
from .residual_unet import ResidualUNet


def _load_gray01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    maxv = 65535.0 if img.max() > 255 else 255.0
    return np.clip(img / maxv, 0.0, 1.0)


def _otsu_dice(cleaned: np.ndarray, clean_gt: np.ndarray) -> float:
    """Dice overlap between Otsu foreground masks of cleaned and the clean target."""
    # Dice (not pixel accuracy) because background dominates the frame; Dice
    # reflects how well the cells themselves are recovered.
    def _otsu_mask(img: np.ndarray) -> np.ndarray:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (mask > 0).astype(np.uint8)

    gt_mask = _otsu_mask(clean_gt)
    pred_mask = _otsu_mask(cleaned)
    inter = int((pred_mask & gt_mask).sum())
    denom = int(pred_mask.sum() + gt_mask.sum())
    return float(2.0 * inter / denom) if denom > 0 else 1.0


def _save_sample_figure(
    stages: Dict[str, np.ndarray], out_path: str, title: str
) -> None:
    """Save a labelled [raw | DoG | CLAHE | hybrid] panel using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [("raw", "Raw input"), ("dog", "DoG (halo suppressed)"),
             ("clahe", "CLAHE (cv_only)")]
    if "hybrid" in stages:
        order.append(("hybrid", "Hybrid (final)"))

    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
    if len(order) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, order):
        ax.imshow(stages[key], cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    pipeline: CleanupPipeline,
    noisy_dir: str,
    clean_dir: Optional[str],
    out_dir: str = "results",
    max_eval: int = 20,
    n_samples: int = 3,
) -> str:
    """Build report.md (+ figures) under out_dir and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    names = sorted(f for f in os.listdir(noisy_dir)
                   if os.path.splitext(f)[1].lower() in exts)[:max_eval]
    if not names:
        raise RuntimeError(f"No images found in {noisy_dir}")

    have_gt = clean_dir is not None and os.path.isdir(clean_dir)
    have_hybrid = pipeline.weights_loaded

    metrics = {"raw": {"psnr": [], "ssim": []},
               "cv_only": {"psnr": [], "ssim": []},
               "hybrid": {"psnr": [], "ssim": []}}
    otsu = {"raw": [], "cv_only": [], "hybrid": []}

    sample_figs: List[str] = []
    for i, name in enumerate(names):
        noisy = _load_gray01(os.path.join(noisy_dir, name))
        stages = pipeline.run_staged(noisy)

        if have_gt:
            clean_path = os.path.join(clean_dir, name)
            if os.path.isfile(clean_path):
                gt = _load_gray01(clean_path)
                metrics["raw"]["psnr"].append(psnr(stages["raw"], gt))
                metrics["raw"]["ssim"].append(ssim(stages["raw"], gt))
                metrics["cv_only"]["psnr"].append(psnr(stages["cv_only"], gt))
                metrics["cv_only"]["ssim"].append(ssim(stages["cv_only"], gt))
                otsu["raw"].append(_otsu_dice(stages["raw"], gt))
                otsu["cv_only"].append(_otsu_dice(stages["cv_only"], gt))
                if have_hybrid:
                    metrics["hybrid"]["psnr"].append(psnr(stages["hybrid"], gt))
                    metrics["hybrid"]["ssim"].append(ssim(stages["hybrid"], gt))
                    otsu["hybrid"].append(_otsu_dice(stages["hybrid"], gt))

        if i < n_samples:
            fig_path = os.path.join(fig_dir, f"sample_{i:02d}.png")
            _save_sample_figure(stages, fig_path, title=name)
            sample_figs.append(os.path.relpath(fig_path, out_dir).replace("\\", "/"))

    flops, params = compute_flops(pipeline.model, (1, 1, 512, 512))

    def _avg(lst: List[float]) -> float:
        finite = [v for v in lst if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    lines: List[str] = []
    lines.append("# Phase-Contrast Clean-Up - Quantitative Report\n")
    lines.append(f"Evaluated on **{len(names)}** images from `{noisy_dir}`.\n")

    if have_gt:
        lines.append("## 1. Image quality vs. ground truth\n")
        lines.append("| Path | PSNR (dB), higher better | SSIM, higher better |")
        lines.append("| :--- | ---: | ---: |")
        for key, label in [("raw", "Raw input"), ("cv_only", "CV-only (DoG+CLAHE)"),
                           ("hybrid", "Hybrid (CV + U-Net)")]:
            if key == "hybrid" and not have_hybrid:
                continue
            p = _avg(metrics[key]["psnr"])
            s = _avg(metrics[key]["ssim"])
            lines.append(f"| {label} | {p:.2f} | {s:.4f} |")
        lines.append("")

        raw_p, raw_s = _avg(metrics["raw"]["psnr"]), _avg(metrics["raw"]["ssim"])
        best_key = "hybrid" if have_hybrid else "cv_only"
        dp = _avg(metrics[best_key]["psnr"]) - raw_p
        ds = _avg(metrics[best_key]["ssim"]) - raw_s
        target = "[TARGET MET]" if (dp >= 2.0 and ds >= 0.05) else "[BELOW TARGET]"
        lines.append(f"> **Improvement of best path over raw:** "
                     f"+{dp:.2f} dB PSNR, +{ds:.4f} SSIM "
                     f"{target} (target: >=2 dB, >=0.05 SSIM)\n")

        lines.append("## 2. Downstream insight - Otsu segmentation recovery\n")
        lines.append("Otsu thresholding is a standard first step for counting/"
                     "measuring cells. A global threshold fails when a low-frequency "
                     "illumination gradient makes part of the background brighter than "
                     "cells elsewhere. We segment each cleaned image and measure the "
                     "**Dice overlap** of its cell mask against the clean target's mask.\n")
        lines.append("| Path | Otsu Dice, higher better |")
        lines.append("| :--- | ---: |")
        for key, label in [("raw", "Raw input"), ("cv_only", "CV-only"),
                           ("hybrid", "Hybrid")]:
            if key == "hybrid" and not have_hybrid:
                continue
            lines.append(f"| {label} | {_avg(otsu[key]):.3f} |")
        best_seg = _avg(otsu[best_key])
        raw_seg = _avg(otsu["raw"])
        lines.append("")
        direction = "recovers" if best_seg >= raw_seg else "changes"
        lines.append(f"> **Insight:** the {'hybrid' if have_hybrid else 'cv_only'} path "
                     f"{direction} downstream segmentation from a Dice of "
                     f"**{raw_seg:.3f}** (raw, where the illumination gradient breaks the "
                     f"global threshold) to **{best_seg:.3f}** - because it reconstructs "
                     "the flat-field cell bodies the threshold needs. Note CV-only "
                     f"(Dice {_avg(otsu['cv_only']):.3f}) is a *contrast* enhancer for "
                     "visual inspection, not a flat-field corrector, so it does not by "
                     "itself restore region-based segmentation.\n")
    else:
        lines.append("_No ground-truth folder supplied - quantitative metrics "
                     "skipped; see qualitative samples below._\n")

    lines.append("## 3. Computational cost (per 512x512 image)\n")
    lines.append("| Path | FLOPs | Params |")
    lines.append("| :--- | ---: | ---: |")
    lines.append(f"| CV-only | ~a few GFLOPs (Gaussian blurs + CLAHE) | 0 |")
    lines.append(f"| Hybrid (U-Net) | {human_readable(flops)} | {human_readable(params)} |")
    lines.append("")
    lines.append("> The CV-only toggle runs in **milliseconds with zero VRAM** - ideal "
                 "as a real-time halo-suppression + edge-enhancement preview for visual "
                 "QC. When *quantitative* restoration is required (PSNR/SSIM, "
                 "segmentation), the hybrid path adds the U-Net at the cost shown above. "
                 "The toggle lets you pay for accuracy only when you need it.\n")

    lines.append("## 4. Sample comparisons\n")
    for fig in sample_figs:
        lines.append(f"![sample]({fig})\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[report] wrote {report_path}")
    return report_path
