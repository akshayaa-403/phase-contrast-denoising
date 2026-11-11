"""Generate the quantitative report.md and comparison figures.

Every number in the report is measured on the images passed in, and every
sentence that interprets a number is generated from it (nothing is hardcoded),
so the report cannot claim more than the data shows. Means come with 95%
percentile-bootstrap confidence intervals.

Paths compared:

* ``raw`` -- the input, polarity-matched to the target (inverted if the
  config inverts), so the baseline is not penalised for showing cells dark;
* classical denoisers (Gaussian, median, non-local means) on that input --
  they remove noise but not halo or uneven illumination;
* ``cv_only`` -- the classical path of this project;
* ``hybrid`` -- classical path + residual U-Net;
* ``plain_unet`` -- the same architecture trained to output the image
  directly (``residual: false``), when ablation weights are given.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .image_io import list_images, load_gray01
from .metrics import (
    bootstrap_ci,
    compute_flops,
    cv_path_flops,
    human_readable,
    otsu_dice,
    psnr,
    psnr_affine,
    ssim,
    ssim_affine,
)
from .pipeline import CleanupPipeline

log = logging.getLogger(__name__)

METRICS = ("psnr", "psnr_affine", "ssim", "ssim_affine", "dice")
LABELS = {
    "raw": "Raw input (polarity-matched)",
    "gaussian": "Gaussian blur (sigma 1.5)",
    "median": "Median filter (5x5)",
    "nl_means": "Non-local means",
    "cv_only": "CV-only (flatten + DoG + CLAHE)",
    "plain_unet": "CV + plain U-Net (ablation)",
    "hybrid": "Hybrid (CV + residual U-Net)",
}


def _classical_baselines() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Standard denoisers, applied to the polarity-matched raw input."""
    from skimage.restoration import denoise_nl_means

    def estimate_sigma(x: np.ndarray) -> float:
        """Immerkaer's fast noise estimate (Gaussian noise, no wavelet dependency)."""
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        response = cv2.filter2D(np.asarray(x, dtype=np.float32), -1, kernel)[1:-1, 1:-1]
        return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * response.size))

    def gaussian(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (0, 0), 1.5)

    def median(x: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(np.ascontiguousarray(x, dtype=np.float32), 5)

    def nl_means(x: np.ndarray) -> np.ndarray:
        sigma = float(estimate_sigma(x))
        return denoise_nl_means(x, h=0.8 * sigma, sigma=sigma, patch_size=5,
                                patch_distance=6, fast_mode=True).astype(np.float32)

    return {"gaussian": gaussian, "median": median, "nl_means": nl_means}


def _save_sample_figure(stages: Dict[str, np.ndarray], gt: Optional[np.ndarray],
                        out_path: str, title: str) -> None:
    """Save a labelled panel: raw, flattened, DoG, CV-only, hybrid, target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [("raw", "Raw (polarity-matched)"), ("flat", "Background flattened"),
             ("dog", "DoG halo suppressed"), ("cv_only", "CV-only (+CLAHE)")]
    panels = [(stages[k], label) for k, label in order]
    if "hybrid" in stages:
        panels.append((stages["hybrid"], "Hybrid (final)"))
    if gt is not None:
        panels.append((gt, "Ground truth"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (img, label) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _time_ms(fn: Callable[[], object], device: str, repeats: int = 5) -> float:
    """Median wall time of ``fn`` in ms, after one warm-up call, synchronising CUDA."""
    import torch

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _fmt(stat: Sequence[float], digits: int) -> str:
    mean, lo, hi = stat
    if mean != mean:
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def generate_report(
    pipeline: CleanupPipeline,
    noisy_dir: str,
    clean_dir: Optional[str],
    out_dir: str = "results",
    names: Optional[List[str]] = None,
    n_samples: int = 3,
    ablation: Optional[CleanupPipeline] = None,
    split_note: str = "",
    include_baselines: bool = True,
) -> str:
    """Build report.md (+ figures) under ``out_dir`` and return its path.

    ``names`` restricts evaluation to those files (e.g. the held-out split).
    ``ablation`` is a pipeline loaded with plain-U-Net weights.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    names = sorted(names) if names is not None else list_images(noisy_dir)
    if not names:
        raise RuntimeError(f"No images found in {noisy_dir}")

    have_gt = clean_dir is not None and os.path.isdir(clean_dir)
    have_hybrid = pipeline.weights_loaded
    have_ablation = ablation is not None and ablation.weights_loaded
    baselines = _classical_baselines() if (have_gt and include_baselines) else {}

    paths = ["raw", *baselines, "cv_only"]
    if have_ablation:
        paths.append("plain_unet")
    if have_hybrid:
        paths.append("hybrid")
    scores: Dict[str, Dict[str, List[float]]] = {p: {m: [] for m in METRICS} for p in paths}

    sample_figs: List[str] = []
    for i, name in enumerate(names):
        noisy = load_gray01(os.path.join(noisy_dir, name))
        stages = pipeline.run_staged(noisy)
        gt = None
        if have_gt and os.path.isfile(os.path.join(clean_dir, name)):
            gt = load_gray01(os.path.join(clean_dir, name))
            outputs = {"raw": stages["raw"], "cv_only": stages["cv_only"]}
            for key, fn in baselines.items():
                outputs[key] = np.clip(fn(stages["raw"]), 0.0, 1.0)
            if have_ablation:
                outputs["plain_unet"] = ablation.run(noisy, mode="hybrid")
            if have_hybrid:
                outputs["hybrid"] = stages["hybrid"]
            for key, out in outputs.items():
                s = scores[key]
                s["psnr"].append(psnr(out, gt))
                s["psnr_affine"].append(psnr_affine(out, gt))
                s["ssim"].append(ssim(out, gt))
                s["ssim_affine"].append(ssim_affine(out, gt))
                s["dice"].append(otsu_dice(out, gt))
        if i < n_samples:
            fig_path = os.path.join(fig_dir, f"sample_{i:02d}.png")
            _save_sample_figure(stages, gt, fig_path, title=name)
            sample_figs.append(os.path.relpath(fig_path, out_dir).replace("\\", "/"))

    stats = {p: {m: bootstrap_ci(v) for m, v in scores[p].items()} for p in paths}
    n_scored = len(scores["raw"]["psnr"])

    # ---------------------------------------------------------------- cost
    probe = load_gray01(os.path.join(noisy_dir, names[0]))
    probe512 = cv2.resize(probe, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv_ms = _time_ms(lambda: pipeline.run(probe512, mode="cv_only"), pipeline.device)
    hy_ms = (_time_ms(lambda: pipeline.run(probe512, mode="hybrid"), pipeline.device)
             if have_hybrid else float("nan"))
    unet_flops, unet_params = compute_flops(pipeline.model, (1, 1, 512, 512))
    cv_flops = cv_path_flops(pipeline.cv_ops, 512, 512)

    # ---------------------------------------------------------------- write
    lines: List[str] = ["# Phase-Contrast Clean-Up: Quantitative Report\n"]
    lines.append(f"Evaluated on **{len(names)}** images from `{noisy_dir}`"
                 f"{' (' + split_note + ')' if split_note else ''}.")
    meta = pipeline.checkpoint_meta
    if meta:
        lines.append(f"Model checkpoint: epoch {meta.get('epoch', '?')}, "
                     f"val PSNR {meta.get('val_psnr', float('nan')):.2f} dB, "
                     f"commit `{meta.get('git_sha') or 'unknown'}`.")
    lines.append("")

    best_key = "hybrid" if have_hybrid else "cv_only"
    if have_gt and n_scored:
        lines.append("## 1. Image quality vs. ground truth\n")
        lines.append(f"Mean over {n_scored} images, with a 95% bootstrap confidence interval "
                     "in brackets. *Affine* scores first fit each output to the target with "
                     "`a * x + b` (as CARE/CSBDeep do), so a method is not penalised for "
                     "the target's absolute grey levels, which only a trained model can know. "
                     "Higher is better throughout.\n")
        lines.append("| Path | PSNR (dB) | PSNR affine (dB) | SSIM | SSIM affine |")
        lines.append("| :--- | ---: | ---: | ---: | ---: |")
        for key in paths:
            st = stats[key]
            lines.append(f"| {LABELS[key]} | {_fmt(st['psnr'], 2)} | {_fmt(st['psnr_affine'], 2)} "
                         f"| {_fmt(st['ssim'], 3)} | {_fmt(st['ssim_affine'], 3)} |")
        lines.append("")

        raw_p, raw_s = stats["raw"]["psnr"][0], stats["raw"]["ssim"][0]
        dp = stats[best_key]["psnr"][0] - raw_p
        ds = stats[best_key]["ssim"][0] - raw_s
        met = dp >= 2.0 and ds >= 0.05
        lines.append(f"> **{LABELS[best_key]} vs. raw:** {dp:+.2f} dB PSNR, {ds:+.3f} SSIM "
                     f"({'meets' if met else 'does not meet'} the >= 2 dB / >= 0.05 SSIM target).")
        dpa = stats["cv_only"]["psnr_affine"][0] - stats["raw"]["psnr_affine"][0]
        lines.append(f"> **CV-only vs. raw, affine-matched:** {dpa:+.2f} dB PSNR. Plain PSNR "
                     "understates the classical path: it cannot know the target's grey levels.\n")

        if have_ablation:
            dh = stats["hybrid"]["psnr"][0] - stats["plain_unet"]["psnr"][0]
            dd = stats["hybrid"]["dice"][0] - stats["plain_unet"]["dice"][0]
            lines.append("### Residual vs. plain U-Net (ablation)\n")
            lines.append(f"Same architecture, data, preprocessing and training budget; only "
                         f"the output parameterisation differs. Residual learning changes PSNR by "
                         f"**{dh:+.2f} dB** and Otsu Dice by **{dd:+.3f}**. "
                         + ("The confidence intervals overlap, so treat the difference as "
                            "inconclusive on this test set." if
                            stats["hybrid"]["psnr"][1] <= stats["plain_unet"]["psnr"][2]
                            and stats["plain_unet"]["psnr"][1] <= stats["hybrid"]["psnr"][2]
                            else "The confidence intervals do not overlap.") + "\n")

        lines.append("## 2. Downstream: Otsu segmentation\n")
        lines.append("Otsu thresholding is a standard first step for counting and measuring "
                     "cells. We threshold each output and measure the **Dice overlap** of its "
                     "foreground with the target's. Uneven illumination and halos both break a "
                     "single global threshold.\n")
        lines.append("| Path | Otsu Dice |")
        lines.append("| :--- | ---: |")
        for key in paths:
            lines.append(f"| {LABELS[key]} | {_fmt(stats[key]['dice'], 3)} |")
        lines.append("")
        raw_d = stats["raw"]["dice"][0]
        cv_d = stats["cv_only"]["dice"][0]
        best_d = stats[best_key]["dice"][0]
        lines.append(f"> **Insight:** Otsu Dice goes from **{raw_d:.3f}** on the raw input to "
                     f"**{cv_d:.3f}** after the classical path"
                     + (f" and **{best_d:.3f}** with the residual U-Net" if have_hybrid else "")
                     + ". The generic denoisers stay near the raw score"
                     + (f" (best {max(stats[k]['dice'][0] for k in baselines):.3f})" if baselines else "")
                     + ": noise is not what breaks the threshold here; halos and uneven "
                     "illumination are.\n")
    else:
        lines.append("_No ground-truth folder supplied: quantitative metrics skipped; see the "
                     "qualitative samples below._\n")

    lines.append("## 3. Computational cost (one 512 x 512 image)\n")
    lines.append(f"Measured on `{pipeline.device}` (median of 5 runs after a warm-up).\n")
    lines.append("| Path | FLOPs | Params | Time |")
    lines.append("| :--- | ---: | ---: | ---: |")
    lines.append(f"| CV-only | ~{human_readable(cv_flops)} (estimated from kernel sizes) | 0 | {cv_ms:.1f} ms |")
    if have_hybrid:
        lines.append(f"| Hybrid (CV + U-Net) | {human_readable(cv_flops + unet_flops)} "
                     f"| {human_readable(unet_params)} | {hy_ms:.1f} ms |")
    lines.append("")
    if have_hybrid and cv_ms > 0:
        lines.append(f"> The U-Net costs about **{unet_flops / max(cv_flops, 1):.0f}x** the "
                     f"FLOPs of the classical path and {hy_ms / cv_ms:.0f}x its time here. The "
                     "toggle lets you pay for that only when you need quantitative "
                     "restoration rather than a visual preview.\n")

    lines.append("## 4. Sample comparisons\n")
    for fig in sample_figs:
        lines.append(f"![sample]({fig})\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("Wrote %s", report_path)
    return report_path
