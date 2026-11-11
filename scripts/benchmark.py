"""Measure inference latency and memory against the performance budget.

    python scripts/benchmark.py                 # 512x512, both modes, current device
    python scripts/benchmark.py --size 1024 --repeats 20

Budget (per 512 x 512 image): CV-only on CPU < 50 ms with no GPU memory;
hybrid on a T4-class GPU < 300 ms and < 2 GB VRAM. Times are the median of
``--repeats`` runs after warm-up, with CUDA synchronised around each run. On a
CPU-only machine the hybrid row is reported but the GPU budget does not apply.
Exits non-zero if a budget that applies to this machine is missed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_cleanup.pipeline import CleanupPipeline, load_yaml  # noqa: E402

BUDGET_MS = {"cv_only": 50.0, "hybrid": 300.0}
BUDGET_VRAM_MB = 2048.0


def time_mode(pipe: CleanupPipeline, img: np.ndarray, mode: str, repeats: int) -> float:
    cuda = pipe.device.startswith("cuda")
    for _ in range(2):  # warm-up: allocator, cuDNN autotune, thread pools
        pipe.run(img, mode)
    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.run(img, mode)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default_config.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    weights = args.weights or str(REPO_ROOT / load_yaml(args.config)["paths"]["pretrained"])
    pipe = CleanupPipeline(args.config, model_weights=weights if Path(weights).is_file() else None)
    img = np.random.default_rng(0).random((args.size, args.size)).astype(np.float32)
    cuda = pipe.device.startswith("cuda")
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    print(f"device={pipe.device} gpu={gpu} threads={torch.get_num_threads()} image={args.size}x{args.size}")

    failures = []
    modes = ["cv_only"] + (["hybrid"] if pipe.weights_loaded else [])
    for mode in modes:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        ms = time_mode(pipe, img, mode, args.repeats)
        vram = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
        applies = args.size == 512 and (mode == "cv_only" or cuda)
        ok = ms < BUDGET_MS[mode] and (mode == "cv_only" or vram < BUDGET_VRAM_MB)
        verdict = ("PASS" if ok else "FAIL") if applies else "n/a (budget is for 512 px on a GPU)"
        print(f"  {mode:<8} {ms:8.1f} ms   peak VRAM {vram:7.1f} MB   budget {BUDGET_MS[mode]:.0f} ms: {verdict}")
        if applies and not ok:
            failures.append(mode)
    if not pipe.weights_loaded:
        print("  hybrid   skipped (no weights; run scripts/download_weights.py)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
