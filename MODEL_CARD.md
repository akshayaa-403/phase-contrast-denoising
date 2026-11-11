# Model card: phase-contrast residual U-Net v1.0.0

## Summary

A 4-level U-Net (7.76 M parameters, base width 32, GroupNorm) that takes the
output of this project's classical path and predicts the residual artifact to
subtract. It is a research and portfolio model trained **only on synthetic
data**. It is not validated for any clinical or diagnostic use.

| | |
| :--- | :--- |
| Files | `pretrained_fp16.pth` (PyTorch, fp16 weights, 15.5 MB); `phase_unet.onnx` (fp32, 31.1 MB) |
| Distribution | GitHub release v1.0.0; `scripts/download_weights.py` checks SHA-256 |
| Input | Single-channel image after `Preprocessor.process` with the settings stored in the checkpoint, float32 in [0, 1], any size (padded to a multiple of 16; tiled above 1024 px) |
| Output | Residual map; cleaned image = `clip(input - residual, 0, 1)`, cells bright on a dark background |
| Licence | MIT |

## Training data

`data/generate_synthetic.py`, seed 42: 100 paired 256 x 256 frames (80 train,
20 validation). Each frame is a field of 6-15 elliptical cells (dome profile,
nucleus, granules) imaged through the linear phase-contrast model of Yin et
al. (2012) with an apodised obscured-Airy kernel (halo width 2.5-7 px),
multiplicative uneven illumination (tilt + vignetting), and Poisson-Gaussian
noise, saved as 16-bit PNG. The target is the cells' optical path difference.

## Training procedure

L1 loss on the residual, AdamW (lr 1e-4, weight decay 1e-5),
ReduceLROnPlateau on validation PSNR (patience 5), early stopping (patience
12), 128 px random crops with flips and 90-degree rotations, batch 16, CPU.
Best validation PSNR 28.97 dB at epoch 94. The checkpoint records the exact
preprocessing settings and git commit.

## Evaluation

| Set | Metric | Raw | CV-only | Hybrid |
| :--- | :--- | ---: | ---: | ---: |
| Synthetic test (30 frames, seed 1042) | PSNR (dB) | 1.46 | 15.05 | 28.65 [27.98, 29.27] |
| Synthetic test | Otsu Dice | 0.081 | 0.566 | 0.943 [0.937, 0.949] |
| Real: CTC PhC-C2DH-U373 (34 frames), native resolution | Otsu Dice vs manual mask | 0.099 | 0.607 | 0.601 |
| Real: same, rescaled 0.35 (chosen before scoring) | Otsu Dice vs manual mask | 0.095 | 0.353 | 0.634 [0.609, 0.658] |

Ablation: the same network trained to output the image directly (no residual)
scored 30.20 dB / Dice 0.953 on the synthetic test set, better than this
model. See the README.

## Limitations and known failure modes

- **Synthetic-to-real gap.** On real U373 cells the model finds cell locations
  but reproduces the smooth, convex shapes of its training cells, losing
  lamellipodia and irregular outlines. Its real-data Dice barely exceeds the
  classical path. Do not use its output for morphology measurements.
- **Scale.** Trained on cells ~15-50 px across. Rescale images so cells are in
  that range; at native resolution on larger cells it does not help.
- **Polarity.** Assumes positive phase contrast (cells darker than the
  background) unless `invert: false` is set, and the model was trained with
  `invert: true`.
- **Preprocessing coupling.** Changing classical-path settings (sliders, the
  config) feeds the network inputs it never saw. The pipeline warns when the
  config differs from the checkpoint, and the demo warns when the sliders move.
- **Hallucination.** Like any learned restoration, it can produce plausible
  but wrong structure. Prefer the classical path when results must be traceable
  step by step.
- **Performance.** 456 ms per 512 x 512 image on a 14-thread CPU. GPU latency
  and memory were not measured for this release.

## Intended use

Demonstrating and comparing classical and learned halo removal; a
pre-processing experiment ahead of segmentation on phase-contrast data whose
cells resemble the training distribution, with results checked against the
classical path.

## Citation

Yin, Kanade & Chen, *Understanding the phase contrast optics to restore
artifact-free microscopy images for segmentation*, Medical Image Analysis
16(5), 2012. Real-data evaluation: Ulman et al., *An objective comparison of
cell-tracking algorithms*, Nature Methods 14, 2017 (Cell Tracking Challenge).
