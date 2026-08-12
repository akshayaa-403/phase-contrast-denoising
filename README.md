# Phase-Contrast Microscopy Clean-Up Pipeline

Suppresses the halo artifact in phase-contrast microscopy and enhances cell
contrast. Two selectable modes:

- `cv_only`: OpenCV only (Difference-of-Gaussians halo suppression + CLAHE). Fast, no model.
- `hybrid`: the CV front-end plus a residual U-Net that predicts and subtracts the leftover artifact (`cleaned = input - residual`).

## Requirements

Python 3.9+. CPU works; CUDA is used automatically if present.

```bash
pip install -r requirements.txt
```

A pre-trained fp16 model ships at `models/pretrained_fp16.pth`, so hybrid mode
works without training.

## Usage

Generate the synthetic dataset:

```bash
python data/generate_synthetic.py --num 100 --size 256
```

Hybrid inference (writes cleaned images + `results/report.md`):

```bash
python main.py --input_dir data/noisy_input --output_dir results --mode hybrid --weights models/pretrained_fp16.pth --clean_dir data/ground_truth
```

CV-only inference:

```bash
python main.py --input_dir data/noisy_input --output_dir results_cv --mode cv_only --clean_dir data/ground_truth
```

Train (writes `runs/best.pth` + TensorBoard logs):

```bash
python main.py --train --data_dir data
```

Run on your own images (inference only, omit `--clean_dir`):

```bash
python main.py --input_dir data/raw --output_dir results_real --mode hybrid --weights models/pretrained_fp16.pth
```

Live demo (local Gradio app):

```bash
python app.py --weights models/pretrained_fp16.pth
```

## Browser demo (GitHub Pages)

`docs/` is a static page that runs the **classical front-end live in the
browser** — drag σ₁, σ₂, α, the CLAHE clip limit and tile count and watch both
the image and the PSNR/SSIM update. No server, no install.

```bash
python data/generate_synthetic.py --num 100 --size 256   # if data/ is empty
python scripts/make_demo_assets.py --count 6             # writes docs/samples/
python -m http.server -d docs 8000                       # preview
```

Enable it once under **Settings → Pages → Source: GitHub Actions**; pushes
touching `docs/` then deploy via `.github/workflows/pages.yml`.

`docs/cv.js` is a hand port of `src/preprocessor.py` (OpenCV's Gaussian kernel
sizing, `BORDER_REFLECT_101`, and the tile-histogram CLAHE from `clahe.cpp`),
which is only worth trusting if it is checked:

```bash
python scripts/verify_web_pipeline.py
```

That runs both implementations over the same images and fails if they diverge.
Current measurements: pipeline output agrees to **~50 dB PSNR** (max per-pixel
difference ≈ 0.016 of full scale, from tile-boundary interpolation), and the
JavaScript PSNR/SSIM match scikit-image **exactly** to five decimals.

The residual U-Net is **not** run in the browser — 31M parameters is a ~124MB
float32 download and far too slow for WebAssembly. Its output is precomputed
into `docs/samples/*_hybrid.png` so the page can still show the comparison
honestly, labelled as precomputed.

One thing the demo makes obvious: the classical output usually scores *worse*
PSNR than the untouched input, while the U-Net scores far better. PSNR compares
absolute intensities and CLAHE deliberately rewrites the tone curve, so a
visibly cleaner image can move further from the target — a good reminder that
one number is a poor proxy for downstream segmentation quality.

Tests:

```bash
python tests/test_preprocessor.py
```

Compress a checkpoint to fp16:

```bash
python scripts/compress_checkpoint.py --src runs/best.pth --dst models/pretrained_fp16.pth
```

## Layout

```
configs/default_config.yaml   all CV + DL hyperparameters
data/generate_synthetic.py    creates paired (clean, haloed) images
data/raw/                     drop your own images here
models/pretrained_fp16.pth    shipped pre-trained model
scripts/compress_checkpoint.py  fp32 -> fp16 checkpoint
scripts/make_demo_assets.py   builds docs/samples/ for the browser demo
scripts/verify_web_pipeline.py checks docs/cv.js against OpenCV
docs/                         the GitHub Pages demo (static)
docs/cv.js                    DoG + CLAHE + PSNR/SSIM ported to JavaScript
src/preprocessor.py           OpenCV DoG + CLAHE
src/residual_unet.py          residual U-Net
src/dataset.py                paired DataLoader + augmentations
src/trainer.py                training loop + TensorBoard
src/metrics.py                PSNR, SSIM, FLOPs
src/pipeline.py               cv_only / hybrid toggle
src/report.py                 report.md + figures
tests/test_preprocessor.py    unit + integration tests
app.py                        Gradio demo
main.py                       CLI
```

## Configuration

Edit `configs/default_config.yaml`. Key parameters:

- `preprocessor.dog_sigma2`: large Gaussian scale; set to about the halo width in pixels.
- `preprocessor.dog_alpha`: halo subtraction strength.
- `preprocessor.clahe_clip_limit`: local contrast cap; lower to reduce background noise.
- `data.patch_size`: training crop (256 on GPU, 128 on CPU).
- `model.base_channels`: U-Net width (64 default).
