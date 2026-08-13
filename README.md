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
browser** — drag sigma1, sigma2, alpha, the CLAHE clip limit and tile count and
watch both the image and the metrics update. No server, no install. Upload your
own micrograph and it runs on that instead.

```bash
python data/generate_synthetic.py --num 100 --size 256   # if data/ is empty
python scripts/make_demo_assets.py --count 6             # writes docs/samples/
python -m http.server -d docs 8000                       # preview
```

Enable it once under **Settings -> Pages -> Source: GitHub Actions**; pushes
touching `docs/` then deploy via `.github/workflows/pages.yml`.

The page is deliberately thin on prose: results first, one paragraph of context,
and the method notes below rather than on screen.

### Is the JavaScript port trustworthy?

`docs/cv.js` is a hand port of `src/preprocessor.py` (OpenCV's Gaussian kernel
sizing, `BORDER_REFLECT_101`, and the tile-histogram CLAHE from `clahe.cpp`),
which is only worth believing if it is checked:

```bash
python scripts/verify_web_pipeline.py
```

That runs both implementations over the same images and fails if they diverge.
Current measurements: pipeline output agrees to **~50 dB PSNR** (max per-pixel
difference ~0.016 of full scale, from tile-boundary interpolation), and the
JavaScript PSNR/SSIM match scikit-image **exactly** to five decimals.

### How the classical path works

The halo is a broad, low-frequency ring while the cell body carries the
high-frequency detail, so blurring at two scales separates them: sigma1 keeps the
cell, sigma2 is set wide enough to straddle the halo, and subtracting a fraction
alpha of their difference removes the ring without hollowing the cell out. CLAHE
then restores local contrast in a way plain histogram equalisation cannot,
because its per-tile clip limit stops background noise being amplified into the
foreground.

### Why the metrics look the way they do

The classical output usually scores **worse** PSNR than the untouched input,
while the U-Net scores far better. That is not a bug. PSNR compares absolute
intensities and CLAHE deliberately rewrites the global tone curve, so an image
that is visibly cleaner can move further from the target in absolute terms. The
residual U-Net is trained against that target, so it stays on the intensity scale
PSNR rewards. It is a good demonstration of why one number is a poor proxy for
"does this help my segmentation".

### Uploads have no ground truth

PSNR and SSIM need a paired clean image, which exists only for the synthetic
samples. For an uploaded image the page switches to two **no-reference** measures
that need no target:

| Measure | Meaning | Direction |
|---|---|---|
| Edge energy | Mean squared Sobel gradient - how much membrane detail is present | higher is better |
| Background variance | Variance of the flattest quarter of an 8x8 tiling, i.e. empty space | lower is better |

Read together they answer "did the halo go", which is a weaker claim than
fidelity to a target - and the page says so.

### What is not run in the browser

The residual U-Net. At 31M parameters it is a ~124MB float32 download and far too
slow for WebAssembly, so its output is precomputed by
`scripts/make_demo_assets.py` and served as an image, labelled as precomputed on
the page. Run it yourself with `python main.py --mode hybrid`.

### Accessibility

Light and dark themes (following the OS preference until you pick one), a skip
link, visible focus rings, labelled controls with live-updating `<output>`
read-outs, table row and column headers, `aria-live` status messages, and support
for `prefers-reduced-motion` and `prefers-contrast: more`. Both palettes meet
WCAG AA contrast for body and secondary text.

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
docs/cv.js                    DoG + CLAHE + PSNR/SSIM + no-reference measures in JS
docs/app.js                   page wiring: samples, sliders, uploads, metrics
docs/theme.js                 light/dark theme, follows the OS until you choose
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
