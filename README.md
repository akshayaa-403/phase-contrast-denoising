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

`docs/` is a static workbench that runs the **classical front-end live in the
browser**, beside the residual U-Net's saved output. No server, no install, no
build step. It reads top to bottom in four numbered steps:

1. **Pick an image**: one of the held-out samples, or upload your own (PNG, JPG
   or TIFF; 16-bit PNG/TIFF keep full precision via `docs/decode.js`). Nothing
   leaves the tab.
2. **Compare**: Before, Cleaned (classical, live), Cleaned + neural network
   (precomputed) and Ideal (ground truth), side by side.
3. **Adjust**: two sliders, *Halo size* (σ₂) and *Strength* (α). σ₁ and the
   CLAHE tile count sit under "More settings". The clip limit is not exposed:
   at 16-bit CLAHE on 256 px frames it has no effect (see below).
4. **How close to ideal?**: PSNR and SSIM against the ground truth, each with
   its change from Before. Uploads have no ground truth, so they get edge
   sharpness and background unevenness instead.

A collapsed "How it works" note explains the three stages in plain words.

```bash
python data/generate_synthetic.py --num 100 --size 256   # if data/ is empty
python scripts/make_demo_assets.py                       # writes docs/samples/
python -m http.server -d docs 8000                       # preview
```

Enable it once under **Settings -> Pages -> Source: GitHub Actions**; pushes
touching `docs/` then deploy via `.github/workflows/pages.yml`.

### How samples are chosen

The samples are the **20 frames the U-Net never trained on**: the held-out
split that `src.dataset.split_pairs` defines (seed 42, 20% of the 100 synthetic
frames). `build_dataloaders` and `scripts/make_demo_assets.py` both call that
function, so the demo and the trainer cannot disagree, and the script asserts
that it still reproduces the original training split. All 20 ship (about
2.9 MB) and appear as thumbnails; the page opens on the first. `make_demo_assets.py` loads images exactly as `main.py` does, and it is
deterministic: two runs give byte-identical output.

### CLAHE precision and the clip limit

The shipped code quantises to **16-bit** before CLAHE. OpenCV turns the clip
limit into a pixel ceiling per grey level, `max(int(clip × tileArea / bins),
1)`, truncated. With 65,536 bins and the default 8×8 tiles on a 256 px image,
the tile area is 1,024 px, so the ceiling is 1 for every clip value the slider
allows and the output is bit-identical. **The clip limit does nothing there.**
It only bites with coarse tiles or large images: a 2048 × 2048 upload at 8×8
tiles has 256 × 256 px tiles, so its ceiling reaches 2 from clip 2.0.

The page runs **16-bit (as shipped)** only, matching the Python code and the
input the U-Net was trained on, so it leaves the clip limit out rather than
show a slider that does nothing. In Python, 8-bit is the opt-in
`Preprocessor(clahe_dtype="uint8")`, where clip works as intended; the
defaults, the config and the CLI are unchanged. On the held-out frames the
8-bit path keeps mean Otsu Dice at 0.875, against 0.480 for the 16-bit path.

`node scripts/check_controls.js` sweeps every control across its range. It
fails if any parameter stops changing the output, if the 16-bit clip control
starts working (a reminder to update this note), or if `PhaseCV.claheCeiling`
disagrees with what the pipeline actually does.

### Is the JavaScript port trustworthy?

`docs/cv.js` is a hand port of `src/preprocessor.py`. It reproduces OpenCV's
Gaussian kernel construction and float32 row and column filter arithmetic,
`BORDER_REFLECT_101`, and the tile-histogram CLAHE from `clahe.cpp` (with its
truncated clip ceiling and round-half-to-even interpolation). It also matches
NumPy's float32 quantisation, and Otsu's threshold as `cv2.threshold`
computes it. That is only worth believing if it is checked:

```bash
python scripts/verify_web_pipeline.py
```

That runs both implementations over all 20 samples at both precisions and
fails if they diverge. Current measurements: at the config defaults the
outputs are **bit-identical** at both precisions (infinite PSNR). PSNR and SSIM
match scikit-image exactly, and Otsu Dice matches `src/report.py` exactly, both
on identical inputs and end to end. One exception: blurs with σ ≤ 0.5 use
OpenCV's 3- and 5-tap kernels, whose row filter still differs by up to one
float32 ulp.

### How the classical path works

The halo is a broad, low-frequency ring while the cell body carries the
high-frequency detail, so blurring at two scales separates them: sigma1 keeps the
cell, sigma2 is set wide enough to straddle the halo, and subtracting a fraction
alpha of their difference removes the ring without hollowing the cell out. CLAHE
then restores local contrast in a way plain histogram equalisation cannot,
because its per-tile clip limit stops background noise being amplified into the
foreground (when the ceiling lets it bite; see above).

### Why the metrics look the way they do

The classical output usually scores **worse** PSNR than the untouched input,
while the U-Net scores far better. That is not a bug. PSNR compares absolute
intensities and CLAHE deliberately rewrites the global tone curve, so an image
that is visibly cleaner can move further from the target in absolute terms. The
residual U-Net is trained against that target, so it stays on the intensity scale
PSNR rewards. It is a good demonstration of why one number is a poor proxy for
"does this help my segmentation". On the held-out frames the U-Net lifts mean
PSNR from 11.46 to 20.26 dB, yet mean Otsu Dice barely moves (0.894 → 0.907).
What changes is the worst frame, which stops failing (0.556 → 0.808).

### Uploads have no ground truth

PSNR, SSIM and Dice need a paired clean image, which exists only for the
synthetic samples. For an uploaded image the page switches to **no-reference**
measures that need no target:

| Measure | Meaning | Direction |
|---|---|---|
| Edge energy | Mean squared Sobel gradient - how much membrane detail is present | higher is better |
| Background variance | Variance of the flattest quarter of an 8x8 tiling, i.e. empty space | lower is better |
| Otsu foreground | Share of pixels a global Otsu threshold calls cells, before and after | context, not a score |

Read together they answer "did the halo go", which is a weaker claim than
fidelity to a target - and the page says so.

### What is not run in the browser

The residual U-Net. At 31M parameters it is a ~124MB float32 download and far too
slow for WebAssembly, so its output (and its residual) is precomputed by
`scripts/make_demo_assets.py` and served as images, labelled as precomputed on
the page. Run it yourself with `python main.py --mode hybrid`.

### Accessibility

- **Themes.** Light and dark, following the OS preference until you pick one.
  The viewer keeps a dark surround in both, because micrographs are read on
  dark.
- **Keyboard.** Skip links, landmarks and visible focus rings throughout. The
  split divider, the profile row and the zoom all work from the keyboard, and
  the sample strip is a single tab stop (arrow keys move within it).
- **Labels and tables.** Controls are labelled and have exact number inputs;
  tables have row and column headers.
- **Screen readers.** An `aria-live` status announces results once a drag
  ends, not on every frame.
- **Colour.** Stage identities use colour plus a second cue (a label or a dash
  pattern). Every colour token is measured against WCAG AA in both themes:
  4.5:1 for text, 3:1 for overlays and UI marks.
- **Preferences.** `prefers-reduced-motion` and `prefers-contrast: more` are
  honoured.

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
scripts/make_demo_assets.py   builds docs/samples/ (the held-out frames) for the browser demo
scripts/verify_web_pipeline.py checks docs/cv.js against OpenCV at both CLAHE precisions
scripts/check_controls.js     checks that every demo control moves the output (Node, no deps)
docs/                         the GitHub Pages demo (static)
docs/cv.js                    DoG + CLAHE + PSNR/SSIM/Otsu Dice + no-reference measures in JS
docs/params.js                parameter ranges used by scripts/check_controls.js
docs/decode.js                8/16-bit PNG and TIFF decoding, no library
docs/app.js                   page wiring: samples, upload, sliders, scores
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
