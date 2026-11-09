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
build step. The first screen shows a large before/after of a held-out frame and
one headline line read from the manifest; everything below it and to the side
is for tuning.

```bash
python data/generate_synthetic.py --num 100 --size 256   # if data/ is empty
python scripts/make_demo_assets.py                       # writes docs/samples/
python -m http.server -d docs 8000                       # preview
```

Enable it once under **Settings -> Pages -> Source: GitHub Actions**; pushes
touching `docs/` then deploy via `.github/workflows/pages.yml`.

The page is deliberately thin on prose: results first, one sentence of context,
and the method notes here rather than on screen.

### What the page does

- **Split viewer.** Two stages in one frame, divided by a draggable line (also
  a keyboard slider: arrows move it 1%, Shift+arrows 10%, Home/End to the
  edges). Each side has its own picker: Input, After DoG, Classical (live),
  Hybrid U-Net (precomputed) and Ground truth, plus derived views: *Removed by
  DoG*, *Error vs truth* and the *U-Net residual* (what the network subtracted,
  at the config defaults). A grid view shows every stage at the same zoom;
  clicking one puts it on the right of the split.
- **Zoom and pan**, in sync across stages: wheel or pinch around the cursor,
  drag to pan, double-click for fit/2×, and `+` `-` `0` `1` and the arrow keys
  when the viewer has focus. Magnifying goes in whole device-pixel steps with
  nearest-neighbour rendering, so pixels stay square and nothing is invented;
  minifying averages. Holding "hold for input", or `\`, shows the input
  across the whole frame.
- **Read-outs.** Hovering shows x, y and each visible stage's value. A
  draggable line profile plots one row for Input, the live stage, Hybrid and
  Truth; a halo is the overshoot either side of each cell edge. The Otsu-mask
  outline of the right-hand image (solid) and of the truth (dashed) can be
  overlaid. While σ₁ or σ₂ is dragged, circles of those radii are drawn, so
  the blur scales can be compared with the cells and the halo.
- **Controls**, grouped by stage. Each has a slider, an exact number box, the
  config default marked on the track, its own reset, and Shift for steps ten
  times larger. σ₂ cannot fall to σ₁ or below: its minimum follows σ₁. If a
  control moves while no live stage is on screen, the right side switches to
  Classical (live) and says so.
- **Measured.** PSNR, SSIM and Otsu Dice for Input, Classical (live) and
  Hybrid, each with its signed change from Input in words ("+10.5 dB,
  better"). Python's own numbers are shown while every parameter sits at its
  default, and "sliders moved" otherwise. Short expandable notes explain why
  the classical PSNR falls below the input's, what Dice measures, and why
  uploads get no-reference measures. Below the fold are the held-out summary
  and a dot plot of every frame, raw → U-Net.
- **Your own images**, by drag-and-drop, paste or file picker. PNG and TIFF are
  decoded in the page at full 8- or 16-bit precision (`docs/decode.js`: PNG,
  and uncompressed, LZW, Deflate or PackBits TIFF, with no library); anything
  else goes through the browser's decoder. Images over 2048 px offer a
  downscale or a centre crop. While a slider moves on a large image, a preview
  of at most 512 px is computed, then the full image on release. Nothing
  leaves the tab.
- **Export.** The right-hand image as a full-resolution PNG; a labelled
  side-by-side figure with the parameters; the YAML block for
  `configs/default_config.yaml`; a ready `python main.py …` command; and a
  link that stores the view (sample, parameters, precision, stages, split and
  overlays) in the URL hash. Uploads are never put in a link.

The pipeline runs in a Web Worker (`docs/worker.js`), so dragging never blocks
the page, and results whose parameters have since moved on are dropped.

### How samples are chosen

The samples are the **20 frames the U-Net never trained on**: the held-out
split that `src.dataset.split_pairs` defines (seed 42, 20% of the 100 synthetic
frames). `build_dataloaders` and `scripts/make_demo_assets.py` both call that
function, so the demo and the trainer cannot disagree, and the script asserts
that it still reproduces the original training split. The page opens on the
frame with the lowest raw Otsu Dice, labelled "worst raw frame". All 20 ship
(about 2.9 MB), so the held-out summary covers exactly the frames a visitor can
open. `make_demo_assets.py` loads images exactly as `main.py` does, and it is
deterministic: two runs give byte-identical output.

### CLAHE precision and the clip limit

The shipped code quantises to **16-bit** before CLAHE. OpenCV turns the clip
limit into a pixel ceiling per grey level, `max(int(clip × tileArea / bins),
1)`, truncated. With 65,536 bins and the default 8×8 tiles on a 256 px image,
the tile area is 1,024 px, so the ceiling is 1 for every clip value the slider
allows and the output is bit-identical. **The clip limit does nothing there.**
It only bites with coarse tiles or large images: a 2048 × 2048 upload at 8×8
tiles has 256 × 256 px tiles, so its ceiling reaches 2 from clip 2.0.

The page therefore offers a precision control. **16-bit (as shipped)** is the
default; it matches the Python code and the input the U-Net was trained on.
**8-bit (fixed)** uses 256 bins, where clip works as intended. Under the clip
slider the page computes the ceiling live, and it marks the control inactive
whenever it cannot change the output at the current tile size, image size and
precision. In Python, 8-bit is the opt-in `Preprocessor(clahe_dtype="uint8")`;
the defaults, the config and the CLI are unchanged. On the held-out frames the
8-bit path keeps mean Otsu Dice at 0.875, against 0.480 for the 16-bit path.

`node scripts/check_controls.js` sweeps every control across its range. It
fails if any parameter stops changing the output, if the 16-bit clip control
starts working (a reminder to update this note), or if the page's "inactive"
marker disagrees with what the pipeline actually does.

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
docs/params.js                the demo's parameters: ranges, steps, YAML keys
docs/decode.js                8/16-bit PNG and TIFF decoding, no library
docs/worker.js                runs cv.js off the main thread
docs/app.js                   page wiring: samples, controls, measurements, export, URL
docs/viewer.js                split/grid viewer: zoom, pan, overlays
docs/charts.js                line profile and held-out dot plot (inline SVG)
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
