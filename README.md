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

Live demo:

```bash
python app.py --weights models/pretrained_fp16.pth
```

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

See `DATA_AND_WORKFLOW.md` for data details and the full command sequence.
