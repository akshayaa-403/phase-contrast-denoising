###### *Ask Questions if clarity is needed at any point of time*

---

# PRD: Phase-Contrast Microscopy "Clean-Up" Pipeline (Hybrid CV + DL)

## 1. Project Overview

**Objective**: Build a modular, production-ready image preprocessing pipeline that suppresses the characteristic "halo" artifacts in Phase-Contrast microscopy and enhances cellular contrast. The pipeline must offer a **toggle** between a pure OpenCV path (lightweight) and a Hybrid path (OpenCV + PyTorch residual denoiser) to demonstrate algorithmic flexibility.

**Success Metric**: Achieve a **≥ 2dB PSNR improvement** and **≥ 0.05 SSIM improvement** on validation data compared to raw input, while maintaining inference speed < 100ms for the CV-only path and < 500ms for the hybrid path on a standard GPU (T4).

---

## 2. Directory Structure (To be generated)

```
phase_contrast_cleanup/
├── configs/
│   └── default_config.yaml          # Hyperparameters for CV and DL
├── data/
│   ├── raw/                         # Placeholder for input .tif/.png
│   ├── intermediate/                # Debug outputs (DoG, CLAHE results)
│   └── ground_truth/                # Clean brightfield targets (for training)
├── src/
│   ├── __init__.py
│   ├── preprocessor.py              # OpenCV logic (DoG, CLAHE, Normalization)
│   ├── residual_unet.py             # PyTorch Residual UNet definition
│   ├── dataset.py                   # PyTorch DataLoader for patch extraction
│   ├── trainer.py                   # Training loop (MSE Loss, AdamW)
│   ├── pipeline.py                  # Main orchestration (toggle CV/Hybrid)
│   └── metrics.py                   # PSNR, SSIM, and FLOPs counter
├── runs/                            # Tensorboard logs and checkpoints
├── requirements.txt
└── main.py                          # CLI entry point
```

---

## 3. Technical Stack & `requirements.txt`

```text
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scikit-image>=0.21.0
tifffile>=2023.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

---

## 4. Functional Requirements (Core Algorithms)

### 4.1. Module: `preprocessor.py` (Pure OpenCV Expertise)

*Implement the following class. Do NOT use deep learning here to prove custom OpenCV skills.*

- **`halo_suppression_dog(image, sigma1=1.0, sigma2=1.6)`**:
  - Apply Gaussian blur with `sigma1` (captures cell body).
  - Apply Gaussian blur with `sigma2` (captures halo). *Crucial: `sigma2` must be larger than `sigma1`*.
  - Subtract the second blur from the first to get the high-frequency DoG mask.
  - **Logic**: Since the halo is low-frequency and large, subtract the DoG response from the original image: `enhanced = image - (DoG_response * alpha)`, where `alpha` is a scaling factor (default 0.5) to prevent over-subtraction.
- **`apply_clahe(image, clip_limit=2.0, grid_size=(8,8))`**:
  - Convert to `uint16` or `float32` normalization.
  - Apply OpenCV's `cv2.createCLAHE` to locally enhance membrane/edge contrast without blowing up background noise.
- **`normalize_01(image)`**: Min-max scaling to [0, 1] for DL input.
- **`save_debug_overlay(original, dog_result, clahe_result, step)`**: Save side-by-side comparisons in the `intermediate/` folder.

### 4.2. Module: `residual_unet.py` (PyTorch Architecture)

*We do NOT train the UNet to output the clean image. We train it to predict the **residual noise**. This is crucial for the JD’s "feature extraction" and "quantitative" focus.*

- **Architecture**: Standard 4-level UNet (Down: 64→128→256→512; Up: 256→128→64→1).
- **Skip Connections**: Standard concatenation.
- **Forward Logic**:
  - Input: `x` (the image after OpenCV DoG + CLAHE).
  - Output: `residual` (the noise/halo artifacts the network thinks remain).
  - **Final Output**: `cleaned_image = x - residual` (Element-wise subtraction).
  - *Justification*: Residual learning forces the network to focus solely on the artifact texture, preserving the biological cell structure perfectly.

### 4.3. Module: `dataset.py` (Data Loading)

- Since phase-contrast lacks inherent "ground truth," the training strategy will use **synthetic corruption** or **public paired datasets** (e.g., phase-contrast vs. DIC/brightfield from Broad Institute).
- **Implementation**:
  - If paired data exists: Load `(noisy_input, clean_target)`.
  - If not: Load clean brightfield, synthetically add halo artifacts (simulate by adding a large Gaussian blur multiplied by a random intensity mask) to create `noisy_input`.
  - **Augmentations**: Random crops of `256x256`, random horizontal/vertical flips, random rotation.

### 4.4. Module: `trainer.py` (Training Loop)

- **Loss Function**: `MSELoss` on the residual (i.e., `(x - cleaned_image) - target_residual`). Alternatively, simply use `L1Loss` for sharper textures.
- **Optimizer**: AdamW (`lr=1e-4`, weight decay=1e-5).
- **Scheduler**: ReduceLROnPlateau (patience=5).
- **Eval Metric**: Track PSNR on the validation set every epoch. Stop if it plateaus.

---

## 5. The "Twist" Pipeline: `pipeline.py` (The Key Deliverable)

*This module implements the toggle feature. The user can call `pipeline.run(mode="cv_only")` or `mode="hybrid"`.*

**Step-by-step execution for `mode="cv_only"`**:

1. Load image -> Convert to float32.
2. Run `halo_suppression_dog()`.
3. Run `apply_clahe()`.
4. Return final image.

**Step-by-step execution for `mode="hybrid"`**:

1. Load image -> Convert to float32.
2. Run `halo_suppression_dog()`. *(This stabilizes the input for the DL model).*
3. Run `apply_clahe()`. *(This enhances edges so the residual UNet has a clean baseline).*
4. Normalize to [0,1].
5. Run PyTorch inference (`model.eval()` with `torch.no_grad()`) to get the `residual`.
6. Subtract residual.
7. Clip values to [0,1] and de-normalize.
8. Return final image.

**Required class structure**:

```python
class CleanupPipeline:
    def __init__(self, config_path, model_weights=None):
        self.config = load_yaml(config_path)
        self.cv_ops = Preprocessor()
        self.model = ResidualUNet()
        if model_weights: self.model.load_state_dict(...)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, image: np.ndarray, mode: str) -> np.ndarray:
        # Step 1: Always run CV baseline
        img = self.cv_ops.halo_suppression_dog(image)
        img = self.cv_ops.apply_clahe(img)
        if mode == "cv_only":
            return img
        elif mode == "hybrid":
            # Step 2: DL enhancement
            tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
            residual = self.model(tensor)
            cleaned = tensor - residual
            return cleaned.squeeze().cpu().numpy()
```

---

## 6. CLI Interface (`main.py`)

Claude must implement a CLI using `argparse`:

```bash
# To run inference on a folder
python main.py --input_dir data/raw/ --output_dir results/ --mode hybrid --weights runs/best.pth

# To run the CV-only baseline for comparison
python main.py --input_dir data/raw/ --output_dir results_cv/ --mode cv_only

# To train the model (requires ground_truth folder)
python main.py --train --data_dir data/ --epochs 50 --batch_size 16
```

---

## 7. Non-Functional Requirements (Performance)

| Mode            | Max Inference Time (per 512x512 image) | Max VRAM Usage |
| :-------------- | :------------------------------------- | :------------- |
| CV-Only (CPU)   | < 50 ms                                | 0 MB           |
| Hybrid (T4 GPU) | < 300 ms                               | < 2 GB         |

- The PRD demands an **FLOPs counter** using `thop` library integrated into `metrics.py` to quantify computational savings of the CV path vs. the DL path (impressing the "optimization" requirement in the JD).

---

## 8. Verification Plan (How Claude knows it works)

1. **Unit Test**: Create `tests/test_preprocessor.py` to verify DoG subtracts the correct channel dimensions and CLAHE doesn't output NaNs.
2. **Integration Test**: Run a synthetic image (circle with a halo blur) through the pipeline. Assert that the peak intensity outside the circle drops by at least 30%.
3. **Quantitative Report**: Generate a `report.md` after inference containing:
   - Average PSNR of raw vs. `cv_only` vs. `hybrid`.
   - Sample output images (Input, DoG only, CLAHE only, Hybrid final).
   - *Crucial Insight (as per JD)*: A statement like: *"The Hybrid pipeline recovers membranal edges with 95% fidelity, enabling a downstream Otsu-threshold segmentation to achieve 98% accuracy, compared to 80% on raw data."* (This fulfills the "One result or insight" requirement).

---

## 9. Implementation Instructions for Claude Code

*Please execute the following in sequential order:*

1. **Start by scaffolding** the exact directory structure listed above.
2. **Implement `preprocessor.py`** first without any PyTorch imports. Test it with a dummy `np.random.rand(256,256)` to ensure no crashes.
3. **Implement `residual_unet.py`**. Ensure the `forward` function explicitly contains the subtractive residual logic.
4. **Generate a synthetic dataset script** in `data/generate_synthetic.py` that creates 100 paired images (clean + halo) for training, since real medical data might not be accessible.
5. **Implement `trainer.py`** with TensorBoard logging.
6. **Implement the `pipeline.py`** with the strict toggle functionality.
7. **Finally, write the `main.py`** and the `README.md` explaining to the hiring manager exactly *why* subtracting the residual is biologically superior to standard denoising.

---

**Final Note for Claude**: Treat every function docstring as an explanation for a Biomedical Engineer who understands Python but not advanced CV. Use type hints everywhere. The final deliverable must be a Zip file of this repository with a pre-trained synthetic model (even if accuracy is poor, the *architecture* and *pipeline logic* must be flawless).
