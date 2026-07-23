"""Interactive Gradio demo for the Phase-Contrast Clean-Up pipeline."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from src.pipeline import CleanupPipeline


CONFIG_PATH = "configs/default_config.yaml"
DEFAULT_WEIGHTS = "runs/best.pth"


def _to_gray01(image: np.ndarray) -> np.ndarray:
    """Convert an uploaded RGB/gray uint8 image to float32 grayscale in [0, 1]."""
    if image is None:
        raise ValueError("No image provided.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    maxv = 65535.0 if image.max() > 255 else 255.0
    return np.clip(image / maxv, 0.0, 1.0)


def _to_display(image01: np.ndarray) -> np.ndarray:
    """Float [0,1] image -> uint8 for Gradio display."""
    return (np.clip(image01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _otsu_foreground_fraction(image01: np.ndarray) -> float:
    """Fraction of pixels Otsu labels as foreground."""
    u8 = _to_display(image01)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((mask > 0).mean())


class DemoState:
    """Holds one shared pipeline instance so the model loads only once."""

    def __init__(self, weights: Optional[str]) -> None:
        self.pipeline = CleanupPipeline(CONFIG_PATH, model_weights=None)
        self.hybrid_available = False
        if weights and os.path.isfile(weights):
            try:
                self.pipeline.load_weights(weights)
                self.hybrid_available = True
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[app] could not load weights ({exc}); hybrid disabled.")

    def process(
        self,
        image: np.ndarray,
        mode: str,
        dog_sigma2: float,
        dog_alpha: float,
        clahe_clip: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Run the pipeline and return (raw_display, cleaned_display, info_md)."""
        if image is None:
            return None, None, "Upload an image to begin."

        self.pipeline.cv_ops.dog_sigma2 = float(dog_sigma2)
        self.pipeline.cv_ops.dog_alpha = float(dog_alpha)
        self.pipeline.cv_ops.clahe_clip_limit = float(clahe_clip)

        gray = _to_gray01(image)
        raw_disp = _to_display(self.pipeline.cv_ops.normalize_01(gray))

        if mode == "hybrid" and not self.hybrid_available:
            mode = "cv_only"
            note = "\n\n> ⚠️ No trained weights found — showing **cv_only**."
        else:
            note = ""

        cleaned = self.pipeline.run(gray, mode=mode)
        cleaned_disp = _to_display(cleaned)

        fg_raw = _otsu_foreground_fraction(self.pipeline.cv_ops.normalize_01(gray))
        fg_clean = _otsu_foreground_fraction(cleaned)
        info = (
            f"**Mode:** `{mode}`  \n"
            f"**Otsu foreground fraction** — raw: {fg_raw*100:.1f}% → "
            f"cleaned: {fg_clean*100:.1f}%  \n"
            f"*(A stable, meaningful foreground fraction indicates the halo is no "
            f"longer fooling the threshold.)*" + note
        )
        return raw_disp, cleaned_disp, info


def build_ui(state: DemoState):
    import gradio as gr

    with gr.Blocks(title="Phase-Contrast Clean-Up") as demo:
        gr.Markdown(
            "# 🔬 Phase-Contrast Microscopy Clean-Up\n"
            "Suppress the phase-contrast **halo** and enhance cell contrast. "
            "Toggle between the lightweight **CV-only** path (Difference-of-"
            "Gaussians + CLAHE) and the **Hybrid** path (CV + a residual U-Net "
            "that subtracts the leftover artifact)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input (phase-contrast)", type="numpy")
                mode = gr.Radio(
                    choices=["cv_only", "hybrid"],
                    value="hybrid" if state.hybrid_available else "cv_only",
                    label="Pipeline mode",
                )
                sigma2 = gr.Slider(2.0, 30.0, value=8.0, step=0.5,
                                   label="DoG sigma2 (halo scale)")
                alpha = gr.Slider(0.0, 2.0, value=0.6, step=0.05,
                                  label="DoG alpha (halo subtraction strength)")
                clip = gr.Slider(0.5, 8.0, value=2.0, step=0.5,
                                 label="CLAHE clip limit (edge contrast)")
                run_btn = gr.Button("Clean image", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    raw_out = gr.Image(label="Raw (normalised)")
                    clean_out = gr.Image(label="Cleaned")
                info = gr.Markdown()

        if not state.hybrid_available:
            gr.Markdown(
                "> ℹ️ **Hybrid disabled** — no `runs/best.pth` found. Train with "
                "`python main.py --train --data_dir data` to enable it."
            )

        run_btn.click(
            state.process,
            inputs=[inp, mode, sigma2, alpha, clip],
            outputs=[raw_out, clean_out, info],
        )
        mode.change(
            state.process,
            inputs=[inp, mode, sigma2, alpha, clip],
            outputs=[raw_out, clean_out, info],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for the clean-up pipeline.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="model checkpoint")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="server port")
    args = parser.parse_args()

    state = DemoState(args.weights)
    demo = build_ui(state)
    try:
        demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    except Exception:
        # A requested public share tunnel can fail when offline/firewalled; fall
        # back to a purely local server so the demo still runs.
        demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
