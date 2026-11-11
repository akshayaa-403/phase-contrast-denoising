"""Interactive Gradio demo for the Phase-Contrast Clean-Up pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from phase_cleanup.image_io import to_unit_range
from phase_cleanup.pipeline import CleanupPipeline, load_yaml
from phase_cleanup.preprocessor import Preprocessor

log = logging.getLogger("app")

CONFIG_PATH = "configs/default_config.yaml"


def _to_gray01(image: np.ndarray) -> np.ndarray:
    """Convert an uploaded RGB/gray image to float32 grayscale in [0, 1] by its bit depth."""
    if image is None:
        raise ValueError("No image provided.")
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    return to_unit_range(image)[0]


def _to_display(image01: np.ndarray) -> np.ndarray:
    """Float [0,1] image -> uint8 for Gradio display."""
    return (np.clip(image01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _otsu_foreground_fraction(image01: np.ndarray) -> float:
    """Fraction of pixels Otsu labels as foreground."""
    _, mask = cv2.threshold(_to_display(image01), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((mask > 0).mean())


class DemoState:
    """Holds one shared pipeline so the model loads only once.

    The pipeline itself is never modified per request: each request builds its
    own ``Preprocessor`` from the slider values, so two visitors moving sliders
    at the same time cannot see each other's settings.
    """

    def __init__(self, weights: Optional[str]) -> None:
        self.pipeline = CleanupPipeline(CONFIG_PATH, model_weights=None)
        self.hybrid_available = False
        if weights:
            try:
                self.pipeline.load_weights(weights)
                self.hybrid_available = True
            except Exception as exc:
                log.warning("Could not load weights (%s); hybrid disabled.", exc)
        self.defaults = self.pipeline.cv_ops

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

        cv_ops = Preprocessor.from_config(
            self.pipeline.config["preprocessor"],
            dog_sigma2=float(dog_sigma2),
            dog_alpha=float(dog_alpha),
            clahe_clip_limit=float(clahe_clip),
        )
        gray = _to_gray01(image)
        raw = cv_ops.normalize_01(cv_ops.match_polarity(gray))

        notes = []
        if mode == "hybrid" and not self.hybrid_available:
            mode = "cv_only"
            notes.append("> No trained weights found, so this shows **cv_only**.")
        changed = cv_ops.to_config() != self.defaults.to_config()
        if mode == "hybrid" and changed:
            notes.append("> The sliders differ from the settings the U-Net was trained with. "
                         "It has never seen inputs like these, so hybrid output may degrade; "
                         "reset them for the trained behaviour.")

        cleaned = self.pipeline.run(gray, mode=mode, preprocessor=cv_ops)

        info = (
            f"**Mode:** `{mode}`  \n"
            f"**Otsu foreground fraction:** raw {_otsu_foreground_fraction(raw) * 100:.1f}% -> "
            f"cleaned {_otsu_foreground_fraction(cleaned) * 100:.1f}%  \n"
            "*(A stable, meaningful foreground fraction indicates the halo and uneven "
            "illumination are no longer fooling the threshold.)*"
        )
        if notes:
            info += "\n\n" + "\n\n".join(notes)
        return _to_display(raw), _to_display(cleaned), info


def build_ui(state: DemoState):
    import gradio as gr

    d = state.defaults
    with gr.Blocks(title="Phase-Contrast Clean-Up") as demo:
        gr.Markdown(
            "# Phase-Contrast Microscopy Clean-Up\n"
            "Suppress the phase-contrast **halo** and uneven illumination, and enhance "
            "cell contrast. Toggle between the lightweight **CV-only** path (background "
            "flattening, Difference-of-Gaussians, CLAHE) and the **Hybrid** path (CV + a "
            "residual U-Net that subtracts the leftover artifact)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input (phase-contrast)", type="numpy", image_mode=None)
                mode = gr.Radio(
                    choices=["cv_only", "hybrid"],
                    value="hybrid" if state.hybrid_available else "cv_only",
                    label="Pipeline mode",
                )
                sigma2 = gr.Slider(2.0, 30.0, value=d.dog_sigma2, step=0.5,
                                   label="DoG sigma2 (halo scale, px)")
                alpha = gr.Slider(0.0, 3.0, value=d.dog_alpha, step=0.05,
                                  label="DoG alpha (halo subtraction strength)")
                clip = gr.Slider(0.5, 8.0, value=d.clahe_clip_limit, step=0.5,
                                 label="CLAHE clip limit (edge contrast)")
                run_btn = gr.Button("Clean image", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    raw_out = gr.Image(label="Raw (polarity-matched, normalised)")
                    clean_out = gr.Image(label="Cleaned")
                info = gr.Markdown()

        if not state.hybrid_available:
            gr.Markdown(
                "> **Hybrid disabled**: no model weights found. Run "
                "`python scripts/download_weights.py`, or train with "
                "`python main.py --train --data_dir data`."
            )

        inputs = [inp, mode, sigma2, alpha, clip]
        outputs = [raw_out, clean_out, info]
        run_btn.click(state.process, inputs=inputs, outputs=outputs)
        # Re-run when a slider is let go (not on every drag step) or the mode flips.
        for slider in (sigma2, alpha, clip):
            slider.release(state.process, inputs=inputs, outputs=outputs)
        mode.change(state.process, inputs=inputs, outputs=outputs)
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    default_weights = load_yaml(CONFIG_PATH)["paths"].get("pretrained", "models/pretrained_fp16.pth")
    parser = argparse.ArgumentParser(description="Gradio demo for the clean-up pipeline.")
    parser.add_argument("--weights", default=default_weights, help="model checkpoint")
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
