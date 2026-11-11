"""Export the trained U-Net to ONNX (for ONNX Runtime, including the browser demo).

    python scripts/export_onnx.py                      # models/pretrained_fp16.pth -> models/phase_unet.onnx
    python scripts/export_onnx.py --weights runs/best.pth --out my.onnx

Model contract (what the browser demo and any other runtime must follow):

* input ``input``: float32 ``[1, 1, H, W]``, the *classical-path output* in
  [0, 1] (run ``Preprocessor.process`` first), with ``H`` and ``W`` multiples
  of 16 -- reflect-pad the bottom/right edges and crop the output back;
* output ``residual``: float32 ``[1, 1, H, W]``; the cleaned image is
  ``clip(input - residual, 0, 1)``.

H and W are dynamic. The weights are exported in float32 (fp16 checkpoints are
upcast) because ONNX Runtime's WebAssembly backend computes in float32. The
script checks the exported graph against PyTorch on a random input and on an
odd-sized one, and fails if they disagree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase_cleanup.pipeline import CleanupPipeline  # noqa: E402
from phase_cleanup.residual_unet import SIZE_MULTIPLE  # noqa: E402


class ResidualHead(torch.nn.Module):
    """Wraps the U-Net to output only the residual for an already-padded input."""

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.unet._features(x)
        return out if self.unet.residual else x - out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default_config.yaml"))
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "pretrained_fp16.pth"))
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "phase_unet.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    pipe = CleanupPipeline(args.config, model_weights=args.weights, device="cpu")
    model = ResidualHead(pipe.model.float().eval()).eval()

    dummy = torch.rand(1, 1, 256, 256)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["residual"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "residual": {2: "height", 3: "width"}},
        opset_version=args.opset, dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(out_path))
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for shape in [(256, 256), (16 * 13, 16 * 21)]:
        x = torch.rand(1, 1, *shape)
        with torch.no_grad():
            expected = model(x).numpy()
        got = session.run(["residual"], {"input": x.numpy()})[0]
        worst = max(worst, float(np.abs(got - expected).max()))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export_onnx] {out_path} ({size_mb:.1f} MB), max |onnx - torch| = {worst:.2e}, "
          f"inputs must be multiples of {SIZE_MULTIPLE}")
    if worst > 1e-3:
        print("[export_onnx] FAIL: ONNX output disagrees with PyTorch.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
