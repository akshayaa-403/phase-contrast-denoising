"""Download the pre-trained model files from the GitHub release and verify them.

    python scripts/download_weights.py            # the PyTorch checkpoint (hybrid mode)
    python scripts/download_weights.py --onnx     # also the ONNX export (browser demo)

The weights are not stored in git (a binary that changes with every retrain
bloats the history permanently). They are attached to a GitHub release instead,
and this script checks each file's SHA-256 so a corrupted or substituted
download is rejected rather than silently loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/akshayaa-403/phase-contrast-denoising/releases/download/v1.0.0"

#: file name -> SHA-256. Update both when publishing a new release.
FILES = {
    "pretrained_fp16.pth": "b0896810245d87a9957e07ef014cd753fc9bfbff33e63fb53ddca9af1d2b89b9",
    "phase_unet.onnx": "7517c031705509605c23fc0e1d396fe8264686a59a087edcaad878d19ca2bbe7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest_dir: Path, force: bool = False) -> Path:
    """Download ``name`` into ``dest_dir`` (skipped if present and valid); verify its hash."""
    expected = FILES[name]
    dest = dest_dir / name
    if dest.is_file() and not force and sha256(dest) == expected:
        print(f"[download_weights] {dest} already present and verified.")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{RELEASE}/{name}"
    print(f"[download_weights] {url}")
    urllib.request.urlretrieve(url, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {name}:\n  expected {expected}\n  got      {actual}")
    tmp.replace(dest)
    print(f"[download_weights] {dest} verified ({dest.stat().st_size / 1e6:.1f} MB).")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--onnx", action="store_true", help="also download the ONNX export")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    names = ["pretrained_fp16.pth"] + (["phase_unet.onnx"] if args.onnx else [])
    for name in names:
        fetch(name, args.dest, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
