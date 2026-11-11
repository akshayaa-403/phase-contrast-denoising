"""Image loading and saving shared by the CLI, the trainer, the report and the demo.

Every entry point used to carry its own copy of the loader, and they disagreed:
one guessed the bit depth from ``max() > 255`` (so a dark 16-bit frame was
divided by 255), another checked the dtype *after* casting to float32 (so the
check never fired and each 16-bit image was scaled by its own maximum). Scaling
by the file's **declared** bit depth is the only rule that gives a model the
same numbers at training and inference time, so it lives here, once.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
TIFF_EXTS = {".tif", ".tiff"}


def list_images(folder: str) -> list[str]:
    """Return the sorted file names of the images in ``folder``."""
    return sorted(
        f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def _read_raw(path: str) -> np.ndarray:
    """Read the pixels exactly as stored (dtype preserved)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF_EXTS:
        import tifffile

        try:
            img = tifffile.imread(path)
        except ValueError:
            # Compressed TIFFs (LZW, Deflate, ...) need the optional
            # `imagecodecs` package in tifffile; OpenCV decodes them natively.
            ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
            if not ok or not pages:
                raise
            img = pages[0]
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # A multi-page stack or a (T/Z, H, W) volume: take the first plane. A
        # trailing small axis (H, W, 3/4) is colour and is handled below.
        while img.ndim > 3 or (img.ndim == 3 and img.shape[-1] not in (3, 4)):
            img = img[0]
        if img.ndim == 3:  # tifffile returns RGB order
            img = img[..., :3]
            img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(img)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def to_unit_range(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Scale an integer or float image to float32 [0, 1] by its declared depth.

    Returns ``(image01, bit_depth)``: 8 or 16 for integer images, 32 for float
    images (which are assumed to already be in [0, 1] and are only clipped).
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 8
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 16
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        scaled = (img.astype(np.float64) - info.min) / float(info.max - info.min)
        return scaled.astype(np.float32), 16
    return np.clip(img.astype(np.float32), 0.0, 1.0), 32


def load_gray01(path: str, return_depth: bool = False):
    """Load any supported image as single-channel float32 in [0, 1].

    Colour images are converted to grey; TIFF stacks give their first plane.
    With ``return_depth=True`` returns ``(image, bit_depth)`` so the result can
    be written back at the precision it came in with.
    """
    img, depth = to_unit_range(_read_raw(path))
    return (img, depth) if return_depth else img


def output_path(out_dir: str, name: str) -> str:
    """Where to write the cleaned version of ``name``: lossy formats become PNG."""
    stem, ext = os.path.splitext(name)
    if ext.lower() in {".jpg", ".jpeg", ".bmp"}:
        ext = ".png"
    return os.path.join(out_dir, stem + ext)


def save_image01(path: str, image01: np.ndarray, bit_depth: Optional[int] = 8) -> None:
    """Write a [0, 1] float image as 8-bit or 16-bit PNG/TIFF.

    ``bit_depth`` 16 (or 32, a float source) keeps 16-bit precision so a 16-bit
    microscope frame does not come back quantised to 256 levels.
    """
    img = np.clip(np.asarray(image01, dtype=np.float32), 0.0, 1.0)
    if bit_depth and bit_depth > 8:
        data = np.round(img * 65535.0).astype(np.uint16)
    else:
        data = np.round(img * 255.0).astype(np.uint8)
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF_EXTS:
        import tifffile

        tifffile.imwrite(path, data)
    elif not cv2.imwrite(path, data):
        raise OSError(f"Could not write image: {path}")
