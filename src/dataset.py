"""PyTorch dataset & dataloaders for paired phase-contrast denoising."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessor import Preprocessor


def _list_pairs(noisy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
    """Return matched (noisy_path, clean_path) pairs sharing a filename."""
    if not (os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)):
        raise FileNotFoundError(
            f"Expected paired folders:\n  noisy: {noisy_dir}\n  clean: {clean_dir}\n"
            "Run `python data/generate_synthetic.py` first."
        )
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    clean_files = {
        f for f in os.listdir(clean_dir)
        if os.path.splitext(f)[1].lower() in exts
    }
    pairs: List[Tuple[str, str]] = []
    for f in sorted(os.listdir(noisy_dir)):
        if os.path.splitext(f)[1].lower() in exts and f in clean_files:
            pairs.append((os.path.join(noisy_dir, f), os.path.join(clean_dir, f)))
    if not pairs:
        raise RuntimeError(f"No matching pairs found between {noisy_dir} and {clean_dir}.")
    return pairs


class PhaseContrastDataset(Dataset):
    """Paired dataset that applies OpenCV pre-processing + optional augmentation."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        preprocessor: Preprocessor,
        patch_size: int = 256,
        train: bool = True,
    ) -> None:
        self.pairs = pairs
        self.pre = preprocessor
        self.patch_size = int(patch_size)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_gray01(path: str) -> np.ndarray:
        """Load an image as float32 grayscale scaled to [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        maxv = 65535.0 if img.dtype == np.uint16 else 255.0
        if img.max() > maxv:
            maxv = float(img.max())
        return np.clip(img / maxv, 0.0, 1.0)

    def _augment(
        self, noisy: np.ndarray, clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical random crop + flips + rot90 to both images."""
        h, w = noisy.shape
        ps = self.patch_size

        if h < ps or w < ps:
            pad_h, pad_w = max(0, ps - h), max(0, ps - w)
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = noisy.shape
        top = int(np.random.randint(0, h - ps + 1))
        left = int(np.random.randint(0, w - ps + 1))
        noisy = noisy[top:top + ps, left:left + ps]
        clean = clean[top:top + ps, left:left + ps]

        if np.random.rand() < 0.5:
            noisy, clean = noisy[:, ::-1], clean[:, ::-1]
        if np.random.rand() < 0.5:
            noisy, clean = noisy[::-1, :], clean[::-1, :]
        k = int(np.random.randint(0, 4))
        if k:
            noisy, clean = np.rot90(noisy, k), np.rot90(clean, k)

        return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[idx]
        noisy = self._load_gray01(noisy_path)
        clean = self._load_gray01(clean_path)

        # Same OpenCV pre-processing as inference, so training matches production.
        noisy = self.pre.halo_suppression_dog(noisy)
        noisy = self.pre.apply_clahe(noisy)
        noisy = self.pre.normalize_01(noisy)

        if self.train:
            noisy, clean = self._augment(noisy, clean)

        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_t = torch.from_numpy(clean).float().unsqueeze(0)
        return noisy_t, clean_t


def build_dataloaders(
    data_dir: str,
    preprocessor: Preprocessor,
    patch_size: int = 256,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from data_dir/{noisy_input,ground_truth}."""
    noisy_dir = os.path.join(data_dir, "noisy_input")
    clean_dir = os.path.join(data_dir, "ground_truth")
    pairs = _list_pairs(noisy_dir, clean_dir)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    train_ds = PhaseContrastDataset(train_pairs, preprocessor, patch_size, train=True)
    val_ds = PhaseContrastDataset(val_pairs, preprocessor, patch_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    # Validation uses full images, whose sizes may differ, so batch_size 1.
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    print(f"[dataset] {len(train_pairs)} train / {len(val_pairs)} val pairs.")
    return train_loader, val_loader
