"""PyTorch dataset & dataloaders for paired phase-contrast denoising."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .image_io import IMG_EXTS, load_gray01
from .preprocessor import Preprocessor

log = logging.getLogger(__name__)


def _list_pairs(noisy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
    """Return matched (noisy_path, clean_path) pairs sharing a filename."""
    if not (os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)):
        raise FileNotFoundError(
            f"Expected paired folders:\n  noisy: {noisy_dir}\n  clean: {clean_dir}\n"
            "Run `python data/generate_synthetic.py` first."
        )
    clean_files = {
        f for f in os.listdir(clean_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    }
    pairs: List[Tuple[str, str]] = []
    for f in sorted(os.listdir(noisy_dir)):
        if os.path.splitext(f)[1].lower() in IMG_EXTS and f in clean_files:
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
        # The classical preprocessing is deterministic, so each pair is loaded
        # and processed once and reused every epoch (per worker process).
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.pairs)

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
        if idx not in self._cache:
            # Exactly the classical path inference runs, so training matches production.
            self._cache[idx] = (self.pre.process(load_gray01(noisy_path)), load_gray01(clean_path))
        noisy, clean = self._cache[idx]

        if self.train:
            noisy, clean = self._augment(noisy, clean)

        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_t = torch.from_numpy(clean).float().unsqueeze(0)
        return noisy_t, clean_t


def split_pairs(
    pairs: List[Tuple[str, str]],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split pairs into (train, val) with a seeded permutation.

    The one place the held-out set is defined. ``build_dataloaders`` uses it for
    training and ``scripts/make_demo_assets.py`` uses it to publish only frames
    the model never trained on, so the two cannot drift apart.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


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

    train_pairs, val_pairs = split_pairs(pairs, val_split, seed)

    train_ds = PhaseContrastDataset(train_pairs, preprocessor, patch_size, train=True)
    val_ds = PhaseContrastDataset(val_pairs, preprocessor, patch_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    # Validation uses full images, whose sizes may differ, so batch_size 1.
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    log.info("%d train / %d val pairs.", len(train_pairs), len(val_pairs))
    return train_loader, val_loader
