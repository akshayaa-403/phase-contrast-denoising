"""Training loop for the Residual U-Net (residual-space supervision, TensorBoard logging)."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import psnr as psnr_metric
from .residual_unet import ResidualUNet


class Trainer:
    """Encapsulates the training/validation loop for ResidualUNet."""

    def __init__(
        self,
        model: ResidualUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        tcfg = config.get("training", {})
        pcfg = config.get("paths", {})

        self.epochs = int(tcfg.get("epochs", 50))
        loss_name = str(tcfg.get("loss", "l1")).lower()
        self.criterion: nn.Module = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(tcfg.get("learning_rate", 1e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max",  # maximise validation PSNR
            factor=0.5, patience=int(tcfg.get("scheduler_patience", 5)),
        )

        self.runs_dir = pcfg.get("runs_dir", "runs")
        self.best_ckpt = pcfg.get("best_checkpoint", os.path.join(self.runs_dir, "best.pth"))
        os.makedirs(self.runs_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.runs_dir)
        self.best_psnr = -float("inf")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            target_residual = noisy - clean
            pred_residual = self.model.predict_residual(noisy)
            loss = self.criterion(pred_residual, target_residual)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running += loss.item() * noisy.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running / max(1, len(self.train_loader.dataset))

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        psnr_vals = []
        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            cleaned = torch.clamp(self.model(noisy), 0.0, 1.0)
            cleaned_np = cleaned.squeeze().cpu().numpy()
            clean_np = clean.squeeze().numpy()
            psnr_vals.append(psnr_metric(cleaned_np, clean_np))
        # Drop inf (identical images) so the average stays finite.
        finite = [v for v in psnr_vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else 0.0

    def fit(self) -> str:
        """Run the full training loop; return the path of the best checkpoint."""
        print(f"[trainer] training for {self.epochs} epochs on {self.device}.")
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_psnr = self._validate(epoch)
            self.scheduler.step(val_psnr)
            lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("psnr/val", val_psnr, epoch)
            self.writer.add_scalar("lr", lr, epoch)
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                  f"| val_PSNR={val_psnr:.2f} dB | lr={lr:.2e}")

            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                torch.save(
                    {"model_state": self.model.state_dict(),
                     "epoch": epoch, "val_psnr": val_psnr},
                    self.best_ckpt,
                )
                print(f"  -> new best ({val_psnr:.2f} dB) saved to {self.best_ckpt}")

        self.writer.close()
        print(f"[trainer] done. Best val PSNR={self.best_psnr:.2f} dB -> {self.best_ckpt}")
        return self.best_ckpt
