"""Training loop for the Residual U-Net (residual-space supervision, TensorBoard logging).

Each epoch trains on random patches, then scores the validation images. The
checkpoint with the best validation PSNR is kept as ``best.pth``; ``last.pth``
holds the most recent epoch *with* optimizer and scheduler state so an
interrupted run can be resumed with ``--resume``. Training stops early once
validation PSNR has not improved for ``early_stopping_patience`` epochs.

Every checkpoint records the model and preprocessing settings it was trained
with, the git commit, and the epoch, so a checkpoint can always be traced back
to the code and configuration that produced it.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import psnr as psnr_metric
from .pipeline import git_sha
from .residual_unet import ResidualUNet

log = logging.getLogger(__name__)


class Trainer:
    """Encapsulates the training/validation loop for ResidualUNet."""

    def __init__(
        self,
        model: ResidualUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cpu",
        preprocessor_config: Optional[Dict] = None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        tcfg = config.get("training", {})
        pcfg = config.get("paths", {})

        self.epochs = int(tcfg.get("epochs", 50))
        loss_name = str(tcfg.get("loss", "l1")).lower()
        self.criterion: nn.Module = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()
        self.early_stopping_patience = int(tcfg.get("early_stopping_patience", 10))
        self.use_amp = bool(tcfg.get("amp", True)) and device.startswith("cuda")
        self.log_images_every = int(tcfg.get("log_images_every", 5))

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(tcfg.get("learning_rate", 1e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max",  # maximise validation PSNR
            factor=0.5, patience=int(tcfg.get("scheduler_patience", 5)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.runs_dir = pcfg.get("runs_dir", "runs")
        self.best_ckpt = pcfg.get("best_checkpoint", os.path.join(self.runs_dir, "best.pth"))
        self.last_ckpt = pcfg.get("last_checkpoint", os.path.join(self.runs_dir, "last.pth"))
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.best_ckpt) or ".", exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.runs_dir)

        self.best_psnr = -float("inf")
        self.start_epoch = 1
        self.epochs_without_improvement = 0
        self.meta = {
            "model_config": dict(config.get("model", {})),
            "preprocessor": dict(preprocessor_config or config.get("preprocessor", {})),
            "git_sha": git_sha(),
        }

    # ------------------------------------------------------------------ checkpoints
    def _save(self, path: str, epoch: int, val_psnr: float, full_state: bool) -> None:
        payload = {"model_state": self.model.state_dict(), "epoch": epoch,
                   "val_psnr": float(val_psnr), **self.meta}
        if full_state:
            payload.update({
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_psnr": float(self.best_psnr),
                "epochs_without_improvement": self.epochs_without_improvement,
            })
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)  # never leave a half-written checkpoint behind

    def resume(self, path: Optional[str] = None) -> None:
        """Restore model, optimizer, scheduler and counters from ``last.pth``."""
        path = path or self.last_ckpt
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_psnr = float(ckpt.get("best_psnr", ckpt.get("val_psnr", -float("inf"))))
        self.epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        log.info("Resumed from %s at epoch %d (best %.2f dB).", path, self.start_epoch, self.best_psnr)

    # ------------------------------------------------------------------ loops
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for noisy, clean in pbar:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            target_residual = noisy - clean
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                pred_residual = self.model.predict_residual(noisy)
                loss = self.criterion(pred_residual.float(), target_residual)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * noisy.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running / max(1, len(self.train_loader.dataset))

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        psnr_vals, losses = [], []
        logged = False
        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                residual = self.model.predict_residual(noisy).float()
            losses.append(float(self.criterion(residual, noisy - clean)))
            cleaned = torch.clamp(noisy - residual, 0.0, 1.0)
            psnr_vals.append(psnr_metric(cleaned[0, 0].cpu().numpy(), clean[0, 0].cpu().numpy()))

            if not logged and self.log_images_every and epoch % self.log_images_every == 0:
                # input | predicted residual (mid-grey = 0) | output | target
                res_vis = torch.clamp(residual * 0.5 + 0.5, 0.0, 1.0)
                grid = torch.cat([noisy, res_vis, cleaned, clean], dim=-1)[0]
                self.writer.add_image("val/input_residual_output_target", grid, epoch)
                logged = True
        finite = [v for v in psnr_vals if np.isfinite(v)]
        return {
            "psnr": float(np.mean(finite)) if finite else 0.0,
            "loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def fit(self) -> str:
        """Run the training loop; return the path of the best checkpoint."""
        log.info("Training for up to %d epochs on %s (AMP %s).", self.epochs, self.device,
                 "on" if self.use_amp else "off")
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val = self._validate(epoch)
            self.scheduler.step(val["psnr"])
            lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val["loss"], epoch)
            self.writer.add_scalar("psnr/val", val["psnr"], epoch)
            self.writer.add_scalar("lr", lr, epoch)
            log.info("Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_PSNR=%.2f dB | lr=%.2e",
                     epoch, train_loss, val["loss"], val["psnr"], lr)

            if val["psnr"] > self.best_psnr:
                self.best_psnr = val["psnr"]
                self.epochs_without_improvement = 0
                self._save(self.best_ckpt, epoch, val["psnr"], full_state=False)
                log.info("  -> new best (%.2f dB) saved to %s", val["psnr"], self.best_ckpt)
            else:
                self.epochs_without_improvement += 1
            self._save(self.last_ckpt, epoch, val["psnr"], full_state=True)

            if self.early_stopping_patience and self.epochs_without_improvement >= self.early_stopping_patience:
                log.info("Validation PSNR has not improved for %d epochs; stopping early.",
                         self.epochs_without_improvement)
                break

        self.writer.close()
        log.info("Done. Best val PSNR=%.2f dB -> %s", self.best_psnr, self.best_ckpt)
        return self.best_ckpt
