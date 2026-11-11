"""Residual U-Net for phase-contrast artifact removal (predicts and subtracts the artifact).

The network does not output the clean image directly. It predicts the
*residual* -- the difference between its input and the clean target -- and the
forward pass subtracts it: ``cleaned = x - residual``. With ``residual=False``
the same architecture outputs the image directly; that variant exists only as
an ablation, to measure what residual learning buys (see the report).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Four 2x poolings: spatial sizes must be divisible by 2**4.
SIZE_MULTIPLE = 16


def _norm(num_channels: int) -> nn.Module:
    """Return a GroupNorm layer for num_channels."""
    # GroupNorm (not BatchNorm): independent of batch size and identical in
    # train/eval, so small CPU batches don't collapse eval-time output.
    groups = 8 if num_channels % 8 == 0 else 1
    return nn.GroupNorm(groups, num_channels)


class DoubleConv(nn.Module):
    """(Conv -> GroupNorm -> ReLU) x 2, the standard U-Net block (size-preserving)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def pad_to_multiple(x: torch.Tensor, multiple: int = SIZE_MULTIPLE) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad the last two dims up to a multiple; return ``(padded, (h, w))``.

    Without this, a 500 px image pools to 250 -> 125 -> 62, upsamples back to
    124, and the skip connection (125) cannot be concatenated.
    """
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        # Reflect padding needs pad < size; fall back to replicate for tiny inputs.
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (h, w)


class ResidualUNet(nn.Module):
    """A 4-level U-Net that predicts and subtracts the residual artifact."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        residual: bool = True,
    ) -> None:
        super().__init__()
        b = base_channels
        self.residual = bool(residual)

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder-decoder pass; ``x`` must already be a multiple of 16 in H and W."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    def predict_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted residual (artifact) map for input ``x``, any H x W.

        For the ``residual=False`` ablation this is ``x - output``, so callers
        can treat both variants the same way.
        """
        padded, (h, w) = pad_to_multiple(x)
        out = self._features(padded)[..., :h, :w]
        return out if self.residual else x - out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean the input by subtracting the predicted residual (cleaned = x - residual)."""
        residual = self.predict_residual(x)
        cleaned = x - residual
        return cleaned


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
