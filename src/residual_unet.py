"""Residual U-Net for phase-contrast halo/noise removal (predicts and subtracts the artifact)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


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


class ResidualUNet(nn.Module):
    """A 4-level U-Net that predicts and subtracts the residual artifact."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        b = base_channels

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

    def predict_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw predicted residual (artifact) map for input x."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = self.head(d1)
        return residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean the input by subtracting the predicted residual (cleaned = x - residual)."""
        residual = self.predict_residual(x)
        cleaned = x - residual
        return cleaned


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = ResidualUNet(in_channels=1, out_channels=1, base_channels=64)
    dummy = torch.rand(2, 1, 256, 256)
    with torch.no_grad():
        residual = net.predict_residual(dummy)
        cleaned = net(dummy)
    assert cleaned.shape == dummy.shape == residual.shape
    assert torch.allclose(cleaned, dummy - residual, atol=1e-6)
    print(f"[residual_unet] OK. Output shape {tuple(cleaned.shape)}, "
          f"params: {count_parameters(net):,}")
