"""Shape and residual-logic tests for the U-Net."""

from __future__ import annotations

import pytest
import torch

from phase_cleanup.residual_unet import ResidualUNet, count_parameters, pad_to_multiple


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ResidualUNet(base_channels=8).eval()


@pytest.mark.parametrize("shape", [(64, 64), (65, 97), (500, 333), (17, 16), (8, 8)])
def test_any_size_round_trips(net, shape):
    """Odd sizes used to crash at the skip concatenation; output must match input size."""
    x = torch.rand(1, 1, *shape)
    with torch.no_grad():
        assert net(x).shape == x.shape
        assert net.predict_residual(x).shape == x.shape


def test_forward_subtracts_residual(net):
    x = torch.rand(2, 1, 48, 48)
    with torch.no_grad():
        assert torch.allclose(net(x), x - net.predict_residual(x), atol=1e-6)


def test_plain_variant_outputs_image_directly():
    """residual=False: forward returns the head output, predict_residual = x - output."""
    torch.manual_seed(0)
    plain = ResidualUNet(base_channels=8, residual=False).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(plain(x), plain._features(x), atol=1e-6)


def test_pad_to_multiple():
    x = torch.rand(1, 1, 30, 47)
    padded, (h, w) = pad_to_multiple(x, 16)
    assert padded.shape[-2:] == (32, 48) and (h, w) == (30, 47)
    assert torch.equal(padded[..., :30, :47], x)


def test_parameter_count_of_shipped_width():
    assert count_parameters(ResidualUNet(base_channels=32)) == 7_762_465
