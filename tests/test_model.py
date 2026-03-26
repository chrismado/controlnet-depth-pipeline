"""Smoke tests for Session 1: model architecture shapes.

Verifies that UNet, ControlNet, and GaussianDiffusion produce correct tensor
shapes for a 128×128 input with batch_size=2. Run with:
    python -m pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import torch

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import UNet, ControlNet, GaussianDiffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2
IMAGE_SIZE = 128
IN_CH = 3
DEPTH_CH = 1


def _make_inputs():
    """Create dummy inputs on DEVICE."""
    x = torch.randn(BATCH, IN_CH, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    t = torch.randint(0, 1000, (BATCH,), device=DEVICE)
    depth = torch.randn(BATCH, DEPTH_CH, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    return x, t, depth


# ---------- UNet ----------


def test_unet_output_shape():
    """UNet output should match input spatial dims and have out_channels=3."""
    model = UNet(image_size=IMAGE_SIZE).to(DEVICE)
    x, t, _ = _make_inputs()

    out = model(x, t)

    assert out.shape == (BATCH, IN_CH, IMAGE_SIZE, IMAGE_SIZE), (
        f"Expected ({BATCH}, {IN_CH}, {IMAGE_SIZE}, {IMAGE_SIZE}), got {out.shape}"
    )


def test_unet_with_controlnet_residuals():
    """UNet should accept per-level ControlNet features without error."""
    model = UNet(image_size=IMAGE_SIZE).to(DEVICE)
    x, t, _ = _make_inputs()

    # Simulate ControlNet outputs: one feature map per level
    channel_mult = (1, 2, 3, 4)
    base_ch = 128
    res = IMAGE_SIZE
    fake_residuals = []
    for mult in channel_mult:
        ch = base_ch * mult
        fake_residuals.append(torch.randn(BATCH, ch, res, res, device=DEVICE))
        res //= 2

    out = model(x, t, controlnet_residuals=fake_residuals)
    assert out.shape == (BATCH, IN_CH, IMAGE_SIZE, IMAGE_SIZE)


# ---------- ControlNet ----------


def test_controlnet_output_shapes():
    """ControlNet should return num_levels feature maps at correct resolutions."""
    cnet = ControlNet(image_size=IMAGE_SIZE).to(DEVICE)
    x, t, depth = _make_inputs()

    outputs = cnet(x, t, depth)

    assert len(outputs) == 4, f"Expected 4 level outputs, got {len(outputs)}"

    channel_mult = (1, 2, 3, 4)
    base_ch = 128
    res = IMAGE_SIZE
    for level, (feat, mult) in enumerate(zip(outputs, channel_mult)):
        expected_ch = base_ch * mult
        expected_shape = (BATCH, expected_ch, res, res)
        assert feat.shape == expected_shape, (
            f"Level {level}: expected {expected_shape}, got {feat.shape}"
        )
        res //= 2


def test_controlnet_zero_conv_init():
    """Zero-conv layers should be initialised to zero → initial output is zero."""
    cnet = ControlNet(image_size=IMAGE_SIZE).to(DEVICE)
    for i, zc in enumerate(cnet.zero_convs):
        assert (zc.weight == 0).all(), f"zero_conv[{i}] weights not zero"
        assert (zc.bias == 0).all(), f"zero_conv[{i}] bias not zero"


# ---------- GaussianDiffusion ----------


def test_q_sample_shape():
    """Forward diffusion should preserve input shape."""
    diffusion = GaussianDiffusion(num_timesteps=1000).to(DEVICE)
    x0 = torch.randn(BATCH, IN_CH, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    t = torch.randint(0, 1000, (BATCH,), device=DEVICE)

    x_t = diffusion.q_sample(x0, t)
    assert x_t.shape == x0.shape


def test_q_sample_noise_level():
    """At t=0, noised image should be very close to clean image."""
    diffusion = GaussianDiffusion(num_timesteps=1000).to(DEVICE)
    x0 = torch.randn(BATCH, IN_CH, 32, 32, device=DEVICE)
    t = torch.zeros(BATCH, device=DEVICE, dtype=torch.long)

    x_t = diffusion.q_sample(x0, t)
    # At t=0, sqrt_alpha_bar ≈ 1, sqrt_one_minus ≈ 0
    # sqrt_one_minus_alpha_bar at t=0 ≈ 0.01, so noise contributes ~0.01 * |z|
    assert torch.allclose(x_t, x0, atol=0.05), "x_t should ≈ x_0 at t=0"


def test_p_losses_scalar():
    """Training loss should be a scalar."""
    model = UNet(image_size=64, base_channels=32, channel_mult=(1, 2, 2, 4)).to(DEVICE)
    diffusion = GaussianDiffusion(num_timesteps=100).to(DEVICE)
    x0 = torch.randn(BATCH, IN_CH, 64, 64, device=DEVICE)
    t = torch.randint(0, 100, (BATCH,), device=DEVICE)

    loss = diffusion.p_losses(model, x0, t)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"


def test_p_sample_shape():
    """Single reverse step should preserve spatial shape."""
    model = UNet(image_size=64, base_channels=32, channel_mult=(1, 2, 2, 4)).to(DEVICE)
    diffusion = GaussianDiffusion(num_timesteps=100).to(DEVICE)
    x_t = torch.randn(BATCH, IN_CH, 64, 64, device=DEVICE)

    x_prev = diffusion.p_sample(model, x_t, t_index=50)
    assert x_prev.shape == x_t.shape


# ---------- End-to-end: UNet + ControlNet through diffusion ----------


def test_end_to_end_controlnet_diffusion():
    """Full forward pass: ControlNet features → UNet → diffusion loss."""
    unet = UNet(image_size=64, base_channels=32, channel_mult=(1, 2, 2, 4)).to(DEVICE)
    cnet = ControlNet(
        image_size=64, base_channels=32, channel_mult=(1, 2, 2, 4)
    ).to(DEVICE)
    diffusion = GaussianDiffusion(num_timesteps=100).to(DEVICE)

    x0 = torch.randn(BATCH, IN_CH, 64, 64, device=DEVICE)
    t = torch.randint(0, 100, (BATCH,), device=DEVICE)
    depth = torch.randn(BATCH, DEPTH_CH, 64, 64, device=DEVICE)

    # ControlNet encodes depth conditioning
    x_t = diffusion.q_sample(x0, t)
    cn_features = cnet(x_t, t, depth)

    # UNet predicts noise with ControlNet features
    predicted_noise = unet(x_t, t, controlnet_residuals=cn_features)

    assert predicted_noise.shape == x0.shape
    loss = torch.nn.functional.mse_loss(predicted_noise, torch.randn_like(x0))
    assert loss.item() > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
