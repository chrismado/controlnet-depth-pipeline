"""
ControlNet for depth-conditioned image generation.

Mirrors the U-Net encoder architecture with an additional depth map encoder
and zero-convolution output layers. At training start, zero-conv weights are
all zeros so the ControlNet contributes nothing — the pretrained U-Net
behaviour is preserved until the ControlNet gradually learns to inject
spatial conditioning.

Architecture:
    depth_map (1ch) → depth_encoder → depth_features (base_ch)
    noisy_image (3ch) → input_conv → image_features (base_ch)
    h = image_features + depth_features
    h → cloned encoder (ResBlocks + Attention + Downsamples)
    per-level outputs → zero_conv → conditioning features
"""

import torch
import torch.nn as nn

from .unet import (
    Downsample,
    ResBlock,
    SelfAttention,
    sinusoidal_embedding,
)


def zero_conv(channels: int) -> nn.Conv2d:
    """1×1 convolution with weights and bias initialized to zero."""
    conv = nn.Conv2d(channels, channels, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class ControlNet(nn.Module):
    """Depth-conditioned ControlNet.

    Clones the U-Net encoder structure and adds:
    - A small conv stack that encodes a 1-channel depth map to base_channels.
    - Zero-conv layers (one per resolution level) that gate the conditioning
      signal. Because they start at zero, the ControlNet initially has no
      effect on generation.

    The returned per-level features are added to the U-Net decoder's hidden
    state after each upsample (see UNet.forward).

    Args:
        image_size: Expected input spatial resolution.
        in_channels: Noisy image channels (3 for RGB).
        depth_channels: Depth map channels (1 for single-channel depth).
        base_channels: Base channel count, scaled by channel_mult per level.
        channel_mult: Per-level channel multipliers (must match U-Net).
        num_res_blocks: Residual blocks per level (must match U-Net).
        attn_resolutions: Resolutions where self-attention is applied.
        num_heads: Number of attention heads.
        num_groups: Groups for GroupNorm.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        depth_channels: int = 1,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16, 8),
        num_heads: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        time_dim = base_channels * 4

        # --- Timestep embedding (own copy, same architecture as U-Net) ---
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self._sinusoidal_dim = base_channels

        # --- Depth map encoder ---
        # Small conv stack: depth_channels → base_channels
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, base_channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1),
        )

        # --- Input projection for noisy image (mirrors U-Net input_conv) ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Cloned encoder (mirrors U-Net encoder architecture) ---
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.zero_convs = nn.ModuleList()

        ch = base_channels
        res = image_size

        for level in range(self.num_levels):
            out_ch = base_channels * channel_mult[level]
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, num_groups)]
                if res in attn_resolutions:
                    layers.append(SelfAttention(out_ch, num_heads, num_groups))
                self.enc_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            # Zero-conv gate for this level's output
            self.zero_convs.append(zero_conv(ch))
            self.downsamples.append(Downsample(ch))
            res //= 2

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        depth: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) noisy image (same input the U-Net receives).
            t: (B,) integer timestep indices.
            depth: (B, 1, H, W) depth map conditioning signal.

        Returns:
            List of `num_levels` tensors, one per resolution level. Each has
            shape (B, level_channels, H_level, W_level) and is intended to be
            added to the U-Net decoder hidden state at the matching resolution.
        """
        t_emb = self.time_embed(sinusoidal_embedding(t, self._sinusoidal_dim))

        # Fuse noisy image features with depth conditioning
        h = self.input_conv(x) + self.depth_encoder(depth)

        # Encode through cloned encoder, extract gated per-level features
        outputs: list[torch.Tensor] = []
        block_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                modules = self.enc_blocks[block_idx]
                h = modules[0](h, t_emb)        # ResBlock
                if len(modules) > 1:
                    h = modules[1](h)            # SelfAttention
                block_idx += 1

            # Gate with zero-conv before emitting
            outputs.append(self.zero_convs[level](h))
            h = self.downsamples[level](h)

        return outputs
