"""
U-Net architecture for diffusion-based image generation.

Standard U-Net with:
- Residual blocks at 4 resolution levels
- Self-attention at 16x16 and 8x8 resolutions
- Sinusoidal timestep embeddings injected via addition after projection
- Group normalization, SiLU activations
- Skip connections between encoder and decoder
- Optional ControlNet conditioning feature injection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timesteps to dense vectors using a fixed sinusoidal basis,
    following the DDPM convention (Ho et al., 2020). Frequencies are log-spaced
    from 1 to 1/10000, producing embeddings that capture both fine and coarse
    temporal structure.

    Args:
        timesteps: (B,) integer timestep indices.
        dim: Embedding dimension. Must be even.

    Returns:
        (B, dim) embedding vectors.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Structure: GroupNorm → SiLU → Conv → (+time) → GroupNorm → SiLU → Conv → (+skip)

    The timestep embedding is projected to the block's channel dimension and
    added to the hidden state after the first convolution. A learned 1x1 conv
    handles channel mismatches on the residual path.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Timestep injection: project then broadcast over spatial dims
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial dimensions.

    Flattens H×W into a sequence, applies standard scaled dot-product
    attention, then reshapes back. Uses pre-norm (GroupNorm before projection).
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Pre-norm, flatten spatial dims, project to Q/K/V
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        qkv = self.qkv(h).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, HW, head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, HW, head_dim)

        # Merge heads and project back
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).permute(0, 2, 1).reshape(B, C, H, W)
        return out + residual


class Downsample(nn.Module):
    """2x spatial downsampling via strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling via nearest interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """U-Net for denoising diffusion models.

    Architecture overview (default config, 128x128 input):

        Encoder                    Decoder
        -------                    -------
        128x128  (128ch)  ←skip→   128x128  (128ch)
          ↓ downsample               ↑ upsample
        64x64   (256ch)  ←skip→   64x64   (256ch)
          ↓ downsample               ↑ upsample
        32x32   (384ch)  ←skip→   32x32   (384ch)
          ↓ downsample               ↑ upsample
        16x16   (512ch)  ←skip→   16x16   (512ch)  [attention]
          ↓ downsample               ↑ upsample
                    8x8 bottleneck [attention]

    Each resolution level has `num_res_blocks` residual blocks. Self-attention
    is applied at resolutions in `attn_resolutions` (default: 16x16 and 8x8).

    The forward pass accepts optional `controlnet_residuals` — a list of
    per-level conditioning tensors from a ControlNet module, added to the
    decoder hidden state after each upsample.

    Args:
        image_size: Expected input spatial resolution.
        in_channels: Input image channels (3 for RGB).
        out_channels: Output channels (3 for predicted noise).
        base_channels: Base channel count, scaled by channel_mult per level.
        channel_mult: Tuple of per-level channel multipliers.
        num_res_blocks: Number of residual blocks per level.
        attn_resolutions: Spatial resolutions where self-attention is applied.
        num_heads: Number of attention heads.
        num_groups: Groups for GroupNorm.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        out_channels: int = 3,
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

        # --- Timestep embedding MLP ---
        # Sinusoidal embedding → Linear → SiLU → Linear
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self._sinusoidal_dim = base_channels

        # --- Input projection ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        # Flat ModuleList: each entry is [ResBlock, (optional) SelfAttention]
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        enc_channels = []  # track per-skip channel counts for building decoder
        res = image_size

        for level in range(self.num_levels):
            out_ch = base_channels * channel_mult[level]
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, num_groups)]
                if res in attn_resolutions:
                    layers.append(SelfAttention(out_ch, num_heads, num_groups))
                self.enc_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                enc_channels.append(ch)
            self.downsamples.append(Downsample(ch))
            res //= 2

        # --- Bottleneck ---
        self.mid_block1 = ResBlock(ch, ch, time_dim, num_groups)
        self.mid_attn = SelfAttention(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock(ch, ch, time_dim, num_groups)

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            self.upsamples.append(Upsample(ch))
            res *= 2
            out_ch = base_channels * channel_mult[level]
            for _ in range(num_res_blocks):
                skip_ch = enc_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_dim, num_groups)]
                if res in attn_resolutions:
                    layers.append(SelfAttention(out_ch, num_heads, num_groups))
                self.dec_blocks.append(nn.ModuleList(layers))
                ch = out_ch

        # --- Output ---
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        controlnet_residuals: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) integer timestep indices.
            controlnet_residuals: Optional list of `num_levels` tensors from
                ControlNet, indexed by level. Each is added to the encoder
                skip connections at the matching resolution before they are
                consumed by the decoder.

        Returns:
            (B, out_channels, H, W) predicted noise.
        """
        # Timestep embedding
        t_emb = self.time_embed(sinusoidal_embedding(t, self._sinusoidal_dim))

        h = self.input_conv(x)

        # ---- Encoder: collect skip connections ----
        skips = []
        block_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                modules = self.enc_blocks[block_idx]
                h = modules[0](h, t_emb)        # ResBlock
                if len(modules) > 1:
                    h = modules[1](h)            # SelfAttention
                skips.append(h)
                block_idx += 1
            h = self.downsamples[level](h)

        # ---- Inject ControlNet features into skip connections ----
        if controlnet_residuals is not None:
            idx = 0
            for level in range(self.num_levels):
                for _ in range(self.num_res_blocks):
                    skips[idx] = skips[idx] + controlnet_residuals[level]
                    idx += 1

        # ---- Bottleneck ----
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ---- Decoder: consume skip connections ----
        block_idx = 0
        for i, level in enumerate(reversed(range(self.num_levels))):
            h = self.upsamples[i](h)

            for _ in range(self.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                modules = self.dec_blocks[block_idx]
                h = modules[0](h, t_emb)        # ResBlock
                if len(modules) > 1:
                    h = modules[1](h)            # SelfAttention
                block_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))
