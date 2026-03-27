"""
Evaluation utilities for the depth-conditioned diffusion pipeline.

Generates sample images from a fixed set of depth maps and optionally
computes FID scores for quantitative evaluation.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.model.diffusion import GaussianDiffusion


@torch.no_grad()
def generate_samples(
    unet: nn.Module,
    controlnet: nn.Module,
    diffusion: GaussianDiffusion,
    depth_maps: torch.Tensor,
    device: torch.device | str = "cuda",
    ddim_steps: int = 50,
    show_progress: bool = True,
) -> torch.Tensor:
    """Generate RGB images conditioned on depth maps.

    Args:
        unet: Denoising U-Net (should be EMA weights for best quality).
        controlnet: ControlNet conditioning module.
        diffusion: Diffusion process.
        depth_maps: (B, 1, H, W) depth maps in [0, 1].
        device: Compute device.
        ddim_steps: Number of DDIM sampling steps.
        show_progress: Show tqdm progress bar.

    Returns:
        (B, 3, H, W) generated images in [0, 1] (clipped from model's [-1, 1] range).
    """
    unet.eval()
    controlnet.eval()
    depth_maps = depth_maps.to(device)
    B, _, H, W = depth_maps.shape

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cn_features = controlnet(x, t, depth_maps)
        return unet(x, t, controlnet_residuals=cn_features)

    samples = diffusion.ddim_sample(
        model=model_fn,
        shape=(B, 3, H, W),
        device=device,
        ddim_steps=ddim_steps,
        show_progress=show_progress,
    )

    # Rescale from [-1, 1] to [0, 1]
    return (samples.clamp(-1, 1) + 1) / 2


def save_comparison_grid(
    depth_maps: torch.Tensor,
    generated: torch.Tensor,
    save_path: str | Path,
    ground_truth: torch.Tensor | None = None,
) -> None:
    """Save a side-by-side comparison grid of depth → generated (→ ground truth).

    Args:
        depth_maps: (B, 1, H, W) depth maps in [0, 1].
        generated: (B, 3, H, W) generated images in [0, 1].
        save_path: Output file path.
        ground_truth: Optional (B, 3, H, W) ground truth images in [0, 1].
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    B = depth_maps.shape[0]
    # Expand depth to 3 channels for visualisation
    depth_vis = depth_maps.repeat(1, 3, 1, 1)

    if ground_truth is not None:
        # Layout: depth | generated | ground_truth (interleaved rows)
        rows = torch.cat([depth_vis, generated, ground_truth], dim=0)
        nrow = B  # 3 rows of B images
    else:
        rows = torch.cat([depth_vis, generated], dim=0)
        nrow = B  # 2 rows of B images

    save_image(rows, save_path, nrow=nrow, padding=2)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    val_loader: torch.utils.data.DataLoader,
    image_size: int = 128,
    n_samples: int = 8,
    ddim_steps: int = 50,
    output_dir: str | Path = "results/eval",
    device: str = "cuda",
) -> None:
    """Load a checkpoint and generate evaluation samples.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        val_loader: Validation data loader.
        image_size: Model's expected input size.
        n_samples: Number of samples to generate.
        ddim_steps: DDIM sampling steps.
        output_dir: Directory for output images.
        device: Compute device.
    """
    from src.model import ControlNet, GaussianDiffusion, UNet

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = state.get("config", {})

    # Reconstruct models
    unet = UNet(
        image_size=config.get("image_size", image_size),
        base_channels=config.get("base_channels", 128),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 3, 4])),
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_resolutions=tuple(config.get("attn_resolutions", [16, 8])),
        num_heads=config.get("num_heads", 4),
    ).to(device)

    controlnet = ControlNet(
        image_size=config.get("image_size", image_size),
        base_channels=config.get("base_channels", 128),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 3, 4])),
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_resolutions=tuple(config.get("attn_resolutions", [16, 8])),
        num_heads=config.get("num_heads", 4),
    ).to(device)

    # Load EMA weights (preferred) or regular weights
    if "ema_unet" in state:
        unet.load_state_dict(state["ema_unet"])
        controlnet.load_state_dict(state["ema_controlnet"])
    else:
        unet.load_state_dict(state["unet"])
        controlnet.load_state_dict(state["controlnet"])

    diffusion = GaussianDiffusion(
        num_timesteps=config.get("num_timesteps", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
    ).to(device)

    # Get depth maps from validation set
    batch = next(iter(val_loader))
    depth = batch["depth"][:n_samples]
    gt_rgb = batch["rgb"][:n_samples]

    # Generate
    print(f"Generating {n_samples} samples with {ddim_steps} DDIM steps...")
    generated = generate_samples(
        unet, controlnet, diffusion, depth, device=device, ddim_steps=ddim_steps
    )

    # Rescale ground truth from [-1, 1] to [0, 1]
    gt_rgb_vis = (gt_rgb.clamp(-1, 1) + 1) / 2

    # Save
    output_dir = Path(output_dir)
    save_comparison_grid(depth, generated, output_dir / "comparison.png", gt_rgb_vis)
    save_image(generated, output_dir / "generated.png", nrow=n_samples)
    print(f"Results saved to {output_dir}")
