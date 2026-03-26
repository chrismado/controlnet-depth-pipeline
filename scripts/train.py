#!/usr/bin/env python3
"""Training entry point for the ControlNet depth-conditioned diffusion pipeline.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_0050.pt
"""

import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import UNet, ControlNet, GaussianDiffusion
from src.data.dataset import NYUDepthV2Dataset
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ControlNet depth pipeline")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=== ControlNet Depth Pipeline Training ===")
    print(f"Config: {args.config}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Models
    model_kwargs = dict(
        image_size=config.get("image_size", 128),
        base_channels=config.get("base_channels", 128),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 3, 4])),
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_resolutions=tuple(config.get("attn_resolutions", [16, 8])),
        num_heads=config.get("num_heads", 4),
    )

    unet = UNet(**model_kwargs)
    controlnet = ControlNet(**model_kwargs)
    diffusion = GaussianDiffusion(
        num_timesteps=config.get("num_timesteps", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
    )

    # Print model sizes
    unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
    cnet_params = sum(p.numel() for p in controlnet.parameters()) / 1e6
    print(f"UNet:       {unet_params:.1f}M params")
    print(f"ControlNet: {cnet_params:.1f}M params")
    print(f"Total:      {unet_params + cnet_params:.1f}M params\n")

    # Dataset
    image_size = config.get("image_size", 128)
    data_dir = config.get("data_dir", "data/nyu_depth_v2")

    train_dataset = NYUDepthV2Dataset(root=data_dir, image_size=image_size, split="train")
    val_dataset = NYUDepthV2Dataset(root=data_dir, image_size=image_size, split="val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_dataset)} samples\n")

    # Trainer
    trainer = Trainer(
        unet=unet,
        controlnet=controlnet,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
