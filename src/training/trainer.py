"""
Training loop for the ControlNet depth-conditioned diffusion pipeline.

Features:
- Mixed precision (torch.amp) with gradient scaling
- Cosine learning rate schedule with linear warmup
- Exponential moving average (EMA) of model weights
- Weights & Biases logging (loss, LR, GPU memory, sample images)
- Periodic checkpointing
"""

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.model.diffusion import GaussianDiffusion


class EMAModel:
    """Exponential Moving Average of model parameters for better sample quality.

    Maintains a shadow copy of model weights that is updated as:
        shadow = decay * shadow + (1 - decay) * current

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (higher = slower update, smoother).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Full training loop for UNet + ControlNet with diffusion loss.

    Args:
        unet: The denoising U-Net.
        controlnet: The ControlNet conditioning module.
        diffusion: Gaussian diffusion process.
        train_loader: DataLoader yielding dicts with 'rgb' and 'depth' keys.
        val_loader: Optional validation DataLoader.
        config: Training configuration dict.
    """

    def __init__(
        self,
        unet: nn.Module,
        controlnet: nn.Module,
        diffusion: GaussianDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.unet = unet.to(self.device)
        self.controlnet = controlnet.to(self.device)
        self.diffusion = diffusion.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimiser — train both UNet and ControlNet jointly
        lr = self.config.get("learning_rate", 1e-4)
        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.controlnet.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        # Schedule
        num_epochs = self.config.get("num_epochs", 100)
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = self.config.get("warmup_steps", 1000)
        self.scheduler = cosine_warmup_schedule(self.optimizer, warmup_steps, total_steps)

        # Mixed precision
        self.use_amp = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = self.config.get("gradient_clip", 1.0)

        # EMA
        ema_decay = self.config.get("ema_decay", 0.9999)
        self.ema_unet = EMAModel(self.unet, decay=ema_decay)
        self.ema_controlnet = EMAModel(self.controlnet, decay=ema_decay)

        # Logging
        self.log_every = self.config.get("log_every_n_steps", 100)
        self.sample_every = self.config.get("sample_every_n_epochs", 5)
        self.save_every = self.config.get("save_every_n_epochs", 10)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B (optional — gracefully degrade if not installed or not configured)
        self.wandb_run = None
        wandb_project = self.config.get("wandb_project")
        if wandb_project:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    config=self.config,
                    resume="allow",
                )
            except Exception:
                print("W&B init failed — continuing without logging.")

        self.global_step = 0
        self.start_epoch = 0

    def train(self) -> None:
        """Run the full training loop."""
        num_epochs = self.config.get("num_epochs", 100)
        print(f"Training on {self.device} for {num_epochs} epochs")
        print(f"  Steps/epoch: {len(self.train_loader)}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Gradient clip: {self.grad_clip}")

        for epoch in range(self.start_epoch, num_epochs):
            self.unet.train()
            self.controlnet.train()
            epoch_loss = 0.0
            epoch_steps = 0
            t0 = time.time()

            for batch in self.train_loader:
                loss = self._train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                self.global_step += 1

                # Periodic logging
                if self.global_step % self.log_every == 0:
                    self._log_step(loss)

            epoch_loss /= max(epoch_steps, 1)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1}/{num_epochs} — "
                f"loss: {epoch_loss:.4f} — "
                f"time: {elapsed:.1f}s — "
                f"lr: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            # Epoch-level logging
            if self.wandb_run:
                import wandb

                wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss}, step=self.global_step)

            # Sample images
            if (epoch + 1) % self.sample_every == 0:
                self._generate_samples(epoch + 1)

            # Checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        # Final save
        self._save_checkpoint(num_epochs, tag="final")
        if self.wandb_run:
            self.wandb_run.finish()

    def _train_step(self, batch: dict) -> float:
        """Single training step: forward + backward + update."""
        rgb = batch["rgb"].to(self.device)      # (B, 3, H, W) in [-1, 1]
        depth = batch["depth"].to(self.device)   # (B, 1, H, W) in [0, 1]
        B = rgb.shape[0]

        # Random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            # ControlNet conditioning
            noise = torch.randn_like(rgb)
            x_t = self.diffusion.q_sample(rgb, t, noise=noise)
            cn_features = self.controlnet(x_t, t, depth)

            # UNet noise prediction
            predicted_noise = self.unet(x_t, t, controlnet_residuals=cn_features)
            loss = nn.functional.mse_loss(predicted_noise, noise)

        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.unet.parameters()) + list(self.controlnet.parameters()),
                self.grad_clip,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # EMA update
        self.ema_unet.update(self.unet)
        self.ema_controlnet.update(self.controlnet)

        return loss.item()

    def _log_step(self, loss: float) -> None:
        """Log metrics for the current step."""
        lr = self.scheduler.get_last_lr()[0]
        log_dict = {
            "train/loss": loss,
            "train/lr": lr,
            "train/step": self.global_step,
        }
        if self.device.type == "cuda":
            log_dict["train/gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

        if self.wandb_run:
            import wandb

            wandb.log(log_dict, step=self.global_step)

    @torch.no_grad()
    def _generate_samples(self, epoch: int, n_samples: int = 4) -> None:
        """Generate sample images using EMA weights and log to W&B."""
        self.ema_unet.shadow.eval()
        self.ema_controlnet.shadow.eval()

        # Use first batch from val_loader (or train_loader) for fixed depth maps
        loader = self.val_loader or self.train_loader
        batch = next(iter(loader))
        depth = batch["depth"][:n_samples].to(self.device)

        # ControlNet features from EMA model
        image_size = depth.shape[-1]
        dummy_noise = torch.randn(n_samples, 3, image_size, image_size, device=self.device)
        t_zeros = torch.zeros(n_samples, device=self.device, dtype=torch.long)

        # Use DDIM for speed
        def model_fn(x, t, **kw):
            cn = self.ema_controlnet.shadow(x, t, depth)
            return self.ema_unet.shadow(x, t, controlnet_residuals=cn)

        samples = self.diffusion.ddim_sample(
            model=model_fn,
            shape=(n_samples, 3, image_size, image_size),
            device=self.device,
            ddim_steps=50,
            show_progress=False,
        )

        # Clamp to [-1, 1] then rescale to [0, 1] for visualisation
        samples = (samples.clamp(-1, 1) + 1) / 2

        # Save locally
        samples_dir = Path("results/samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        from torchvision.utils import save_image

        save_image(samples, samples_dir / f"epoch_{epoch:04d}.png", nrow=n_samples)

        # Log to W&B
        if self.wandb_run:
            import wandb

            images = [wandb.Image(samples[i].cpu()) for i in range(n_samples)]
            depth_vis = [wandb.Image(depth[i].cpu()) for i in range(n_samples)]
            wandb.log(
                {"samples/generated": images, "samples/depth_input": depth_vis},
                step=self.global_step,
            )

    def _save_checkpoint(self, epoch: int, tag: str | None = None) -> None:
        """Save training checkpoint."""
        name = f"checkpoint_epoch_{epoch:04d}.pt" if tag is None else f"checkpoint_{tag}.pt"
        path = self.checkpoint_dir / name

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "unet": self.unet.state_dict(),
            "controlnet": self.controlnet.state_dict(),
            "ema_unet": self.ema_unet.state_dict(),
            "ema_controlnet": self.ema_controlnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
        }
        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(state["unet"])
        self.controlnet.load_state_dict(state["controlnet"])
        self.ema_unet.load_state_dict(state["ema_unet"])
        self.ema_controlnet.load_state_dict(state["ema_controlnet"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        print(f"  Resumed from epoch {self.start_epoch}, step {self.global_step}")
