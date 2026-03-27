"""
Model loading, preprocessing, and inference for the serving endpoint.

Handles:
- Loading a trained checkpoint (UNet + ControlNet + diffusion)
- Preprocessing input depth maps (resize, normalise)
- Running DDIM sampling for fast inference
- Postprocessing output to PIL images
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.model import ControlNet, GaussianDiffusion, UNet
from .monitoring import InferenceTimer


class InferencePipeline:
    """End-to-end inference pipeline: depth map → generated image.

    Args:
        checkpoint_path: Path to a saved training checkpoint (.pt file).
        device: Compute device ('cuda' or 'cpu').
        ddim_steps: Number of DDIM sampling steps (fewer = faster).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        ddim_steps: int = 50,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ddim_steps = ddim_steps

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        config = state.get("config", {})
        self.image_size = config.get("image_size", 128)

        # Reconstruct models
        model_kwargs = dict(
            image_size=self.image_size,
            base_channels=config.get("base_channels", 128),
            channel_mult=tuple(config.get("channel_mult", [1, 2, 3, 4])),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_resolutions=tuple(config.get("attn_resolutions", [16, 8])),
            num_heads=config.get("num_heads", 4),
        )

        self.unet = UNet(**model_kwargs).to(self.device)
        self.controlnet = ControlNet(**model_kwargs).to(self.device)
        self.diffusion = GaussianDiffusion(
            num_timesteps=config.get("num_timesteps", 1000),
            beta_start=config.get("beta_start", 1e-4),
            beta_end=config.get("beta_end", 0.02),
        ).to(self.device)

        # Prefer EMA weights
        if "ema_unet" in state:
            self.unet.load_state_dict(state["ema_unet"])
            self.controlnet.load_state_dict(state["ema_controlnet"])
        else:
            self.unet.load_state_dict(state["unet"])
            self.controlnet.load_state_dict(state["controlnet"])

        self.unet.eval()
        self.controlnet.eval()

        print(f"Model loaded from {checkpoint_path} (image_size={self.image_size})")

    def preprocess_depth(self, depth_image: Image.Image) -> torch.Tensor:
        """Convert a PIL depth image to model input tensor.

        Args:
            depth_image: PIL image (any mode — will be converted to grayscale).

        Returns:
            (1, 1, H, W) float32 tensor in [0, 1] on self.device.
        """
        depth = depth_image.convert("L")
        depth = TF.resize(depth, [self.image_size, self.image_size])
        depth_t = TF.to_tensor(depth)  # (1, H, W) in [0, 1]
        return depth_t.unsqueeze(0).to(self.device)

    def postprocess(self, samples: torch.Tensor) -> list[Image.Image]:
        """Convert model output tensor to PIL images.

        Args:
            samples: (B, 3, H, W) in [-1, 1].

        Returns:
            List of PIL RGB images.
        """
        # Clamp and rescale to [0, 255]
        images = (samples.clamp(-1, 1) + 1) / 2 * 255
        images = images.byte().cpu()
        return [TF.to_pil_image(img) for img in images]

    @torch.no_grad()
    def generate(
        self,
        depth_image: Image.Image,
        ddim_steps: int | None = None,
    ) -> Image.Image:
        """Generate a single RGB image from a depth map.

        Args:
            depth_image: Input depth map as PIL image.
            ddim_steps: Override default DDIM steps.

        Returns:
            Generated PIL RGB image.
        """
        steps = ddim_steps or self.ddim_steps
        depth_t = self.preprocess_depth(depth_image)

        with InferenceTimer():
            def model_fn(x, t):
                cn = self.controlnet(x, t, depth_t)
                return self.unet(x, t, controlnet_residuals=cn)

            samples = self.diffusion.ddim_sample(
                model=model_fn,
                shape=(1, 3, self.image_size, self.image_size),
                device=self.device,
                ddim_steps=steps,
                show_progress=False,
            )

        return self.postprocess(samples)[0]

    @torch.no_grad()
    def generate_batch(
        self,
        depth_images: list[Image.Image],
        ddim_steps: int | None = None,
    ) -> list[Image.Image]:
        """Generate RGB images from a batch of depth maps.

        Args:
            depth_images: List of PIL depth images.
            ddim_steps: Override default DDIM steps.

        Returns:
            List of generated PIL RGB images.
        """
        steps = ddim_steps or self.ddim_steps
        B = len(depth_images)

        # Stack depth maps into a batch
        depth_batch = torch.cat(
            [self.preprocess_depth(d) for d in depth_images], dim=0
        )

        with InferenceTimer():
            def model_fn(x, t):
                cn = self.controlnet(x, t, depth_batch)
                return self.unet(x, t, controlnet_residuals=cn)

            samples = self.diffusion.ddim_sample(
                model=model_fn,
                shape=(B, 3, self.image_size, self.image_size),
                device=self.device,
                ddim_steps=steps,
                show_progress=False,
            )

        return self.postprocess(samples)
