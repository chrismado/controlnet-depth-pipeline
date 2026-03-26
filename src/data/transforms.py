"""
Image and depth map transforms for training and evaluation.

All spatial transforms (flip, crop) are applied consistently to both the RGB
image and its paired depth map so correspondence is preserved.
"""

import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class PairedTransform:
    """Spatial and colour transforms applied consistently to RGB + depth pairs.

    Spatial transforms (flip, crop) use the same random state for both
    modalities. Intensity transforms only affect the RGB image.

    Args:
        image_size: Target spatial resolution (square).
        random_flip: Apply random horizontal flip during training.
        random_crop: If True, random-crop from a slightly larger resize;
            otherwise centre-crop.
        crop_scale: Intermediate resize factor before cropping
            (e.g. 1.1 → resize to 110% then crop to image_size).
    """

    def __init__(
        self,
        image_size: int = 256,
        random_flip: bool = True,
        random_crop: bool = True,
        crop_scale: float = 1.1,
    ):
        self.image_size = image_size
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_scale = crop_scale

    def __call__(
        self, rgb: Image.Image, depth: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: PIL RGB image.
            depth: PIL depth image (mode 'L' or 'I' or 'F').

        Returns:
            (rgb_tensor, depth_tensor):
                rgb_tensor: (3, H, W) float32 in [-1, 1].
                depth_tensor: (1, H, W) float32 in [0, 1].
        """
        target = self.image_size

        if self.random_crop:
            # Resize slightly larger, then random crop
            resize_to = int(target * self.crop_scale)
            rgb = TF.resize(rgb, [resize_to, resize_to], interpolation=TF.InterpolationMode.BILINEAR)
            depth = TF.resize(depth, [resize_to, resize_to], interpolation=TF.InterpolationMode.NEAREST)

            # Same crop coordinates for both
            i, j, h, w = self._random_crop_params(resize_to, target)
            rgb = TF.crop(rgb, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)
        else:
            # Resize + centre crop
            rgb = TF.resize(rgb, [target, target], interpolation=TF.InterpolationMode.BILINEAR)
            depth = TF.resize(depth, [target, target], interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip (same decision for both)
        if self.random_flip and random.random() > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)

        # To tensor
        rgb_t = TF.to_tensor(rgb)          # (3, H, W) in [0, 1]
        depth_t = TF.to_tensor(depth)      # (1, H, W) in [0, 1] (or raw range)

        # Normalise RGB to [-1, 1]
        rgb_t = rgb_t * 2.0 - 1.0

        # Normalise depth to [0, 1] (handle arbitrary input range)
        d_min = depth_t.min()
        d_max = depth_t.max()
        if d_max - d_min > 1e-6:
            depth_t = (depth_t - d_min) / (d_max - d_min)
        else:
            depth_t = torch.zeros_like(depth_t)

        return rgb_t, depth_t

    @staticmethod
    def _random_crop_params(source_size: int, crop_size: int) -> tuple[int, int, int, int]:
        """Return (top, left, height, width) for a random crop."""
        margin = source_size - crop_size
        top = random.randint(0, margin)
        left = random.randint(0, margin)
        return top, left, crop_size, crop_size


class EvalTransform:
    """Deterministic transform for evaluation / inference.

    No augmentation — just resize and normalise.
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def __call__(
        self, rgb: Image.Image, depth: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.image_size
        rgb = TF.resize(rgb, [target, target], interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resize(depth, [target, target], interpolation=TF.InterpolationMode.NEAREST)

        rgb_t = TF.to_tensor(rgb) * 2.0 - 1.0

        depth_t = TF.to_tensor(depth)
        d_min, d_max = depth_t.min(), depth_t.max()
        if d_max - d_min > 1e-6:
            depth_t = (depth_t - d_min) / (d_max - d_min)
        else:
            depth_t = torch.zeros_like(depth_t)

        return rgb_t, depth_t
