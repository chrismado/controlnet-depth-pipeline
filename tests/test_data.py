"""Tests for the data pipeline (transforms and dataset)."""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms import PairedTransform, EvalTransform


def _make_dummy_images(size: int = 480):
    """Create dummy RGB and depth PIL images."""
    rgb = Image.new("RGB", (size, size), color=(128, 64, 32))
    depth = Image.new("I", (size, size))  # 16-bit mode
    return rgb, depth


# ---------- PairedTransform ----------


def test_paired_transform_output_shape():
    """PairedTransform should produce correctly sized tensors."""
    transform = PairedTransform(image_size=128, random_flip=True, random_crop=True)
    rgb, depth = _make_dummy_images()
    rgb_t, depth_t = transform(rgb, depth)

    assert rgb_t.shape == (3, 128, 128), f"RGB shape: {rgb_t.shape}"
    assert depth_t.shape == (1, 128, 128), f"Depth shape: {depth_t.shape}"


def test_paired_transform_rgb_range():
    """RGB tensor should be in [-1, 1]."""
    transform = PairedTransform(image_size=64, random_flip=False, random_crop=False)
    rgb, depth = _make_dummy_images()
    rgb_t, _ = transform(rgb, depth)

    assert rgb_t.min() >= -1.0
    assert rgb_t.max() <= 1.0


def test_paired_transform_depth_range():
    """Depth tensor should be in [0, 1]."""
    transform = PairedTransform(image_size=64, random_flip=False, random_crop=False)
    rgb, depth = _make_dummy_images()
    _, depth_t = transform(rgb, depth)

    assert depth_t.min() >= 0.0
    assert depth_t.max() <= 1.0


def test_paired_transform_consistent_flip():
    """Horizontal flip should be applied to both RGB and depth consistently.

    Strategy: create images with a left-right gradient so we can detect flips.
    """
    import numpy as np

    # Create gradient image: left side dark, right side bright
    arr_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_rgb[:, 32:, :] = 255
    arr_depth_raw = np.zeros((64, 64), dtype=np.int32)
    arr_depth_raw[:, 32:] = 65535

    rgb = Image.fromarray(arr_rgb)
    depth = Image.fromarray(arr_depth_raw, mode="I")

    # Run transform many times — at least some should be flipped
    transform = PairedTransform(image_size=64, random_flip=True, random_crop=False)

    flipped_count = 0
    for _ in range(20):
        rgb_t, depth_t = transform(rgb, depth)
        # If flipped, left side of tensor should be bright
        left_mean = rgb_t[:, :, :32].mean()
        if left_mean > 0:
            flipped_count += 1
            # Both should be flipped consistently
            depth_left = depth_t[:, :, :32].mean()
            assert depth_left > 0.3, "Depth should be flipped when RGB is flipped"

    # At least one flip should have occurred in 20 trials (p > 0.999999)
    assert flipped_count > 0, "Flip never triggered in 20 trials"


# ---------- EvalTransform ----------


def test_eval_transform_deterministic():
    """EvalTransform should produce identical outputs on repeated calls."""
    transform = EvalTransform(image_size=128)
    rgb, depth = _make_dummy_images()

    rgb_t1, depth_t1 = transform(rgb, depth)
    rgb_t2, depth_t2 = transform(rgb, depth)

    assert torch.equal(rgb_t1, rgb_t2)
    assert torch.equal(depth_t1, depth_t2)


def test_eval_transform_shape():
    """EvalTransform should produce correctly sized tensors."""
    transform = EvalTransform(image_size=256)
    rgb, depth = _make_dummy_images()
    rgb_t, depth_t = transform(rgb, depth)

    assert rgb_t.shape == (3, 256, 256)
    assert depth_t.shape == (1, 256, 256)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
