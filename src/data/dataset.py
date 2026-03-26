"""
NYU Depth V2 dataset loader with paired RGB + depth samples.

Supports both the labeled subset (~1449 images for fast iteration) and
the full dataset (~50K frames for real training). Each sample returns
an RGB image normalised to [-1, 1] and a depth map normalised to [0, 1],
both at the configured spatial resolution.

The dataset expects the following directory layout (created by
scripts/download_data.py):

    data/nyu_depth_v2/
    ├── images/         # RGB images as .png
    └── depths/         # Depth maps as .png (16-bit or 8-bit)
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import EvalTransform, PairedTransform


class NYUDepthV2Dataset(Dataset):
    """Paired RGB + depth dataset from NYU Depth V2.

    Args:
        root: Path to the dataset root (containing images/ and depths/).
        image_size: Target spatial resolution (square).
        split: 'train' or 'val'. Uses an 85/15 split of available images.
        augment: Apply training augmentations (flip, crop). Ignored for val.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        split: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.depth_dir = self.root / "depths"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                f"Run scripts/download_data.py first."
            )

        # Collect image paths sorted for reproducible train/val split
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            # Also check for jpg
            self.image_paths = sorted(
                list(self.image_dir.glob("*.png"))
                + list(self.image_dir.glob("*.jpg"))
            )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. Run scripts/download_data.py first."
            )

        # Deterministic train/val split (85% train, 15% val)
        split_idx = int(len(self.image_paths) * 0.85)
        if split == "train":
            self.image_paths = self.image_paths[:split_idx]
        elif split == "val":
            self.image_paths = self.image_paths[split_idx:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Transforms
        if split == "train" and augment:
            self.transform = PairedTransform(
                image_size=image_size,
                random_flip=True,
                random_crop=True,
            )
        else:
            self.transform = EvalTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'rgb': (3, H, W) float32 tensor in [-1, 1]
                'depth': (1, H, W) float32 tensor in [0, 1]
        """
        img_path = self.image_paths[idx]
        depth_path = self.depth_dir / img_path.name

        rgb = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("I")  # 16-bit integer

        rgb_t, depth_t = self.transform(rgb, depth)

        return {"rgb": rgb_t, "depth": depth_t}
