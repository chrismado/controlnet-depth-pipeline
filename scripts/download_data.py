#!/usr/bin/env python3
"""Download and prepare the NYU Depth V2 labeled subset.

Downloads the official labeled subset (~2.8 GB .mat file containing ~1449
synchronised RGB + depth pairs), extracts them into individual PNG files,
and organises the directory structure expected by the dataset loader.

Output layout:
    data/nyu_depth_v2/
    ├── images/           # RGB images as {index:05d}.png
    └── depths/           # Depth maps as {index:05d}.png (16-bit)

Usage:
    python scripts/download_data.py [--output-dir data/nyu_depth_v2]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

# The official labeled subset hosted by the NYU authors
NYU_DEPTH_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return

    print(f"  Downloading {url}")
    print(f"  Saving to {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.urlopen(url)
    total = int(req.headers.get("Content-Length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb = downloaded / 1e6
                total_mb = total / 1e6
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print()


def extract_mat(mat_path: Path, output_dir: Path) -> int:
    """Extract RGB and depth images from the .mat file.

    Returns the number of image pairs extracted.
    """
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required to read the .mat file.")
        print("Install it with: pip install h5py")
        sys.exit(1)

    images_dir = output_dir / "images"
    depths_dir = output_dir / "depths"
    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Reading {mat_path} ...")
    with h5py.File(mat_path, "r") as f:
        # h5py reads MATLAB v7.3 with reversed dims:
        #   images: (N, 3, W, H) uint8   (MATLAB was H×W×3×N)
        #   depths: (N, W, H) float32     (MATLAB was H×W×N)
        images = f["images"]  # (1449, 3, 640, 480)
        depths = f["depths"]  # (1449, 640, 480)

        n_samples = images.shape[0]
        print(f"  Found {n_samples} image pairs")

        from PIL import Image

        for i in range(n_samples):
            # RGB: (3, W, H) → transpose to (H, W, 3)
            rgb = np.array(images[i]).transpose(2, 1, 0)  # (H, W, 3)
            rgb_img = Image.fromarray(rgb)
            rgb_img.save(images_dir / f"{i:05d}.png")

            # Depth: (W, H) → transpose to (H, W), save as 16-bit PNG
            depth = np.array(depths[i]).T  # (H, W)
            depth_normalised = depth / max(depth.max(), 1e-6) * 65535
            depth_16bit = depth_normalised.astype(np.uint16)
            depth_img = Image.fromarray(depth_16bit)
            depth_img.save(depths_dir / f"{i:05d}.png")

            if (i + 1) % 100 == 0 or i == n_samples - 1:
                print(f"\r  Extracted {i + 1}/{n_samples} pairs", end="", flush=True)

        print()
        return n_samples


def print_stats(output_dir: Path) -> None:
    """Print dataset statistics."""
    images_dir = output_dir / "images"
    depths_dir = output_dir / "depths"

    n_images = len(list(images_dir.glob("*.png")))
    n_depths = len(list(depths_dir.glob("*.png")))

    train_count = int(n_images * 0.85)
    val_count = n_images - train_count

    print("\n--- Dataset Statistics ---")
    print(f"  Location:     {output_dir.resolve()}")
    print(f"  RGB images:   {n_images}")
    print(f"  Depth maps:   {n_depths}")
    print(f"  Train split:  {train_count} (85%)")
    print(f"  Val split:    {val_count} (15%)")

    # Check a sample image size
    from PIL import Image

    sample = Image.open(next(images_dir.iterdir()))
    print(f"  Native size:  {sample.size[0]}x{sample.size[1]}")
    print("--- Ready for training ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYU Depth V2 labeled subset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/nyu_depth_v2",
        help="Output directory (default: data/nyu_depth_v2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    mat_path = output_dir / "nyu_depth_v2_labeled.mat"

    # Check if already extracted
    images_dir = output_dir / "images"
    if images_dir.exists() and len(list(images_dir.glob("*.png"))) > 0:
        print("Dataset already extracted. Printing stats:")
        print_stats(output_dir)
        return

    print("=== NYU Depth V2 Download ===\n")

    # Step 1: Download
    print("[1/3] Downloading labeled subset (~2.8 GB)...")
    download_file(NYU_DEPTH_URL, mat_path)

    # Step 2: Extract
    print("[2/3] Extracting image pairs...")
    extract_mat(mat_path, output_dir)

    # Step 3: Stats
    print("[3/3] Dataset ready!")
    print_stats(output_dir)

    # Optionally remove the .mat file to save space
    print(f"Tip: You can delete {mat_path} to save ~2.8 GB of disk space.")
    print(f"     The extracted PNGs in {output_dir} are all you need.\n")


if __name__ == "__main__":
    main()
