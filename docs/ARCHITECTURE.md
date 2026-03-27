# Architecture Overview

This document describes the architecture of the ControlNet depth-conditioned
diffusion pipeline, covering the model components, conditioning mechanism,
training loop, and serving infrastructure.

## System Components

```
                        +-----------------+
                        |  FastAPI Server  |
                        |   (app.py)       |
                        +--------+--------+
                                 |
                        +--------v--------+
                        | InferencePipeline|
                        |  (inference.py)  |
                        +--------+--------+
                                 |
              +------------------+------------------+
              |                  |                   |
     +--------v------+  +-------v--------+  +-------v---------+
     |   ControlNet   |  |     UNet       |  | GaussianDiffusion|
     | (controlnet.py)|  |   (unet.py)    |  |  (diffusion.py)  |
     +----------------+  +----------------+  +------------------+
```

## Model Architecture

### UNet (Denoising Network)

The UNet is a standard encoder-decoder architecture for denoising diffusion
models. Default configuration for 128x128 inputs:

- **Encoder**: 4 resolution levels (128, 64, 32, 16), each with 2 residual
  blocks. Self-attention is applied at 16x16 and 8x8 resolutions.
- **Bottleneck**: ResBlock, self-attention, ResBlock at the lowest resolution.
- **Decoder**: Mirrors the encoder with skip connections from the encoder
  at each level. Upsampling uses nearest-neighbour interpolation + conv.
- **Timestep conditioning**: Sinusoidal positional embeddings are projected
  through an MLP and added to each residual block.
- **Channel progression**: 128 -> 256 -> 384 -> 512 (base_channels * channel_mult).

The UNet predicts the noise component of a noisy input, which the diffusion
process uses to iteratively denoise.

### ControlNet (Depth Conditioning)

The ControlNet mirrors the UNet encoder architecture and adds a parallel
pathway for spatial conditioning via depth maps:

1. **Depth encoder**: A small conv stack (1ch -> 64 -> 64 -> 128) encodes
   the single-channel depth map to the base channel dimension.
2. **Feature fusion**: Depth features are added to the noisy image features
   at the input stage.
3. **Cloned encoder**: Processes the fused features through the same
   architecture as the UNet encoder (ResBlocks + Attention + Downsamples).
4. **Zero-convolution gates**: Each resolution level's output passes through
   a 1x1 convolution initialized to all zeros. This ensures the ControlNet
   has no effect at training start, preserving the pretrained UNet behaviour.

The ControlNet outputs per-level feature tensors that are injected into the
UNet decoder's skip connections.

### How Depth Conditioning Works

The key insight of ControlNet is the zero-convolution initialization:

```
Training start:  zero_conv weights = 0
                 => ControlNet output = 0
                 => UNet behaves as if unconditional

Training progresses: zero_conv weights learn non-zero values
                     => ControlNet gradually injects spatial structure
                     => UNet learns to use depth cues for generation
```

At inference time, the depth map flows through:

```
depth_map (1, 1, H, W)
    |
    v
depth_encoder -> depth_features (B, 128, H, W)
    |
    + image_features (from noisy image input_conv)
    |
    v
cloned_encoder (ResBlocks + Attention per level)
    |
    v
zero_conv per level -> conditioning features
    |
    v
added to UNet skip connections at each resolution
```

This means areas with strong depth gradients (edges, surfaces) provide
stronger conditioning, guiding the diffusion process to generate images
that respect the 3D structure encoded in the depth map.

### Gaussian Diffusion

Implements the DDPM forward and reverse processes:

- **Forward process**: Gradually adds Gaussian noise to clean images
  according to a linear beta schedule over T=1000 timesteps.
- **Training objective**: Simple MSE loss between true noise and the
  UNet's predicted noise at randomly sampled timesteps.
- **DDPM sampling**: Full T-step reverse diffusion (slow but high quality).
- **DDIM sampling**: Accelerated sampling using a subsequence of timesteps
  (default 50 steps). Deterministic when eta=0.

## Training Pipeline

```
RGB image (x_0) ----+----> q_sample(x_0, t) = x_t ----> UNet(x_t, t, cn) ----> noise_pred
                    |                                        ^
                    |                                        |
Depth map ----------+----> ControlNet(x_t, t, depth) -------+
                    |
Random noise (eps) -+----> MSE(noise_pred, eps) = loss
```

Key training features:
- **Mixed precision** (torch.amp) with gradient scaling for memory efficiency.
- **Cosine LR schedule** with linear warmup (default 1000 steps).
- **EMA** (exponential moving average) of both UNet and ControlNet weights
  for improved sample quality at inference time.
- **Joint optimization**: Both UNet and ControlNet are trained together
  with a single AdamW optimizer.

## Serving Architecture

The FastAPI server (`src/serving/app.py`) exposes three generation endpoints:

| Endpoint          | Input                    | Output           |
|-------------------|--------------------------|------------------|
| POST /generate    | Multipart depth map PNG  | PNG image        |
| POST /generate_json | Base64-encoded depth map | Base64 PNG image |
| POST /generate_batch | Multiple depth map PNGs | ZIP of PNG images|

Additional endpoints:
- `GET /health` -- readiness check (model loaded status).
- `GET /metrics` -- Prometheus metrics (request counts, latencies, GPU memory).

The `InferencePipeline` handles checkpoint loading, input validation,
preprocessing (resize, normalise), DDIM sampling, and postprocessing.

## Data Pipeline

Uses the NYU Depth V2 dataset with paired RGB + depth images:

- **PairedTransform**: Training augmentations (random flip, random crop)
  applied consistently to both RGB and depth to preserve correspondence.
- **EvalTransform**: Deterministic resize-only transform for validation.
- Both transforms normalise RGB to [-1, 1] and depth to [0, 1].

## Directory Structure

```
src/
  model/
    unet.py          # Denoising U-Net with skip connections
    controlnet.py    # Depth-conditioned ControlNet
    diffusion.py     # DDPM/DDIM forward and reverse processes
  data/
    dataset.py       # NYU Depth V2 dataset loader
    transforms.py    # Paired spatial transforms
  training/
    trainer.py       # Training loop with EMA, AMP, scheduling
    evaluate.py      # Evaluation and sample generation
  serving/
    app.py           # FastAPI endpoints
    inference.py     # Model loading and inference pipeline
    monitoring.py    # Prometheus metrics
scripts/
  train.py           # Training entry point
  serve.py           # Server entry point
  download_data.py   # Dataset download script
```
