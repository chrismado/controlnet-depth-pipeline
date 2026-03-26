# controlnet-depth-pipeline

Depth-conditioned image generation using ControlNet-style spatial conditioning in PyTorch. Takes a depth map as input, generates a textured RGB image that respects the spatial structure of the scene.

Built from scratch — no pretrained Stable Diffusion, no Hugging Face wrappers. The goal was to understand and implement the conditioning mechanism at the architecture level, not call an API.

## Architecture

```
Depth Map ──→ ControlNet Encoder ──→ Zero-Conv Gates
                                          │
                                          ▼ (added to skip connections)
Noisy Image ──→ U-Net Encoder ──→ Bottleneck ──→ U-Net Decoder ──→ Predicted Noise
                    ▲                                   │
                    └───────── Skip Connections ─────────┘
```

**Base model**: U-Net (75.5M params) with 4 resolution levels, self-attention at 16×16 and 8×8, sinusoidal timestep embeddings, group normalization throughout. Residual blocks with SiLU activations at every level.

**Conditioning**: ControlNet (25.5M params) — a parallel encoder that processes the depth map and injects spatial features into the U-Net's skip connections via zero-convolution gates. The zero-convs are initialized with all weights and biases set to zero, so the conditioning branch outputs nothing at training start. This prevents the injected signal from destroying the base model's representations in early training. The network gradually learns to use the spatial conditioning as training progresses.

**Diffusion**: DDPM noise schedule (linear β from 1e-4 to 0.02, 1000 steps). DDIM accelerated sampling at inference (50 steps, deterministic when η=0).

## Why These Decisions

**Why U-Net over DiT**: A Diffusion Transformer would be the modern production choice — Runway Gen-4.5, WAN 2.2 all use DiT backbones. I chose U-Net because it trains on a single 4090 in reasonable time at this dataset scale. The conditioning mechanism transfers regardless of backbone — what matters is understanding how spatial control signals get injected into the denoising process.

**Why depth conditioning**: In production, I built an environment generation pipeline that used depth maps and wireframe renders from Blender as control signals to texture CG scenes while maintaining 1:1 spatial consistency. This project demonstrates the same principle on a public dataset.

**Why NYU Depth V2**: ~1,449 labeled pairs (50K+ full frames), well-studied benchmark, diverse indoor scenes. Large enough to train a meaningful model, small enough to iterate quickly.

## Results

<!-- TODO: Add after training completes -->
<!-- [Training loss curves from W&B] -->
<!-- [Sample grid: depth input | generated output | ground truth RGB] -->

## Quick Start

### Training

```bash
pip install -r requirements.txt
python scripts/download_data.py          # Download NYU Depth V2 (~2.8 GB)
python scripts/train.py --config configs/train_config.yaml
```

Resume from a checkpoint:
```bash
python scripts/train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_0050.pt
```

### Serving

```bash
# Run directly
python scripts/serve.py --checkpoint checkpoints/checkpoint_final.pt

# Or with Docker
docker compose up --build
```

### API Usage

```bash
# Multipart form upload
curl -X POST http://localhost:8000/generate \
  -F "depth_map=@input_depth.png" \
  -o output.png

# Base64 JSON (for service-to-service calls)
curl -X POST http://localhost:8000/generate_json \
  -H "Content-Type: application/json" \
  -d '{"depth_map_base64": "<base64-encoded-png>"}'

# Batch generation
curl -X POST http://localhost:8000/generate_batch \
  -F "depth_maps=@depth1.png" \
  -F "depth_maps=@depth2.png" \
  -o results.zip
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Depth map (multipart) → PNG image |
| `POST` | `/generate_json` | Base64 depth map (JSON) → base64 image |
| `POST` | `/generate_batch` | Multiple depth maps → ZIP of images |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

## Experiment Tracking

All training runs logged to Weights & Biases:
- Denoising loss per epoch
- Sample outputs every N epochs (same depth inputs for visual comparison)
- Learning rate schedule, GPU memory utilization
- Full hyperparameter configuration

## Project Structure

```
src/
├── model/
│   ├── unet.py              # U-Net with attention (75.5M params)
│   ├── controlnet.py         # ControlNet + zero-conv gates (25.5M params)
│   └── diffusion.py          # DDPM/DDIM noise schedule and sampling
├── data/
│   ├── dataset.py            # NYU Depth V2 paired loader
│   └── transforms.py         # Consistent RGB+depth augmentations
├── training/
│   ├── trainer.py            # Training loop (AMP, EMA, cosine LR, W&B)
│   └── evaluate.py           # Generate comparison grids from checkpoints
└── serving/
    ├── app.py                # FastAPI endpoints
    ├── inference.py          # Checkpoint → DDIM → PIL image
    └── monitoring.py         # Prometheus request/latency/GPU metrics
```

## What I'd Do Differently at Scale

- **DiT backbone**: Replace U-Net with a Diffusion Transformer for better scaling with compute and data
- **Temporal extension**: Add temporal attention layers to condition on video frame sequences
- **Multi-condition**: Accept multiple control signals (depth + edges + normals) with learned weighting
- **Distributed training**: Multi-GPU with FSDP or DeepSpeed for larger models and datasets
- **TensorRT serving**: Quantize and optimize for production inference latency
- **Dataset domain gap**: NYU Depth V2 is heavily biased toward indoor structural scenes — the model overfits to this distribution and will struggle with outdoor landscapes or cinematic environments. A production system needs a diverse multi-domain dataset or domain adaptation. In my production work, I solved this by training on synthetic CG renders from Blender, which gave full control over scene diversity and paired ground truth.

## Stack

PyTorch · FastAPI · Docker · Weights & Biases · Prometheus · CUDA · DDPM/DDIM
