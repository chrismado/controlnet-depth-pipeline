# SPEC: controlnet-depth-pipeline

## Overview
Depth-conditioned image generation using ControlNet-style spatial conditioning. PyTorch, from scratch. No pretrained models, no Hugging Face diffusers.

Given a depth map, generate a plausible textured RGB image that respects the spatial structure.

---

## Phase 1: Model Architecture

### src/model/unet.py
Standard U-Net for denoising diffusion.

Architecture:
- 4 resolution levels: 256→128→64→32→16 (at 256x256 input)
- Channel multipliers: [1, 2, 4, 8] × base_channels (base_channels=64)
- Each level: 2 residual blocks
- Self-attention at 16×16 and 8×8 resolutions (levels 3 and 4)
- Sinusoidal timestep embedding → MLP → injected via addition into each residual block
- Group normalization (num_groups=32), SiLU activations
- Skip connections between encoder and decoder (concatenation, not addition)
- Final output: same spatial dims as input, predicts noise epsilon

Residual block structure:
```
x → GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Dropout → Conv3x3 → + residual
                                      ↑
                              timestep_emb (projected via linear)
```

Attention block structure:
```
x → GroupNorm → reshape to (B, C, H*W) → Q, K, V projections → scaled dot-product attention → reshape back → Conv1x1 → + residual
```

### src/model/controlnet.py
ControlNet conditioning module.

Architecture:
- Copy the U-Net ENCODER only (not decoder)
- Same residual blocks and attention as the U-Net encoder
- Input: depth map (1 channel) → Conv3x3 to match base_channels
- At each encoder level output: zero-conv layer (Conv1x1)
- Returns: list of conditioning features, one per encoder level

CRITICAL — Zero Convolutions:
```python
class ZeroConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)
```
Every connection from ControlNet to U-Net goes through a ZeroConv. This ensures the conditioning signal starts at zero and gradually learns to contribute. WITHOUT THIS THE MODEL WILL NOT TRAIN CORRECTLY.

Integration with U-Net:
- U-Net's forward pass accepts an optional `controlnet_features` list
- At each decoder level, add the corresponding controlnet feature to the skip connection
- When controlnet_features is None, U-Net operates normally (unconditional)

### src/model/diffusion.py
DDPM/DDIM noise scheduling and sampling.

Forward process:
- Linear beta schedule: beta_start=1e-4, beta_end=0.02, T=1000
- Precompute: alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar
- q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

Training loss:
- Sample random t ~ Uniform(0, T)
- Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
- Predict: epsilon_pred = model(x_t, t, controlnet_features)
- Loss: MSE(epsilon_pred, epsilon)

DDPM sampling (training validation):
- Start from x_T ~ N(0, I)
- For t = T, T-1, ..., 1: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * model(x_t, t, cf)) + sigma_t * z

DDIM sampling (inference — faster):
- Same start, but skip steps (e.g., 50 steps instead of 1000)
- Deterministic: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * direction
- where x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)

---

## Phase 2: Data Pipeline

### scripts/download_data.py
- Download NYU Depth V2 labeled dataset (~1449 images for the labeled subset)
- Also support downloading the full raw dataset (~50K) via flag
- Extract to data/nyu_depth_v2/
- Print stats: number of images, depth range, image dimensions

### src/data/dataset.py
- Load paired (RGB, depth) from NYU Depth V2
- RGB: normalize to [-1, 1]
- Depth: normalize to [0, 1] (min-max per image)
- Resize to configurable resolution (default 128×128 for fast iteration, 256×256 for final)
- Return dict: {"rgb": tensor, "depth": tensor}

### src/data/transforms.py
- Random horizontal flip (MUST apply to both RGB and depth with same random state)
- Random crop (same crop coordinates for both)
- Color jitter on RGB only (not depth)
- Normalize (separate params for RGB and depth)

CRITICAL: Use torch manual seed or apply transforms to concatenated [RGB, depth] tensor to guarantee paired consistency.

---

## Phase 3: Training

### configs/train_config.yaml
```yaml
model:
  base_channels: 64
  channel_multipliers: [1, 2, 4, 8]
  attention_resolutions: [16, 8]
  num_res_blocks: 2
  dropout: 0.1

diffusion:
  timesteps: 1000
  beta_start: 1e-4
  beta_end: 0.02
  sampling_method: ddim
  sampling_steps: 50

training:
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 1e-6
  epochs: 200
  image_size: 128  # Start small, scale to 256
  mixed_precision: true
  ema_decay: 0.9999
  gradient_clip: 1.0

  lr_schedule:
    type: cosine
    warmup_epochs: 10

  logging:
    wandb_project: controlnet-depth-pipeline
    log_every_n_steps: 50
    sample_every_n_epochs: 10
    num_eval_samples: 16

  checkpointing:
    save_every_n_epochs: 25
    keep_last_n: 3

data:
  dataset: nyu_depth_v2
  data_dir: data/nyu_depth_v2
  num_workers: 4
  pin_memory: true

paths:
  checkpoint_dir: checkpoints/
  sample_dir: results/samples/
  eval_depth_maps: data/eval_depths/  # Fixed depth maps for consistent evaluation
```

### src/training/trainer.py
- Standard PyTorch training loop
- Mixed precision via torch.cuda.amp.GradScaler and autocast
- EMA: maintain exponential moving average of model weights, use EMA weights for sampling
- W&B logging: loss per step, learning rate, GPU memory, generated samples
- Cosine LR schedule with linear warmup
- Gradient clipping (max_norm=1.0)
- Checkpoint saving: model state, optimizer state, epoch, EMA state, config

### src/training/evaluate.py
- At training start: randomly select and save 16 depth maps from validation set to data/eval_depths/
- Every N epochs: generate RGB from these same 16 depth maps using EMA model
- Save as grid image: [depth_1 | generated_1 | depth_2 | generated_2 | ...]
- Filename: results/samples/epoch_{N:04d}.png
- Log grid to W&B
- Optional: compute FID between generated and real images (use torch-fidelity)

### scripts/train.py
- Parse args: --config path
- Load config from YAML
- Initialize: dataset, dataloader, model, controlnet, diffusion, optimizer, scheduler, trainer
- Run training
- On completion: print best loss, sample output path, W&B run URL

---

## Phase 4: Serving & Deployment

### src/serving/app.py (FastAPI)
Endpoints:
- `POST /generate` — accepts depth map (multipart/form-data image upload), returns generated RGB as PNG
- `POST /generate_batch` — accepts multiple depth maps, returns zip of generated images
- `GET /health` — returns {"status": "healthy", "model_loaded": true, "device": "cuda"}
- `GET /metrics` — Prometheus metrics endpoint

### src/serving/inference.py
- Load model checkpoint + EMA weights on startup
- Preprocess: resize depth map to model resolution, normalize to [0, 1]
- Run DDIM sampling (50 steps by default, configurable)
- Postprocess: denormalize from [-1, 1] to [0, 255], convert to PIL Image
- Return PNG bytes

### src/serving/monitoring.py
Prometheus metrics:
- `request_count` (Counter) — total requests by endpoint
- `request_latency_seconds` (Histogram) — end-to-end latency
- `inference_time_seconds` (Histogram) — model inference time only
- `gpu_memory_bytes` (Gauge) — current GPU memory usage

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# Install Python, copy code, install deps, expose 8000
# Entrypoint: python scripts/serve.py
```

### docker-compose.yml
```yaml
services:
  controlnet:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Phase 5: Tests & CI

### tests/test_model.py
- U-Net forward: random input → output has same spatial dims, correct channels
- ControlNet forward: depth input → returns list of features at expected resolutions
- Zero-conv: verify weights and biases are initialized to zero
- Diffusion: forward process adds noise correctly, shapes are preserved
- Integration: U-Net + ControlNet + diffusion forward pass works end-to-end

### tests/test_data.py
- Dataset loads correctly, returns expected keys and shapes
- Paired augmentations: flip is consistent between RGB and depth
- Normalization ranges: RGB in [-1, 1], depth in [0, 1]

### tests/test_api.py
- /health returns 200 with expected fields
- /generate accepts image, returns image
- /generate with invalid input returns 422
- /metrics returns Prometheus format

### .github/workflows/ci.yml
- Trigger: push to main, pull requests
- Steps: checkout, setup Python 3.10, install deps, ruff lint, pytest (CPU mode)
- No GPU in CI — tests must work on CPU with small tensor sizes

---

## Commit Strategy
Commit after each completed component:
```
feat: unet - U-Net with attention and timestep conditioning
feat: controlnet - ControlNet encoder with zero-conv initialization
feat: diffusion - DDPM/DDIM noise scheduling and sampling
feat: data - NYU Depth V2 dataset loader with paired augmentations
feat: training - training loop with W&B, mixed precision, EMA
feat: evaluate - fixed sample evaluation and grid generation
feat: serving - FastAPI endpoint with DDIM inference
feat: docker - CUDA container with compose
feat: monitoring - Prometheus metrics for inference
feat: tests - model, data, and API test suites
feat: ci - GitHub Actions lint and test workflow
docs: readme - architecture, results, deployment instructions
```
