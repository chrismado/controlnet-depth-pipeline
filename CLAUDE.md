# ControlNet Depth Pipeline

## Project
Depth-conditioned image generation using ControlNet-style spatial conditioning in PyTorch. Built from scratch — no pretrained Stable Diffusion, no Hugging Face diffusers.

## Stack
- Python 3.10+
- PyTorch 2.x (CUDA)
- FastAPI (serving)
- Docker (containerization)
- Weights & Biases (experiment tracking)
- Prometheus (monitoring)
- pytest + ruff (testing/linting)

## Structure
```
controlnet-depth-pipeline/
├── CLAUDE.md
├── SPEC.md
├── TASKS.md
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── train_config.yaml
├── src/
│   ├── model/
│   │   ├── unet.py              # U-Net with attention blocks
│   │   ├── controlnet.py        # ControlNet conditioning module (ZERO-CONV CRITICAL)
│   │   └── diffusion.py         # DDPM/DDIM noise scheduling and sampling
│   ├── data/
│   │   ├── dataset.py           # NYU Depth V2 loader
│   │   └── transforms.py        # Augmentations (must apply consistently to RGB+depth pairs)
│   ├── training/
│   │   ├── trainer.py           # Training loop with W&B, mixed precision, EMA
│   │   └── evaluate.py          # Sample generation from fixed depth maps, optional FID
│   └── serving/
│       ├── app.py               # FastAPI endpoints
│       ├── inference.py         # Model loading, DDIM sampling, pre/post processing
│       └── monitoring.py        # Prometheus metrics
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── serve.py                 # Serving entry point
│   └── download_data.py         # NYU Depth V2 download and prep
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_data.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Conventions
- All hyperparameters in configs/train_config.yaml, never hardcoded
- Type hints on all function signatures
- Docstrings on all public methods (Google style)
- No Hugging Face wrappers — implement from scratch
- No pretrained weights — train from scratch on NYU Depth V2
- Commit after each completed component with message format: `feat: [component] - [what it does]`

## Critical Implementation Rules
1. **Zero-convolutions in controlnet.py**: The conv layers connecting ControlNet to U-Net MUST be initialized with `nn.init.zeros_` on BOTH weights AND biases. This is the defining ControlNet mechanism. Without it, conditioning destroys the base model in early training.
2. **Paired augmentations**: Any spatial augmentation (flip, crop, rotate) must be applied identically to both RGB and depth map. Use shared random state or apply to concatenated tensor.
3. **DDIM at inference**: Train with DDPM (1000 steps) but serve with DDIM (50 steps) for practical inference speed.
4. **Fixed evaluation samples**: Save 8-16 depth maps at training start. Generate from these same inputs every N epochs to show progression. Save to results/samples/ with epoch number in filename.
5. **Mixed precision**: Use torch.cuda.amp throughout training. The 4090 has good fp16 throughput.

## Current Sprint
Building the full pipeline from scratch. See TASKS.md for execution order.
