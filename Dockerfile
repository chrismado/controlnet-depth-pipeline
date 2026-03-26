FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (install first for Docker layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Default checkpoint location (mount or copy at build time)
# COPY checkpoints/checkpoint_final.pt checkpoints/
RUN mkdir -p checkpoints

# Environment
ENV CHECKPOINT_PATH=checkpoints/checkpoint_final.pt
ENV DEVICE=cuda
ENV DDIM_STEPS=50

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
