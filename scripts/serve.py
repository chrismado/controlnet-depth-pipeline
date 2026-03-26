#!/usr/bin/env python3
"""Entry point for the inference server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --checkpoint checkpoints/checkpoint_final.pt --port 8000

Environment variables (also settable via CLI):
    CHECKPOINT_PATH  — path to model checkpoint
    DEVICE           — 'cuda' or 'cpu'
    DDIM_STEPS       — number of DDIM sampling steps
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the inference server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_final.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Set env vars for the FastAPI lifespan handler
    os.environ.setdefault("CHECKPOINT_PATH", args.checkpoint)
    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("DDIM_STEPS", str(args.ddim_steps))

    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
