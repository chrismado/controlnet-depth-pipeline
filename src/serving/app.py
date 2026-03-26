"""
FastAPI application for depth-conditioned image generation.

Endpoints:
    POST /generate        — Accept a depth map (multipart), return generated RGB image.
    POST /generate_json   — Accept a base64-encoded depth map (JSON), return base64 image.
    POST /generate_batch  — Accept multiple depth maps (multipart), return ZIP of images.
    GET  /health          — Health check.
    GET  /metrics         — Prometheus metrics.
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .inference import InferencePipeline
from .monitoring import REQUEST_COUNT, REQUEST_LATENCY, get_metrics

# Global pipeline instance (initialised on startup)
pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global pipeline

    checkpoint = os.environ.get("CHECKPOINT_PATH", "checkpoints/checkpoint_final.pt")
    device = os.environ.get("DEVICE", "cuda")
    ddim_steps = int(os.environ.get("DDIM_STEPS", "50"))

    print(f"Loading model from {checkpoint} ...")
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint,
        device=device,
        ddim_steps=ddim_steps,
    )
    print("Model loaded. Ready to serve.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="ControlNet Depth Pipeline",
    description="Depth-conditioned image generation via ControlNet + DDPM/DDIM",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    output, content_type = get_metrics()
    return Response(content=output, media_type=content_type)


@app.post("/generate")
async def generate(depth_map: UploadFile = File(...)):
    """Generate an RGB image from a single depth map.

    Accepts: image file (PNG/JPEG) as multipart form data.
    Returns: generated RGB image as PNG.
    """
    start = time.perf_counter()
    try:
        # Read and validate input
        contents = await depth_map.read()
        try:
            depth_image = Image.open(io.BytesIO(contents))
        except Exception:
            REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
            raise HTTPException(status_code=400, detail="Invalid image file")

        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Generate
        result = pipeline.generate(depth_image)

        # Encode output as PNG
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)

        REQUEST_COUNT.labels(endpoint="/generate", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/generate").observe(time.perf_counter() - start)

        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


class GenerateRequest(BaseModel):
    """JSON request body for base64-encoded depth map input."""
    depth_map_base64: str
    ddim_steps: int | None = None


class GenerateResponse(BaseModel):
    """JSON response body with base64-encoded generated image."""
    image_base64: str
    format: str = "png"


@app.post("/generate_json", response_model=GenerateResponse)
async def generate_json(request: GenerateRequest):
    """Generate an RGB image from a base64-encoded depth map.

    Accepts: JSON with base64-encoded depth map image.
    Returns: JSON with base64-encoded generated PNG image.

    Common pattern in production MLOps microservices where multipart
    form data isn't convenient (e.g. service-to-service calls).
    """
    start = time.perf_counter()
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Decode base64 input
        try:
            depth_bytes = base64.b64decode(request.depth_map_base64)
            depth_image = Image.open(io.BytesIO(depth_bytes))
        except Exception:
            REQUEST_COUNT.labels(endpoint="/generate_json", status="error").inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 image data. Expected a base64-encoded PNG/JPEG.",
            )

        # Generate
        result = pipeline.generate(depth_image, ddim_steps=request.ddim_steps)

        # Encode output as base64 PNG
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        REQUEST_COUNT.labels(endpoint="/generate_json", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/generate_json").observe(time.perf_counter() - start)

        return GenerateResponse(image_base64=image_b64)

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate_json", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_batch")
async def generate_batch(depth_maps: list[UploadFile] = File(...)):
    """Generate RGB images from multiple depth maps.

    Accepts: multiple image files as multipart form data.
    Returns: ZIP archive containing generated PNG images.
    """
    import zipfile

    start = time.perf_counter()
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(depth_maps) == 0:
            raise HTTPException(status_code=400, detail="No depth maps provided")

        if len(depth_maps) > 16:
            raise HTTPException(status_code=400, detail="Maximum 16 images per batch")

        # Read all depth maps
        depth_images = []
        for dm in depth_maps:
            contents = await dm.read()
            try:
                depth_images.append(Image.open(io.BytesIO(contents)))
            except Exception:
                REQUEST_COUNT.labels(endpoint="/generate_batch", status="error").inc()
                raise HTTPException(status_code=400, detail=f"Invalid image: {dm.filename}")

        # Generate batch
        results = pipeline.generate_batch(depth_images)

        # Pack into ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, img in enumerate(results):
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"generated_{i:03d}.png", img_buf.getvalue())
        buf.seek(0)

        REQUEST_COUNT.labels(endpoint="/generate_batch", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/generate_batch").observe(time.perf_counter() - start)

        return StreamingResponse(buf, media_type="application/zip")

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate_batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
