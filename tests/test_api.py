"""Tests for the FastAPI serving endpoints.

Uses FastAPI's TestClient for synchronous testing without needing a running
server or model checkpoint. The model is monkey-patched with a dummy pipeline.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch the pipeline before importing the app
import src.serving.app as app_module


class DummyPipeline:
    """Fake inference pipeline that returns a solid-colour image."""

    def __init__(self):
        self.image_size = 64

    def generate(self, depth_image, ddim_steps=None):
        return Image.new("RGB", (64, 64), color=(0, 128, 255))

    def generate_batch(self, depth_images, ddim_steps=None):
        return [Image.new("RGB", (64, 64), color=(0, 128, 255)) for _ in depth_images]


@pytest.fixture(autouse=True)
def mock_pipeline():
    """Inject a dummy pipeline into the app module for all tests."""
    app_module.pipeline = DummyPipeline()
    yield
    app_module.pipeline = None


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def _make_depth_png() -> bytes:
    """Create a minimal valid PNG depth map."""
    buf = io.BytesIO()
    Image.new("L", (64, 64), color=128).save(buf, format="PNG")
    return buf.getvalue()


# ---------- Health ----------


def test_health(client):
    """GET /health should return 200 with status healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ---------- Generate ----------


def test_generate_success(client):
    """POST /generate with valid depth map should return a PNG image."""
    depth_bytes = _make_depth_png()
    resp = client.post(
        "/generate",
        files={"depth_map": ("depth.png", depth_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # Verify it's a valid PNG
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (64, 64)


def test_generate_invalid_input(client):
    """POST /generate with non-image data should return 400."""
    resp = client.post(
        "/generate",
        files={"depth_map": ("bad.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_generate_no_model(client):
    """POST /generate with no model loaded should return 503."""
    app_module.pipeline = None
    resp = client.post(
        "/generate",
        files={"depth_map": ("depth.png", _make_depth_png(), "image/png")},
    )
    assert resp.status_code == 503


# ---------- Generate JSON (base64) ----------


def test_generate_json_success(client):
    """POST /generate_json with base64-encoded depth map should return base64 image."""
    import base64

    depth_b64 = base64.b64encode(_make_depth_png()).decode("utf-8")
    resp = client.post("/generate_json", json={"depth_map_base64": depth_b64})
    assert resp.status_code == 200

    data = resp.json()
    assert "image_base64" in data
    assert data["format"] == "png"

    # Decode and verify it's a valid image
    img_bytes = base64.b64decode(data["image_base64"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (64, 64)


def test_generate_json_invalid_base64(client):
    """POST /generate_json with invalid base64 should return 400."""
    resp = client.post("/generate_json", json={"depth_map_base64": "not-valid-base64!!!"})
    assert resp.status_code == 400


# ---------- Generate Batch ----------


def test_generate_batch_success(client):
    """POST /generate_batch with multiple depth maps should return a ZIP."""
    depth_bytes = _make_depth_png()
    files = [
        ("depth_maps", ("d1.png", depth_bytes, "image/png")),
        ("depth_maps", ("d2.png", depth_bytes, "image/png")),
    ]
    resp = client.post("/generate_batch", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    # Verify ZIP contents
    import zipfile

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert len(zf.namelist()) == 2


# ---------- Metrics ----------


def test_metrics(client):
    """GET /metrics should return Prometheus text format."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format contains '# HELP' and '# TYPE' lines
    text = resp.text
    assert "inference_requests_total" in text or "# HELP" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
