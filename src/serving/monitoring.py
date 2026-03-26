"""
Prometheus metrics for the inference service.

Tracks:
- Request count (total and by endpoint)
- Latency histogram (seconds)
- GPU memory usage (bytes)
- Inference time histogram (model forward pass only)
"""

import time
from functools import wraps
from typing import Callable

import torch
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# --- Metrics ---

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

INFERENCE_TIME = Histogram(
    "inference_model_seconds",
    "Model inference time (forward pass only)",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

GPU_MEMORY_BYTES = Gauge(
    "gpu_memory_allocated_bytes",
    "Current GPU memory allocated",
)

GPU_MEMORY_RESERVED_BYTES = Gauge(
    "gpu_memory_reserved_bytes",
    "Current GPU memory reserved by allocator",
)


def update_gpu_metrics() -> None:
    """Update GPU memory gauges if CUDA is available."""
    if torch.cuda.is_available():
        GPU_MEMORY_BYTES.set(torch.cuda.memory_allocated())
        GPU_MEMORY_RESERVED_BYTES.set(torch.cuda.memory_reserved())


def get_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics output."""
    update_gpu_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST


class InferenceTimer:
    """Context manager that records model inference time to the histogram."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        INFERENCE_TIME.observe(elapsed)
