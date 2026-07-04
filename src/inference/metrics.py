"""Prometheus instrumentation for the inference service."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Info, generate_latest

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of prediction requests",
    ["model_version", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Prediction request latency in seconds",
    ["model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_INFO = Info("inference_model", "Model metadata")


def set_model_version(version: str) -> None:
    """Expose the model version as a Prometheus info metric label."""
    MODEL_INFO.info({"version": version})


def metrics_response() -> bytes:
    """Return the latest Prometheus exposition format."""
    return generate_latest()
