"""FastAPI inference server with health checks and Prometheus metrics."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from inference import __version__
from inference.metrics import REQUEST_COUNT, REQUEST_LATENCY, set_model_version
from inference.model import NUM_CLASSES, TinyConvNet, load_model, train_dummy_model
from inference.pipeline import postprocess, preprocess_batch

_MODEL: TinyConvNet | None = None
_MODEL_VERSION = os.environ.get("MODEL_VERSION", __version__)
_MODEL_PATH = os.environ.get("MODEL_PATH", "models/dummy_classifier.pt")


class PredictRequest(BaseModel):
    """Batch prediction request body."""

    items: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of inputs, each containing 'image_b64' or 'flat'.",
    )
    top_k: int = Field(default=NUM_CLASSES, ge=1, le=NUM_CLASSES)


class Prediction(BaseModel):
    """Single prediction result."""

    probabilities: dict[str, float]
    top_class: int
    confidence: float


class PredictResponse(BaseModel):
    """Batch prediction response."""

    model_version: str
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or train the model once at startup."""
    global _MODEL
    if not os.path.exists(_MODEL_PATH):
        _MODEL = train_dummy_model(save_path=_MODEL_PATH)
    else:
        _MODEL = load_model(_MODEL_PATH)
    set_model_version(_MODEL_VERSION)
    yield
    _MODEL = None


app = FastAPI(
    title="ML Inference Starter",
    description="Reference architecture for a containerized ML inference service.",
    version=__version__,
    docs_url="/docs",
    lifespan=lifespan,
)


@app.get("/health", tags=["probes"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "healthy"}


@app.get("/ready", tags=["probes"])
async def ready() -> Response:
    """Readiness probe; returns 503 until the model is loaded."""
    if _MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return Response(content='{"status":"ready"}', media_type="application/json")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on a batch of inputs."""
    if _MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start = time.perf_counter()
    status_label = "success"
    try:
        inputs = preprocess_batch(request.items)
        with torch.no_grad():
            logits = _MODEL(inputs)
        predictions = postprocess(logits, top_k=request.top_k)
    except (ValueError, RuntimeError) as exc:
        status_label = "error"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        REQUEST_COUNT.labels(model_version=_MODEL_VERSION, status=status_label).inc()
        REQUEST_LATENCY.labels(model_version=_MODEL_VERSION).observe(time.perf_counter() - start)

    return PredictResponse(
        model_version=_MODEL_VERSION,
        predictions=predictions,
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
async def metrics() -> bytes:
    """Prometheus metrics endpoint."""
    from inference.metrics import metrics_response

    return metrics_response()


def main() -> None:
    """CLI entry point for the inference server."""
    import uvicorn

    uvicorn.run(
        "inference.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
