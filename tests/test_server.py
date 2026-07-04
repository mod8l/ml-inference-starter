"""Unit tests for the FastAPI inference server."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference.model import INPUT_SIZE, NUM_CLASSES
from inference.server import app


@pytest.fixture
def client():
    """Return a TestClient with the model loaded via lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def make_b64_image() -> str:
    """Create a small random RGB image encoded as base64 PNG."""
    array = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_health(client):
    """The health endpoint should report healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready(client):
    """The ready endpoint should report ready after model load."""
    response = client.get("/ready")
    assert response.status_code == 200


def test_predict_b64(client):
    """The predict endpoint should return class probabilities."""
    payload = {"items": [{"image_b64": make_b64_image()} for _ in range(2)]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["top_class"] in range(NUM_CLASSES)


def test_predict_flat(client):
    """The predict endpoint should accept flat array inputs."""
    flat = list(np.random.randint(0, 256, (INPUT_SIZE * INPUT_SIZE * 3)).tolist())
    payload = {"items": [{"flat": flat}], "top_k": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_invalid_input(client):
    """The predict endpoint should reject invalid input."""
    payload = {"items": [{}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_metrics(client):
    """The metrics endpoint should expose Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inference_requests_total" in response.text
