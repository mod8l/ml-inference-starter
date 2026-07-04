"""Minimal Locust load-test script for the /predict endpoint."""

from __future__ import annotations

import base64
import io

import numpy as np
from locust import HttpUser, between, task
from PIL import Image

from inference.model import INPUT_SIZE


def make_b64_image() -> str:
    """Generate a random base64-encoded PNG image."""
    array = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class InferenceUser(HttpUser):
    """Simulated user that calls the inference endpoints."""

    wait_time = between(0.5, 2.0)

    @task(1)
    def health(self) -> None:
        """Hit the health probe."""
        self.client.get("/health")

    @task(3)
    def predict(self) -> None:
        """Send a small batch to /predict."""
        payload = {
            "items": [{"image_b64": make_b64_image()} for _ in range(2)],
            "top_k": 3,
        }
        self.client.post("/predict", json=payload)
