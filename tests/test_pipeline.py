"""Unit tests for preprocess and postprocess helpers."""

import base64
import io

import numpy as np
import torch
from PIL import Image

from inference.model import INPUT_SIZE, NUM_CLASSES
from inference.pipeline import postprocess, preprocess_batch


def make_b64_image() -> str:
    """Create a small random RGB image encoded as base64 PNG."""
    array = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_preprocess_b64_batch():
    """Preprocessing should return a normalized batch tensor."""
    payload = [{"image_b64": make_b64_image()} for _ in range(3)]
    tensor = preprocess_batch(payload)
    assert tensor.shape == (3, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_flat_batch():
    """Preprocessing should accept a flat array representation."""
    flat = list(np.random.randint(0, 256, (INPUT_SIZE * INPUT_SIZE * 3)).tolist())
    tensor = preprocess_batch([{"flat": flat}])
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_postprocess_sums_to_one():
    """Postprocessed probabilities should sum to one."""
    logits = torch.randn(2, NUM_CLASSES)
    results = postprocess(logits)
    assert len(results) == 2
    for result in results:
        assert sum(result["probabilities"].values()) == pytest_approx(1.0, abs=1e-5)
        assert 0 <= result["top_class"] < NUM_CLASSES


def pytest_approx(value, **kwargs):
    """Compatibility helper for pytest.approx."""
    import pytest

    return pytest.approx(value, **kwargs)
