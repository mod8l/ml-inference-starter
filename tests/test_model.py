"""Unit tests for the tiny ConvNet model."""

import os
import tempfile

import torch

from inference.model import NUM_CLASSES, TinyConvNet, load_model, train_dummy_model


def test_model_forward_shape():
    """The model should return logits for the expected number of classes."""
    model = TinyConvNet()
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (4, NUM_CLASSES)


def test_train_and_load_model():
    """Training should produce a loadable checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        model = train_dummy_model(epochs=1, num_samples=32, save_path=path)
        assert model.training is False
        assert os.path.exists(path)

        loaded = load_model(path)
        assert loaded.training is False
        sample = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(model(sample), loaded(sample))
