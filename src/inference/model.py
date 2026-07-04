"""Tiny deterministic ConvNet for 3-class classification on 32x32 images."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as func

NUM_CLASSES = 3
INPUT_SIZE = 32


class TinyConvNet(nn.Module):
    """A small deterministic ConvNet for synthetic image classification.

    I keep the architecture intentionally shallow so the model trains in seconds
    on CPU and remains easy to reason about during debugging.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(func.relu(self.conv1(x)))  # 16 x 16
        x = self.pool(func.relu(self.conv2(x)))  # 8 x 8
        x = self.pool(func.relu(self.conv3(x)))  # 4 x 4
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        return self.fc2(x)


def train_dummy_model(
    epochs: int = 5,
    batch_size: int = 16,
    num_samples: int = 256,
    save_path: str | os.PathLike = "models/dummy_classifier.pt",
    seed: int = 42,
) -> TinyConvNet:
    """Train the tiny ConvNet on in-memory random data and save the checkpoint.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        num_samples: Number of synthetic samples to generate.
        save_path: Destination for the saved checkpoint.
        seed: Random seed for reproducibility.

    Returns:
        The trained model in eval mode.
    """
    torch.manual_seed(seed)
    model = TinyConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        inputs = torch.randn(num_samples, 3, INPUT_SIZE, INPUT_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
        },
        save_path,
    )
    model.eval()
    return model


def load_model(checkpoint_path: str | os.PathLike) -> TinyConvNet:
    """Load a TinyConvNet from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Model in eval mode with weights loaded.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TinyConvNet(num_classes=checkpoint.get("num_classes", NUM_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
