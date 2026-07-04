"""Preprocess and postprocess helpers for the inference pipeline."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from inference.model import INPUT_SIZE, NUM_CLASSES

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


def preprocess_batch(payload: list[dict[str, Any]]) -> torch.Tensor:
    """Convert a batch payload into a normalized tensor.

    Each item in ``payload`` must contain either:
      - ``image_b64``: a base64-encoded PNG/JPEG image, or
      - ``flat``: a list of length ``INPUT_SIZE * INPUT_SIZE * 3`` representing
        a CHW or HWC image.

    Args:
        payload: List of input dictionaries.

    Returns:
        A tensor of shape ``(batch_size, 3, INPUT_SIZE, INPUT_SIZE)``.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )

    tensors: list[torch.Tensor] = []
    for item in payload:
        if "image_b64" in item:
            raw = base64.b64decode(item["image_b64"])
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            tensors.append(transform(image))
        elif "flat" in item:
            flat = np.asarray(item["flat"], dtype=np.float32)
            expected = INPUT_SIZE * INPUT_SIZE * 3
            if flat.size != expected:
                raise ValueError(f"flat array must have {expected} elements, got {flat.size}")
            # Accept either HWC or CHW by checking shape heuristics.
            if flat.shape == (3, INPUT_SIZE, INPUT_SIZE):
                arr = flat
            else:
                arr = flat.reshape(INPUT_SIZE, INPUT_SIZE, 3).transpose(2, 0, 1)
            tensor = torch.from_numpy(arr) / 255.0
            tensor = transforms.Normalize(mean=_MEAN, std=_STD)(tensor)
            tensors.append(tensor)
        else:
            raise ValueError("Each input must contain 'image_b64' or 'flat'")

    return torch.stack(tensors)


def postprocess(logits: torch.Tensor, top_k: int = NUM_CLASSES) -> list[dict[str, Any]]:
    """Apply softmax and return top-k class probabilities.

    Args:
        logits: Raw model outputs of shape ``(batch_size, num_classes)``.
        top_k: Number of top predictions to return per sample.

    Returns:
        List of prediction dictionaries, one per sample.
    """
    probs = torch.softmax(logits, dim=1)
    top = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)
    return [
        {
            "probabilities": {
                str(int(idx)): float(round(p, 6))
                for idx, p in zip(top.indices[i].tolist(), top.values[i].tolist(), strict=False)
            },
            "top_class": int(top.indices[i][0]),
            "confidence": float(round(top.values[i][0].item(), 6)),
        }
        for i in range(probs.size(0))
    ]
