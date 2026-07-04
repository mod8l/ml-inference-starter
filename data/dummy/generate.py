"""Generate synthetic images and a sample /predict payload."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from inference.model import INPUT_SIZE, NUM_CLASSES


def generate_synthetic_images(
    num_images: int = 10,
    output_dir: str | os.PathLike = "data/dummy/images",
) -> list[Path]:
    """Generate random RGB images and save them as PNG files.

    Args:
        num_images: Number of images to generate.
        output_dir: Directory to save the generated images.

    Returns:
        List of saved image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(num_images):
        array = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        path = output_dir / f"image_{i:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


def encode_image(path: Path) -> str:
    """Return a base64-encoded PNG string for an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_labels(num_labels: int = 10) -> list[int]:
    """Generate random integer labels for synthetic data."""
    return np.random.randint(0, NUM_CLASSES, size=num_labels).tolist()


def main() -> None:
    """Generate images, labels, and a sample JSON payload for /predict."""
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    paths = generate_synthetic_images(num_images=10, output_dir=images_dir)
    labels = generate_labels(num_labels=10)

    sample_payload = {
        "items": [
            {"image_b64": encode_image(path), "label": label}
            for path, label in zip(paths, labels)
        ],
        "top_k": NUM_CLASSES,
    }

    sample_path = base_dir / "sample_input.json"
    with open(sample_path, "w") as f:
        json.dump(sample_payload, f, indent=2)

    print(f"Generated {len(paths)} images in {images_dir}")
    print(f"Saved sample payload to {sample_path}")


if __name__ == "__main__":
    main()
