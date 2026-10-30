"""Image preprocessing utilities for chest X-ray images."""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import IMAGE_SIZE
from src.utils import get_logger

LOGGER = get_logger(__name__)


def load_preprocessed_image(
    image_path: Path,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Load an image as RGB and resize it to a square array."""
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.float32)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        LOGGER.error("Could not load image %s: %s", image_path, exc)
        raise


def preprocess_image_batch(
    image_paths: list[Path],
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Load and preprocess multiple images into one array."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    images = [
        load_preprocessed_image(path, image_size)
        for path in image_paths
    ]
    return np.stack(images, axis=0)
