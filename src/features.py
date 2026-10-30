"""Feature extraction utilities using HOG, LBP, and image statistics."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog, local_binary_pattern

from src.config import IMAGE_SIZE
from src.utils import get_logger

LOGGER = get_logger(__name__)

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"


def extract_features(
    image_path: Path,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Extract HOG, LBP histogram, and intensity statistics from an image."""
    try:
        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )
            gray_image = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        LOGGER.error("Could not read image %s: %s", image_path, exc)
        raise

    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    lbp_image = local_binary_pattern(
        gray_image,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD,
    )
    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, LBP_POINTS + 4),
        range=(0, LBP_POINTS + 2),
    )
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)

    intensity_stats = np.array(
        [
            gray_image.mean() / 255.0,
            gray_image.std() / 255.0,
            np.percentile(gray_image, 25) / 255.0,
            np.percentile(gray_image, 50) / 255.0,
            np.percentile(gray_image, 75) / 255.0,
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [hog_features.astype(np.float32), histogram, intensity_stats]
    )
    return features


def extract_feature_batch(
    image_paths: list[Path],
) -> np.ndarray:
    """Extract feature vectors for a collection of image paths."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    feature_rows: list[np.ndarray] = []
    for image_path in image_paths:
        feature_rows.append(extract_features(image_path))

    return np.vstack(feature_rows)


def get_feature_metadata() -> dict[str, Any]:
    """Return the settings used for feature extraction."""
    return {
        "image_size": IMAGE_SIZE,
        "hog_orientations": 9,
        "hog_pixels_per_cell": [16, 16],
        "hog_cells_per_block": [2, 2],
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
        "lbp_method": LBP_METHOD,
        "statistics": [
            "mean_intensity",
            "std_intensity",
            "percentile_25",
            "percentile_50",
            "percentile_75",
        ],
    }
