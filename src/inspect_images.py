"""Inspect image readability, dimensions, and color modes."""

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, RAW_DATA_DIR, METRICS_DIR
from src.utils import get_logger, save_json

LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def inspect_images(
    dataset_dir: Path = RAW_DATA_DIR,
) -> dict[str, Any]:
    """Inspect image files and summarize dimensions and modes."""
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "checked_images": 0,
        "unreadable_images": [],
        "splits": {},
    }

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_summary: dict[str, Any] = {
            "classes": {},
            "total": 0,
        }

        if not split_dir.is_dir():
            LOGGER.warning("Missing split directory: %s", split_dir)
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            dimensions: Counter[str] = Counter()
            modes: Counter[str] = Counter()
            checked = 0

            if class_dir.is_dir():
                paths = (
                    path for path in class_dir.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )

                for path in paths:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            dimensions[f"{image.width}x{image.height}"] += 1
                            modes[image.mode] += 1
                        checked += 1
                    except (OSError, ValueError, UnidentifiedImageError) as exc:
                        summary["unreadable_images"].append({
                            "path": str(path),
                            "error": str(exc),
                        })

            split_summary["classes"][class_name] = {
                "readable_images": checked,
                "dimensions": dict(dimensions),
                "color_modes": dict(modes),
            }
            split_summary["total"] += checked
            summary["checked_images"] += checked

        summary["splits"][split] = split_summary
        LOGGER.info("%s inspection completed: %s", split, split_summary)

    summary["unreadable_count"] = len(summary["unreadable_images"])
    output_path = METRICS_DIR / "image_inspection.json"
    save_json(summary, output_path)
    LOGGER.info("Inspection summary saved to %s", output_path)
    LOGGER.info("Unreadable images: %d", summary["unreadable_count"])
    return summary


if __name__ == "__main__":
    inspect_images()
