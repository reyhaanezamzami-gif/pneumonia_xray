"""Display representative chest X-ray images from each class."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.config import CLASS_NAMES, RAW_DATA_DIR, FIGURES_DIR
from src.utils import get_logger

LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def find_sample(class_name: str) -> Path:
    """Return the first supported image found for a class."""
    class_dir = RAW_DATA_DIR / "train" / class_name
    images = sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {class_dir}")
    return images[0]


def main() -> None:
    """Plot and save one sample image from each class."""
    figure, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(10, 5))

    for axis, class_name in zip(axes, CLASS_NAMES):
        image_path = find_sample(class_name)
        with Image.open(image_path) as image:
            axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(class_name)
        axis.axis("off")
        LOGGER.info("Sample selected: %s", image_path)

    figure.suptitle("Chest X-ray Training Samples")
    figure.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "sample_images.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    LOGGER.info("Sample figure saved to %s", output_path)
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
