"""Dataset inspection, image paths, and stratified data splitting."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    DATASET_SUMMARY_PATH,
    RAW_DATA_DIR,
    SEED,
    SPLITS_DIR,
    VALIDATION_SIZE,
)
from src.utils import get_logger, save_json

LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def inspect_dataset(
    dataset_dir: Path = RAW_DATA_DIR,
) -> dict[str, Any]:
    """Count supported images by split and class."""
    summary: dict[str, Any] = {"dataset_dir": str(dataset_dir), "splits": {}}

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_summary: dict[str, Any] = {}
        total = 0

        if not split_dir.is_dir():
            LOGGER.warning("Missing dataset split: %s", split_dir)
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            count = (
                sum(
                    1
                    for path in class_dir.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if class_dir.is_dir()
                else 0
            )
            split_summary[class_name] = count
            total += count

        split_summary["total"] = total
        summary["splits"][split] = split_summary
        LOGGER.info("%s image counts: %s", split, split_summary)

    return summary


def save_dataset_summary(
    dataset_dir: Path = RAW_DATA_DIR,
) -> dict[str, Any]:
    """Inspect the dataset and save its summary as JSON."""
    summary = inspect_dataset(dataset_dir)
    save_json(summary, DATASET_SUMMARY_PATH)
    LOGGER.info("Dataset summary saved to %s", DATASET_SUMMARY_PATH)
    return summary


def collect_image_paths(
    split: str = "train",
    dataset_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Collect image paths and numeric labels from a dataset split."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    records: list[dict[str, Any]] = []
    split_dir = dataset_dir / split

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "image_path": str(path.resolve()),
                        "class_name": class_name,
                        "label": CLASS_TO_LABEL[class_name],
                    }
                )

    if not records:
        raise ValueError(f"No images found in {split_dir}")

    return pd.DataFrame(records)


def create_data_splits() -> dict[str, Any]:
    """Create reproducible stratified train, validation, and test CSV files."""
    all_train = collect_image_paths("train")
    test_data = collect_image_paths("test")

    train_data, validation_data = train_test_split(
        all_train,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=all_train["label"],
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(SPLITS_DIR / "train.csv", index=False)
    validation_data.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test_data.to_csv(SPLITS_DIR / "test.csv", index=False)

    summary: dict[str, Any] = {}
    for name, frame in (
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
    ):
        counts = frame["class_name"].value_counts().to_dict()
        summary[name] = {
            "total": len(frame),
            "NORMAL": int(counts.get("NORMAL", 0)),
            "PNEUMONIA": int(counts.get("PNEUMONIA", 0)),
        }
        LOGGER.info("%s split: %s", name, summary[name])

    save_json(summary, SPLITS_DIR / "split_summary.json")
    return summary


if __name__ == "__main__":
    save_dataset_summary()
    create_data_splits()
