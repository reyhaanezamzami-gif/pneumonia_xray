
from pathlib import Path
import json
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"

RANDOM_STATE = 42
N_SPLITS = 7
TARGET_VALIDATION_SIZE = 0.15


def get_group(row):
    filename = Path(row["image_path"]).stem.lower()

    if row["class_name"] == "PNEUMONIA":
        match = re.search(
            r"(person\d+_(?:bacteria|virus))",
            filename,
        )
        if not match:
            raise ValueError(f"Cannot identify group: {filename}")
        return "PNEUMONIA_" + match.group(1)

    if row["class_name"] == "NORMAL":
        return "NORMAL_" + re.sub(r"-\d{4}$", "", filename)

    raise ValueError(f"Unknown class: {row['class_name']}")


def get_counts(frame):
    counts = frame["class_name"].value_counts().to_dict()
    return {
        "total": int(len(frame)),
        "NORMAL": int(counts.get("NORMAL", 0)),
        "PNEUMONIA": int(counts.get("PNEUMONIA", 0)),
    }


def main():
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"
    summary_path = SPLITS_DIR / "split_summary.json"

    # Load original development data
    train = pd.read_csv(train_path)
    validation = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    data = pd.concat([train, validation], ignore_index=True)

    required_columns = {"image_path", "class_name", "label"}
    for name, frame in [
        ("development", data),
        ("test", test),
    ]:
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"Invalid columns in {name} CSV")

    if data["image_path"].duplicated().any():
        raise ValueError("Duplicate paths in development data")

    if test["image_path"].duplicated().any():
        raise ValueError("Duplicate paths in test data")

    data["group"] = data.apply(get_group, axis=1)
    test["group"] = test.apply(get_group, axis=1)

    # Find the best group-aware split
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_split = None
    best_score = float("inf")

    for train_idx, val_idx in splitter.split(
        data,
        data["label"],
        groups=data["group"],
    ):
        candidate_train = data.iloc[train_idx]
        candidate_val = data.iloc[val_idx]

        size_error = abs(
            len(candidate_val) / len(data)
            - TARGET_VALIDATION_SIZE
        )

        class_error = abs(
            candidate_val["label"].mean()
            - data["label"].mean()
        )

        score = size_error + class_error

        if score < best_score:
            best_score = score
            best_split = (
                candidate_train.copy(),
                candidate_val.copy(),
            )

    new_train, new_validation = best_split

    # Validate group separation
    train_val_overlap = (
        set(new_train["group"]) & set(new_validation["group"])
    )
    train_test_overlap = (
        set(new_train["group"]) & set(test["group"])
    )
    val_test_overlap = (
        set(new_validation["group"]) & set(test["group"])
    )

    original_paths = set(data["image_path"])
    new_paths = (
        set(new_train["image_path"])
        | set(new_validation["image_path"])
    )

    missing_paths = original_paths - new_paths
    duplicate_paths = (
        set(new_train["image_path"])
        & set(new_validation["image_path"])
    )

    checks_passed = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
        and len(missing_paths) == 0
        and len(duplicate_paths) == 0
    )

    print("GROUP-AWARE SPLIT REPORT")
    print("-" * 40)
    print("Development:", len(data))
    print("Train:", len(new_train))
    print("Validation:", len(new_validation))
    print("Test:", len(test))
    print("Validation percentage:",
          round(len(new_validation) / len(data) * 100, 2))

    print("\nGroup overlaps:")
    print("Train-Validation:", len(train_val_overlap))
    print("Train-Test:", len(train_test_overlap))
    print("Validation-Test:", len(val_test_overlap))

    print("\nTrain classes:")
    print(new_train["class_name"].value_counts())

    print("\nValidation classes:")
    print(new_validation["class_name"].value_counts())

    print("\nMissing images:", len(missing_paths))
    print("Duplicate images:", len(duplicate_paths))

    if not checks_passed:
        raise RuntimeError(
            "Validation failed. Original files were not changed."
        )

    # Remove internal group column before saving
    new_train = new_train.drop(columns=["group"])
    new_validation = new_validation.drop(columns=["group"])

    # Prepare new summary
    summary = {
        "train": get_counts(new_train),
        "validation": get_counts(new_validation),
        "test": get_counts(test),
        "split_method": "StratifiedGroupKFold",
        "random_state": RANDOM_STATE,
        "validation_target": TARGET_VALIDATION_SIZE,
        "grouping": {
            "PNEUMONIA": "person number + bacteria/virus",
            "NORMAL": "filename without final four-digit suffix",
        },
        "group_overlap_checks": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }

    # Write CSV files only after all checks pass
    train_temp = SPLITS_DIR / "train_new.csv"
    val_temp = SPLITS_DIR / "validation_new.csv"
    summary_temp = SPLITS_DIR / "split_summary_new.json"

    new_train.to_csv(train_temp, index=False)
    new_validation.to_csv(val_temp, index=False)

    with open(summary_temp, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Replace original development files
    train_temp.replace(train_path)
    val_temp.replace(val_path)
    summary_temp.replace(summary_path)

    # Remove stale feature caches
    for cache_name in [
        "train_features.npz",
        "validation_features.npz",
    ]:
        cache_path = SPLITS_DIR / cache_name
        if cache_path.exists():
            cache_path.unlink()

    print("\n" + "=" * 40)
    print("RESULT: ALL CHECKS PASSED")
    print("New split files saved successfully.")
    print("Test CSV was not modified.")
    print("Old feature caches were removed.")
    print("=" * 40)


if __name__ == "__main__":
    main()
