"""Train baseline SVM and Random Forest models using HOG/LBP features."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.config import (
    BASELINE_METRICS_PATH,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SPLITS_DIR,
    SVM_MODEL_PATH,
    RF_MODEL_PATH,
)
from src.features import extract_features
from src.utils import get_logger, save_json, set_seed

LOGGER = get_logger(__name__)


def load_features(
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached features or extract them from a split CSV."""
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    cache_path = SPLITS_DIR / f"{split_name}_features.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    frame = pd.read_csv(csv_path)
    feature_rows: list[np.ndarray] = []

    for image_path in tqdm(
        frame["image_path"],
        desc=f"Extracting {split_name} features",
    ):
        feature_rows.append(extract_features(Path(image_path)))

    X = np.vstack(feature_rows).astype(np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)

    np.savez_compressed(cache_path, X=X, y=y)
    LOGGER.info("Saved %s feature cache: %s", split_name, cache_path)
    return X, y


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Calculate validation metrics with pneumonia recall as priority."""
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "pneumonia_recall": float(
            recall_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "pneumonia_f1": float(
            f1_score(y, predictions, pos_label=1, zero_division=0)
        ),
    }


def main() -> None:
    """Extract features, train baseline models, and save validation metrics."""
    set_seed(RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_features("train")
    X_val, y_val = load_features("validation")

    models = {
        "SVM": (
            make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="linear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            SVM_MODEL_PATH,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            RF_MODEL_PATH,
        ),
    }

    results: dict[str, Any] = {
        "validation_samples": int(len(y_val)),
        "feature_count": int(X_train.shape[1]),
        "models": {},
    }

    for name, (model, model_path) in models.items():
        LOGGER.info("Training %s", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        joblib.dump(model, model_path)
        results["models"][name] = metrics
        LOGGER.info("%s validation metrics: %s", name, metrics)
        LOGGER.info("Saved %s model to %s", name, model_path)

    save_json(results, BASELINE_METRICS_PATH)
    LOGGER.info("Baseline metrics saved to %s", BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()
