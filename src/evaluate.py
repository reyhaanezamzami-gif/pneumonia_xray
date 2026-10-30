
"""Evaluate the trained MobileNetV2 model on the held-out test set."""

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from src.config import (
    CNN_MODEL_PATH,
    FINAL_METRICS_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    ensure_directories,
)
from src.cnn_data import create_dataset
from src.utils import get_logger

LOGGER = get_logger(__name__)


def evaluate_model() -> dict:
    """Evaluate MobileNetV2 on the independent test dataset."""

    ensure_directories()

    test_csv = METRICS_DIR.parent.parent / "data" / "splits" / "test.csv"

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {CNN_MODEL_PATH}")

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    LOGGER.info("Loading best MobileNetV2 model...")
    model = tf.keras.models.load_model(str(CNN_MODEL_PATH))

    LOGGER.info("Loading independent test dataset...")
    test_dataset = create_dataset(str(test_csv), shuffle=False)

    y_true = np.concatenate(
        [labels.numpy().astype(int) for _, labels in test_dataset]
    )

    probabilities = model.predict(test_dataset, verbose=1).ravel()
    y_pred = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "MobileNetV2",
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": 0.5,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        ),
    }

    FINAL_METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("MobileNetV2 - Test Confusion Matrix")
    fig.tight_layout()

    figure_path = FIGURES_DIR / "mobilenetv2_test_confusion_matrix.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    LOGGER.info("Test evaluation completed.")
    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("Precision: %.4f", metrics["precision"])
    LOGGER.info("Pneumonia Recall: %.4f", metrics["recall"])
    LOGGER.info("F1-score: %.4f", metrics["f1_score"])
    LOGGER.info("Metrics saved to %s", FINAL_METRICS_PATH)
    LOGGER.info("Confusion matrix saved to %s", figure_path)

    return metrics


if __name__ == "__main__":
    evaluate_model()
