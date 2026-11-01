
"""Evaluate the augmented MobileNetV2 model on the held-out test set."""

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    ensure_directories,
)
from src.cnn_data import create_dataset
from src.utils import get_logger

LOGGER = get_logger(__name__)

AUGMENTED_MODEL_PATH = MODELS_DIR / "mobilenetv2_nih.keras"
AUGMENTED_METRICS_PATH = METRICS_DIR / "nih_test_metrics.json"


def evaluate_model() -> dict:
    """Evaluate the augmented model on the independent test dataset."""

    ensure_directories()

    test_csv = METRICS_DIR.parent.parent / "data" / "splits" / "test.csv"

    if not AUGMENTED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {AUGMENTED_MODEL_PATH}"
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {test_csv}"
        )

    LOGGER.info("Loading augmented MobileNetV2 model...")

    model = tf.keras.models.load_model(
        str(AUGMENTED_MODEL_PATH)
    )

    LOGGER.info("Loading independent test dataset...")

    test_dataset = create_dataset(
        str(test_csv),
        shuffle=False,
        augment=False,
    )

    y_true = np.concatenate(
        [
            labels.numpy().astype(int)
            for _, labels in test_dataset
        ]
    )

    probabilities = model.predict(
        test_dataset,
        verbose=1,
    ).ravel()

    threshold = 0.5
    y_pred = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    metrics = {
        "model": "MobileNetV2 with Data Augmentation",
        "test_samples": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        ),
    }

    AUGMENTED_METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    ax.set_title(
        "MobileNetV2 with Augmentation - Test Confusion Matrix"
    )

    fig.tight_layout()

    figure_path = (
        FIGURES_DIR / "mobilenetv2_nih_test_confusion_matrix.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
    )

    plt.close(fig)

    LOGGER.info("Augmented model test evaluation completed.")
    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("Balanced Accuracy: %.4f", metrics["balanced_accuracy"])
    LOGGER.info("Precision: %.4f", metrics["precision"])
    LOGGER.info("Sensitivity: %.4f", metrics["sensitivity"])
    LOGGER.info("Specificity: %.4f", metrics["specificity"])
    LOGGER.info("F1-score: %.4f", metrics["f1_score"])
    LOGGER.info("ROC-AUC: %.4f", metrics["roc_auc"])
    LOGGER.info("PR-AUC: %.4f", metrics["pr_auc"])
    LOGGER.info("Metrics saved to %s", AUGMENTED_METRICS_PATH)
    LOGGER.info("Confusion matrix saved to %s", figure_path)

    return metrics


if __name__ == "__main__":
    evaluate_model()
