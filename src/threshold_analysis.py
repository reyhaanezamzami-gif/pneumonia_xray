import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import MODELS_DIR, SPLITS_DIR, METRICS_DIR, FIGURES_DIR
from src.cnn_data import create_dataset

MODEL_PATH = MODELS_DIR / "mobilenetv2_augmented.keras"
OUTPUT_PATH = METRICS_DIR / "threshold_analysis.json"

def main():
    model = tf.keras.models.load_model(str(MODEL_PATH))

    dataset = create_dataset(
        str(SPLITS_DIR / "validation.csv"),
        shuffle=False,
        augment=False,
    )

    y_true = np.concatenate([
        labels.numpy().astype(int)
        for _, labels in dataset
    ])

    probabilities = model.predict(dataset, verbose=1).ravel()

    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    youden_scores = tpr - fpr

    best_index = int(np.argmax(youden_scores))
    best_threshold = float(roc_thresholds[best_index])

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, best_threshold]
    results = []

    for threshold in sorted(set(thresholds)):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, predictions)
            ),
            "precision": float(
                precision_score(y_true, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, predictions, zero_division=0)
            ),
            "specificity": float(tn / (tn + fp)),
            "f1_score": float(
                f1_score(y_true, predictions, zero_division=0)
            ),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        })

    output = {
        "dataset": "validation",
        "samples": int(len(y_true)),
        "selection_method": "Youden's J statistic",
        "recommended_threshold": best_threshold,
        "youden_j": float(youden_scores[best_index]),
        "threshold_results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(
        fpr[best_index],
        tpr[best_index],
        label=f"Selected threshold = {best_threshold:.3f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MobileNetV2 - Validation ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_roc_curve.png", dpi=300)
    plt.close()

    print(json.dumps(output, indent=4))
    print(f"\nSaved results: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
