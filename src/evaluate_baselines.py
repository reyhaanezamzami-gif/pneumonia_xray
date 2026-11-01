import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from src.baseline_models import load_features
from src.config import SVM_MODEL_PATH, RF_MODEL_PATH, METRICS_DIR

X, y = load_features("test")

models = {
    "SVM": joblib.load(SVM_MODEL_PATH),
    "RandomForest": joblib.load(RF_MODEL_PATH),
}

results = {}

for name, model in models.items():
    predictions = model.predict(X)

    if name == "SVM":
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results[name] = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions)),
        "recall": float(recall_score(y, predictions)),
        "specificity": float(tn / (tn + fp)),
        "f1_score": float(f1_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    print(f"\n{name}:")
    print(json.dumps(results[name], indent=4))

output_path = METRICS_DIR / "baseline_test_metrics.json"
output_path.write_text(
    json.dumps({"test_samples": len(y), "models": results}, indent=4),
    encoding="utf-8"
)

print(f"\nResults saved to: {output_path}")
