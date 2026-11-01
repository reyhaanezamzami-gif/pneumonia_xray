import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

metrics_dir = Path("outputs/metrics")
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

baseline = json.loads((metrics_dir / "baseline_test_metrics.json").read_text(encoding="utf-8"))
augmented = json.loads((metrics_dir / "augmented_test_metrics.json").read_text(encoding="utf-8"))
nih = json.loads((metrics_dir / "nih_test_metrics.json").read_text(encoding="utf-8-sig"))

models = ["SVM", "Random Forest", "CNN Augmented", "CNN + NIH"]

values = {
    "Accuracy": [
        baseline["models"]["SVM"]["accuracy"],
        baseline["models"]["RandomForest"]["accuracy"],
        augmented["accuracy"],
        nih["accuracy"]
    ],
    "F1-score": [
        baseline["models"]["SVM"]["f1_score"],
        baseline["models"]["RandomForest"]["f1_score"],
        augmented["f1_score"],
        nih["f1_score"]
    ],
    "Balanced Accuracy": [
        baseline["models"]["SVM"]["balanced_accuracy"],
        baseline["models"]["RandomForest"]["balanced_accuracy"],
        augmented["balanced_accuracy"],
        nih["balanced_accuracy"]
    ]
}

x = np.arange(len(models))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 6.5))
for i, (metric, scores) in enumerate(values.items()):
    bars = ax.bar(x + (i - 1) * width, np.array(scores) * 100, width, label=metric)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

ax.set_title("Test Performance Comparison of Four Models", fontsize=15, pad=18)
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 112)
ax.legend(ncol=3, loc="upper center")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()
path = output_dir / "model_comparison.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Chart saved: {path.resolve()}")
print(f"File size: {path.stat().st_size:,} bytes")
