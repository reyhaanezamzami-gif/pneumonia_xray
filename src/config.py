"""Central configuration for the pneumonia X-ray project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "chest_xray"
SPLITS_DIR = DATA_DIR / "splits"

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
METRICS_DIR = OUTPUTS_DIR / "metrics"

SEED = 42
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

VALIDATION_SIZE = 0.15
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}

RANDOM_STATE = SEED
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DATASET_NAME = "chest-xray-pneumonia"
DATASET_SOURCE = "paultimothymooney/chest-xray-pneumonia"

SVM_MODEL_PATH = MODELS_DIR / "baseline_svm.joblib"
RF_MODEL_PATH = MODELS_DIR / "baseline_rf.joblib"
CNN_MODEL_PATH = MODELS_DIR / "mobilenetv2_best.keras"
MOBILENET_WEIGHTS_PATH = MODELS_DIR / "mobilenet_v2_weights.h5"

# Classification thresholds
CNN_NIH_MODEL_PATH = MODELS_DIR / "mobilenetv2_nih.keras"
CNN_NIH_VALIDATION_THRESHOLD = 0.45
CNN_NIH_DEFAULT_EVALUATION_THRESHOLD = 0.50


DATASET_SUMMARY_PATH = METRICS_DIR / "dataset_summary.json"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
CNN_HISTORY_PATH = METRICS_DIR / "cnn_history.json"
FINAL_METRICS_PATH = METRICS_DIR / "final_metrics.json"

RESULTS_PATH = OUTPUTS_DIR / "results.md"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoint.md"
LOG_PATH = METRICS_DIR / "project.log"


def ensure_directories() -> None:
    """Create all project directories required by the pipeline."""
    directories = (
        DATA_DIR,
        RAW_DATA_DIR,
        SPLITS_DIR,
        SRC_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        GRADCAM_DIR,
        METRICS_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)