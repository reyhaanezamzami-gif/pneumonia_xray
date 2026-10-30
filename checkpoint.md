# Project Checkpoint — Pneumonia X-ray Classification

## Completed Phases

### Phase 1 — Project Setup
- Python 3.10.11 and virtual environment configured.
- Required packages installed.
- Project directory structure created.
- Configuration and utility modules implemented.

### Phase 2 — Dataset Inspection
- Dataset: `paultimothymooney/chest-xray-pneumonia`
- Total images: 5,856
- Unreadable images: 0
- Dataset structure and class counts inspected.
- Sample images saved.

### Phase 3 — Preprocessing and Data Splitting
- Images resized to 224×224.
- RGB preprocessing implemented.
- Original validation directory excluded.
- Stratified 15% validation split created from training data.

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,140 | 3,293 | 4,433 |
| Validation | 201 | 582 | 783 |
| Test | 234 | 390 | 624 |

### Phase 4 — Feature Extraction and Baseline Models
- HOG features extracted.
- LBP features extracted.
- Five image intensity statistics extracted.
- Combined feature vector: 6,100 features per image.
- Train and validation feature caches saved.
- SVM and Random Forest trained.
- Both models evaluated on validation data.
- Models and metrics saved successfully.

#### Validation Results

| Metric | SVM | Random Forest |
|---|---:|---:|
| Accuracy | 98.21% | 96.42% |
| Pneumonia Recall | 99.14% | 98.63% |
| Pneumonia F1 | 98.80% | 97.62% |

Best baseline model: SVM

Important output files:
- `models/baseline_svm.joblib`
- `models/baseline_rf.joblib`
- `outputs/metrics/baseline_metrics.json`
- `data/splits/train_features.npz`
- `data/splits/validation_features.npz`

## Current Status

Phases 1–4: COMPLETED

Next phase: Phase 5 — MobileNetV2 Transfer Learning

## Project Rules
- Random seed: 42
- Image size: 224×224
- Primary metric: Pneumonia Recall
- Original validation directory must not be used.
- Test data must remain untouched until final evaluation.
- Use class weights during CNN training.
- Keep source files at or below 200 lines.
