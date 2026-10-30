# Project Checkpoint — Pneumonia X-ray Classification

## Completed Phases

### Phase 1 — Project Setup
- Python 3.10.11 and virtual environment configured.
- Required packages installed.
- Project directory structure created.
- Configuration and utility modules implemented.
- Git repository initialized and connected to the private GitHub repository.

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

### Phase 5 — MobileNetV2 Transfer Learning
- MobileNetV2 implemented using TensorFlow/Keras.
- ImageNet-pretrained weights loaded from a local weights file.
- Convolutional base frozen during initial training.
- Global Average Pooling, Dropout, and Sigmoid classification head added.
- CNN data pipeline implemented using `tf.data`.
- Model trained with early stopping and best-checkpoint restoration.
- Best model saved to `models/mobilenetv2_best.keras`.
- Independent test evaluation completed.

#### Training and Validation Notes
- Early stopping occurred at epoch 10.
- Best checkpoint restored from epoch 5.
- Best observed validation Pneumonia Recall: 92.96%.
- Validation Accuracy at the final logged epoch: 93.74%.

#### Independent Test Results

Test set: 624 images (234 NORMAL, 390 PNEUMONIA)

| Metric | MobileNetV2 |
|---|---:|
| Accuracy | 85.26% |
| Precision (PNEUMONIA) | 83.41% |
| Pneumonia Recall | 95.38% |
| Pneumonia F1 | 89.00% |
| Classification threshold | 0.50 |

#### Test Confusion Matrix

Rows represent actual classes; columns represent predicted classes.

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 160 | 74 |
| PNEUMONIA | 18 | 372 |

#### Class-wise Test Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 89.89% | 68.38% | 77.67% | 234 |
| PNEUMONIA | 83.41% | 95.38% | 89.00% | 390 |

#### Interpretation
- The model correctly identified 372 of 390 pneumonia images.
- It missed 18 pneumonia images (false negatives).
- It incorrectly classified 74 normal images as pneumonia (false positives).
- Pneumonia Recall is high, but NORMAL Recall is comparatively low.
- The difference between validation and test performance indicates a generalization gap that requires further investigation.

Important output files:
- `src/cnn_model.py`
- `src/cnn_data.py`
- `src/train_cnn.py`
- `src/evaluate.py`
- `models/mobilenetv2_best.keras`
- `outputs/metrics/final_metrics.json`
- `outputs/figures/mobilenetv2_test_confusion_matrix.png`

## Current Status

- Phases 1–5: COMPLETED
- Independent test evaluation: COMPLETED
- Final documentation and GitHub synchronization: PENDING

## Project Rules
- Random seed: 42
- Image size: 224×224
- Primary metric: Pneumonia Recall
- Original validation directory must not be used.
- Test data must remain untouched during training and model selection.
- Use class weights during CNN training.
- Keep source files at or below 200 lines.
- Report validation and test results separately.
- Do not interpret experimental model predictions as clinical diagnoses.

## Next Steps
1. Review the final test results and confusion matrix.
2. Update README with MobileNetV2 test results.
3. Verify output files and source changes.
4. Commit and push the completed evaluation phase to GitHub.
5. Investigate the validation-to-test performance gap in a subsequent phase.