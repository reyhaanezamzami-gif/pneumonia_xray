# Automated Pneumonia Detection from Chest X-ray Images Using MobileNetV2

## Abstract

This project investigates automated pneumonia classification from chest X-ray images using conventional machine-learning methods and a transfer-learning-based convolutional neural network. The primary model is based on MobileNetV2, trained using the Kaggle Chest X-Ray Images (Pneumonia) dataset. A supplementary subset of the NIH ChestX-ray14 dataset was incorporated into the training data to investigate whether additional radiographic examples could improve generalization.

The final MobileNetV2 model, trained with the supplementary NIH data, achieved 86.86% accuracy, 83.08% balanced accuracy, 83.62% precision, 98.21% sensitivity, and 67.95% specificity on the held-out test set at a decision threshold of 0.50. Its ROC-AUC was 96.01%. These results indicate strong pneumonia detection sensitivity, although false-positive predictions remain an important limitation.

The findings are experimental and do not establish clinical readiness. Dataset label uncertainty, possible distribution differences, and limitations in patient-level separation must be considered when interpreting the results.

## 1. Introduction

Pneumonia is a respiratory infection that can produce radiographic abnormalities visible in chest X-ray images. The interpretation of these images is an important part of clinical assessment. Deep learning offers a way to investigate automated image classification and potentially support the analysis of radiographic data.

The objective of this project is to develop and evaluate a binary image-classification system that distinguishes between:

- NORMAL: images labelled as normal.
- PNEUMONIA: images labelled as showing pneumonia.

The project follows a comparative approach. Handcrafted image features combined with conventional classifiers are used as baselines, while MobileNetV2 is investigated as the principal deep-learning architecture. An additional experiment incorporates a small, selected subset of NIH ChestX-ray14 images into the training data.

This system is intended as a research prototype rather than a diagnostic medical device.

## 2. Datasets

### 2.1 Kaggle Chest X-Ray Images (Pneumonia)

The primary dataset was obtained from the Kaggle dataset:

`paultimothymooney/chest-xray-pneumonia`

The project uses the following CSV-based data partitions:

| Partition | Number of images | NORMAL | PNEUMONIA |
|---|---:|---:|---:|
| Training | 4,469 | 1,147 | 3,322 |
| Validation | 747 | 194 | 553 |
| Test | 624 | 234 | 390 |
| Total | 5,840 | 1,575 | 4,265 |

The validation partition was reconstructed from the development data because the original validation directory contained too few images for reliable model selection. The test partition was retained separately and was not included in model training.

A stratified group-splitting procedure using seven folds and a fixed random seed was applied. Group identifiers were derived from filename-based proxies. Consequently, the absence of group overlap is not equivalent to verified patient-level separation: the dataset does not provide verified patient identifiers for this procedure.

### 2.2 NIH ChestX-ray14 Supplementary Data

A supplementary subset was selected from the NIH ChestX-ray14 dataset. The NIH metadata contains 112,120 image records belonging to 30,805 unique patients.

The project selected images from the first downloaded archive, `images_001.tar.gz`. The selected balanced subset contains:

| Label | Images |
|---|---:|
| Pneumonia | 65 |
| No Finding | 48 |
| Total | 113 |

The 65 pneumonia-labelled images represent 48 unique patients. The 48 negative examples were selected from different patients, with no patient-ID overlap between the selected positive and negative groups.

The NIH labels require cautious interpretation. Pneumonia labels are derived from radiology reports and may be noisy. In addition, the `No Finding` label is not equivalent to a clinically confirmed healthy diagnosis. The supplementary dataset is small and represents only a limited portion of the full NIH collection.

### 2.3 Combined Training Data

The supplementary NIH subset was added only to the original training partition.

| Source | Images |
|---|---:|
| Original Kaggle training data | 4,469 |
| Selected NIH subset | 113 |
| Combined training data | 4,582 |

The combined training CSV contains 3,387 pneumonia images and 1,195 normal images. No duplicate paths or missing image paths were detected in the combined CSV.

The original Kaggle validation and test partitions were retained unchanged.

## 3. Data Preprocessing

The image pipeline performs the following operations:

1. Images are decoded and converted to three channels.
2. Images are resized to 224 × 224 pixels.
3. Pixel values are represented as float32 values in the range 0–255 before model preprocessing.
4. The MobileNetV2 preprocessing function is applied within the model.

The decoding pipeline supports grayscale and RGBA input images by converting them into three-channel images.

### 3.1 Data Augmentation

Mild augmentation is applied to training images only. The implemented transformations include:

- Random rotation of approximately 1.4% of a full turn.
- Translation of up to approximately 5%.
- Zoom variation of up to approximately 8%.
- Contrast adjustment of approximately 10%.

The augmented pixel values are clipped to the valid 0–255 range. Validation and test images are not augmented.

## 4. Baseline Machine-Learning Models

Before training the convolutional neural network, two conventional classifiers were evaluated.

Handcrafted image features included:

- Histogram of Oriented Gradients (HOG).
- Local Binary Patterns (LBP) histograms.
- Basic image-intensity statistics.

These features were used to train:

1. Support Vector Machine (SVM).
2. Random Forest (RF).

The baseline models provide a reference point for evaluating the performance of the deep-learning approach.

## 5. Deep-Learning Architecture

### 5.1 MobileNetV2

MobileNetV2 was selected as the convolutional neural network architecture. It is designed to provide computationally efficient image feature extraction through depthwise separable convolutions and inverted residual blocks.

The implemented model uses:

- MobileNetV2 with `include_top=False`.
- Input resolution of 224 × 224 × 3.
- Locally stored pretrained weights.
- A frozen convolutional backbone.
- Global Average Pooling.
- Dropout with a rate of 0.30.
- A single sigmoid output neuron for binary classification.

The backbone weights were loaded from the local file:

`models/mobilenet_v2_weights.h5`

The backbone was frozen during training. Consequently, the trainable classification head contains 1,281 parameters.

### 5.2 Training Configuration

The principal training configuration includes:

| Parameter | Setting |
|---|---|
| Architecture | MobileNetV2 |
| Input size | 224 × 224 × 3 |
| Batch size | 16 |
| Maximum epochs | 20 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Loss function | Binary Cross-Entropy |
| Dropout | 0.30 |
| Output activation | Sigmoid |

The training process uses EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and TerminateOnNaN callbacks.

The final NIH-augmented model was saved as:

`models/mobilenetv2_nih.keras`

The corresponding training history was saved as:

`outputs/metrics/cnn_nih_history.json`

Training completed 20 epochs. At the final epoch, the recorded validation accuracy was approximately 96.12%, and the validation ROC-AUC was approximately 98.78%.

## 6. Evaluation Methodology

The final models were evaluated using the same 624-image Kaggle test partition.

The following metrics were calculated:

- Accuracy.
- Balanced accuracy.
- Precision.
- Recall (sensitivity).
- Specificity.
- F1-score.
- ROC-AUC.
- PR-AUC.

For the main comparison, the CNN models use a decision threshold of 0.50. The confusion matrix uses the following convention:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| Actual NORMAL | True Negative (TN) | False Positive (FP) |
| Actual PNEUMONIA | False Negative (FN) | True Positive (TP) |

Sensitivity measures the proportion of pneumonia cases correctly identified. Specificity measures the proportion of normal cases correctly identified.

## 7. Experimental Results

### 7.1 Main Test-Set Comparison

The following results were obtained on the same 624-image test partition.

| Metric (%) | SVM | Random Forest | CNN Augmented | CNN + NIH |
|---|---:|---:|---:|---:|
| Accuracy | 75.00 | 71.96 | 85.26 | 86.86 |
| Balanced Accuracy | 66.75 | 62.95 | 80.85 | 83.08 |
| Precision | 71.51 | 69.30 | 81.70 | 83.62 |
| Recall / Sensitivity | 99.74 | 98.97 | 98.46 | 98.21 |
| Specificity | 33.76 | 26.92 | 63.25 | 67.95 |
| F1-score | 83.30 | 81.52 | 89.30 | 90.33 |
| ROC-AUC | 92.96 | 93.39 | 95.62 | 96.01 |
| PR-AUC | 95.47 | 95.78 | 97.09 | 97.23 |

The CNN + NIH model achieved the highest accuracy, balanced accuracy, precision, F1-score, ROC-AUC, and PR-AUC among the compared models.

The SVM achieved the highest sensitivity, but its specificity was substantially lower. This indicates that it identified nearly all pneumonia cases while incorrectly classifying many normal images as pneumonia.

### 7.2 Confusion Matrix of the Final Model

At a threshold of 0.50, the final MobileNetV2 + NIH model produced:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| Actual NORMAL | 159 | 75 |
| Actual PNEUMONIA | 7 | 383 |

Therefore:

- True negatives: 159.
- False positives: 75.
- False negatives: 7.
- True positives: 383.

The model detected 383 of the 390 pneumonia images. However, 75 of the 234 normal images were incorrectly classified as pneumonia.

This demonstrates the model's high sensitivity, alongside a remaining false-positive problem.

### 7.3 Effect of Supplementary NIH Data

Compared with the previous augmented CNN model, the NIH-augmented model achieved:

| Metric | Previous CNN | CNN + NIH | Change |
|---|---:|---:|---:|
| Accuracy | 85.26% | 86.86% | +1.60 pp |
| Balanced Accuracy | 80.85% | 83.08% | +2.22 pp |
| Precision | 81.70% | 83.62% | +1.92 pp |
| Recall | 98.46% | 98.21% | -0.26 pp |
| Specificity | 63.25% | 67.95% | +4.70 pp |
| F1-score | 89.30% | 90.33% | +1.03 pp |
| ROC-AUC | 95.62% | 96.01% | +0.39 pp |
| PR-AUC | 97.09% | 97.23% | +0.14 pp |

The number of false positives decreased from 86 to 75, while false negatives increased from 6 to 7.

These results are consistent with a modest improvement in the balance between sensitivity and specificity. Nevertheless, the NIH subset contains only 113 images, so the experiment does not establish that the supplementary data caused the improvement. Further experiments with larger and independently verified datasets are required.

## 8. Decision Threshold Analysis

The default classification threshold was 0.50. A threshold search was also performed using the validation partition.

Among the tested thresholds, 0.45 achieved the highest validation F1-score and accuracy:

| Metric | Threshold 0.45 | Threshold 0.50 |
|---|---:|---:|
| Validation accuracy | 96.65% | 96.12% |
| Validation precision | 98.18% | 98.34% |
| Validation recall | 97.29% | 96.38% |
| Validation specificity | 94.85% | 95.36% |
| Validation F1-score | 97.73% | 97.35% |

The 0.45 threshold was selected using validation data, not test data.

For transparency, both thresholds were subsequently evaluated on the test set:

| Metric | Test at 0.45 | Test at 0.50 |
|---|---:|---:|
| Accuracy | 85.42% | 86.86% |
| Balanced accuracy | 81.07% | 83.08% |
| Precision | 81.88% | 83.62% |
| Recall | 98.46% | 98.21% |
| Specificity | 63.68% | 67.95% |
| F1-score | 89.41% | 90.33% |
| ROC-AUC | 96.01% | 96.01% |
| PR-AUC | 97.23% | 97.23% |

The threshold of 0.45 increased test sensitivity slightly but also increased false positives. The threshold of 0.50 produced better test accuracy, specificity, precision, and F1-score.

The main comparison therefore reports the 0.50 results. The 0.45 test results are included as a transparent secondary analysis, not as a basis for further threshold tuning. Because the test set has now been used to inspect both thresholds, future model or threshold selection should use a new independent test set.

## 9. Error Analysis

The final model generated 624 test predictions:

- Correct predictions: 542.
- False positives: 75.
- False negatives: 7.

The false-negative probabilities were below 0.50, with a median of approximately 0.3965. Some false-positive predictions had high confidence, including probabilities close to 1.0.

These errors may reflect difficult radiographic patterns, label uncertainty, image acquisition differences, or dataset distribution differences. Their causes cannot be established from prediction probabilities alone.

The project includes per-image predictions in:

`outputs/metrics/nih_test_predictions.csv`

Visual inspection and expert review would be necessary to determine whether individual errors correspond to ambiguous images or annotation problems.

## 10. Explainability

Grad-CAM was implemented to inspect image regions that contribute to the CNN's predictions. The final convolutional layer used for this analysis is `out_relu` within the MobileNetV2 backbone.

Grad-CAM heatmaps can help visualize which regions influence a prediction. However, highlighted regions do not prove that the model has identified pneumonia-related pathology correctly. These visualizations should be interpreted as exploratory explanations rather than clinical evidence.

## 11. Limitations

Several limitations affect the interpretation of the results.

1. **Dataset size and representativeness:** The project uses a limited collection of chest X-ray images and may not represent the full diversity of clinical populations and imaging environments.

2. **Label uncertainty:** The NIH pneumonia labels are report-derived, and `No Finding` does not guarantee the absence of disease.

3. **Patient-level separation:** Filename-derived group proxies were used for the Kaggle split. They are not verified patient identifiers, so complete patient-level independence cannot be guaranteed.

4. **Cross-dataset differences:** The Kaggle and NIH datasets may differ in image acquisition, patient populations, annotations, and preprocessing characteristics.

5. **Small supplementary subset:** Only 113 NIH images were added to training. The observed improvement requires confirmation through larger experiments.

6. **Test-set threshold inspection:** Both 0.45 and 0.50 were evaluated on the test set. The main result uses 0.50, but future model selection should use a new untouched test set.

7. **Clinical applicability:** The model has not undergone prospective clinical validation and must not be used as a substitute for radiologist assessment.

8. **Frozen backbone:** The MobileNetV2 feature extractor was frozen. Fine-tuning could be investigated in future work, but it would require careful validation and leakage control.

## 12. Conclusion

This project developed a binary pneumonia classification pipeline using handcrafted image features, conventional machine-learning classifiers, and MobileNetV2.

The final MobileNetV2 model trained with the supplementary NIH subset achieved 86.86% test accuracy, 98.21% sensitivity, 67.95% specificity, and a 96.01% ROC-AUC at a threshold of 0.50.

Compared with the previous augmented CNN model, the NIH-augmented model showed improvements in accuracy, balanced accuracy, precision, specificity, F1-score, and AUC metrics. It also reduced the number of false-positive predictions from 86 to 75, while increasing false negatives from 6 to 7.

The results suggest that supplementary radiographic data may be useful for improving model performance. However, the small size and uncertain labels of the NIH subset prevent a strong causal conclusion.

Overall, MobileNetV2 provided a better balance of classification performance than the handcrafted-feature baselines in this experiment. Further work should focus on larger and better-curated datasets, verified patient-level separation, external validation, calibration, threshold selection on independent validation data, and expert review of model explanations.

The system should be regarded as an experimental research prototype and not as a clinically validated diagnostic tool.

## 13. Figures and Visual Results

The following figures are provided in the project output directory:

- **Figure 1. Representative chest X-ray images:** `outputs/figures/sample_images.png`. This figure presents example images from the classification dataset.
- **Figure 2. Validation ROC curve:** `outputs/figures/validation_roc_curve.png`. This figure illustrates the discrimination performance on the validation partition.
- **Figure 3. Final model confusion matrix:** `outputs/figures/mobilenetv2_nih_test_confusion_matrix.png`. This figure summarizes the test predictions of the NIH-augmented MobileNetV2 model at a threshold of 0.50.
- **Figure 4. Previous CNN confusion matrix:** `outputs/figures/mobilenetv2_augmented_test_confusion_matrix.png`. This figure shows the test predictions of the previous augmented CNN model.
- **Figure 5. Decision threshold comparison:** `outputs/figures/nih_threshold_comparison.png`. This figure compares the reported performance at thresholds of 0.45 and 0.50.

- **Figure 6. Four-model performance comparison:** `outputs/figures/model_comparison.png`. This figure compares Accuracy, F1-score, and Balanced Accuracy across SVM, Random Forest, the augmented CNN, and CNN + NIH on the same test set.

The confusion matrices and threshold comparison provide complementary views of classification errors and the sensitivity-specificity trade-off. The ROC curve is based on validation data and should not be interpreted as a test-set ROC curve.

## References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.e9.
2. Wang, X., et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
3. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
4. Kaggle. Chest X-Ray Images (Pneumonia). Dataset by Paul Timothy Mooney. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
5. National Institutes of Health Clinical Center. ChestX-ray14 dataset and associated documentation. https://nihcc.app.box.com/v/ChestXray-NIHCC
