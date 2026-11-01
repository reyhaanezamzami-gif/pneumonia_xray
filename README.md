# Pneumonia Detection from Chest X-Ray Images

**A Machine Learning and Deep Learning Project for Medical Image Classification**

## 1. Project Overview

This project focuses on classifying chest X-ray images into two categories:

- **NORMAL** — No pneumonia detected.
- **PNEUMONIA** — Pneumonia detected.

The main objective is to compare traditional machine learning algorithms with deep learning approaches using transfer learning.

The project also explores model interpretability using Grad-CAM (Gradient-weighted Class Activation Mapping).

## 2. Models and Methods

The following methods are implemented and evaluated:

1. **HOG and LBP:** Hand-crafted image feature extraction.
2. **Support Vector Machine (SVM):** Traditional machine learning classifier.
3. **Random Forest:** Ensemble machine learning classifier.
4. **MobileNetV2:** Deep learning model using transfer learning.
5. **Grad-CAM:** Visualization of image regions influencing model predictions.

## 3. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)  
**Source:** Kaggle — `paultimothymooney/chest-xray-pneumonia`

The dataset contains chest X-ray images belonging to two classes: NORMAL and PNEUMONIA.

The original validation directory is not used. Instead, a stratified validation split is created from the training data.

The test set remains independent and is used only for final model evaluation.

## 4. Dataset Statistics

| Dataset Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,140 | 3,293 | 4,433 |
| Validation | 201 | 582 | 783 |
| Test | 234 | 390 | 624 |
| **Total** | **1,575** | **4,265** | **5,840** |

## 5. Baseline Results

Initial experiments using hand-crafted image features produced the following validation results:

| Model | Accuracy | Pneumonia Recall | F1-score |
|---|---:|---:|---:|
| SVM | 98.21% | 99.14% | 98.80% |
| Random Forest | 96.42% | 98.63% | 97.62% |

**Primary evaluation metric:** Recall for the PNEUMONIA class.

## 6. MobileNetV2 — Independent Test Results

MobileNetV2 was implemented using transfer learning with pretrained weights.

The convolutional base was used as a feature extractor, followed by a classification head.

The model was evaluated on the independent test set.

### Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 85.26% |
| Pneumonia Precision | 83.41% |
| Pneumonia Recall | 95.38% |
| Pneumonia F1-score | 89.00% |
| NORMAL Recall | 68.38% |

### Confusion Matrix

The following confusion matrix summarizes the model's predictions:

![MobileNetV2 Test Confusion Matrix](outputs/figures/mobilenetv2_test_confusion_matrix.png)

| Actual Class / Predicted Class | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 160 | 74 |
| PNEUMONIA | 18 | 372 |

### Results Interpretation

- 160 NORMAL images were correctly classified.
- 74 NORMAL images were incorrectly classified as PNEUMONIA.
- 18 PNEUMONIA images were incorrectly classified as NORMAL.
- 372 PNEUMONIA images were correctly classified.

The model achieved high pneumonia recall (95.38%). However, NORMAL-class recall was lower (68.38%).

This indicates that the model still has difficulty distinguishing some normal X-ray images from pneumonia cases.

## 7. Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is an interpretability technique used to visualize image regions that contribute to a model's prediction.

The following example shows the original chest X-ray image alongside its Grad-CAM heatmap overlay.

![Grad-CAM Visualization](outputs/gradcam/person1000_bacteria_2931_gradcam.png)

### Example Prediction

- **Predicted class:** PNEUMONIA
- **Pneumonia probability:** 96.01%

The heatmap provides an approximate visual explanation of the regions influencing the model's prediction.

Grad-CAM does not establish clinical validity and does not prove that the model focuses on medically relevant regions.

## 8. Evaluation Metrics

The following evaluation metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Particular attention is given to pneumonia recall because false negative predictions are important in medical image classification.

## 9. Technology Stack

The project uses the following technologies and libraries:

- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

## 10. Project Structure

```text
pneumonia_xray/
│
├── data/
│
├── models/
│
├── outputs/
│   ├── figures/
│   ├── gradcam/
│   └── metrics/
│
├── src/
│
├── app.py
├── checkpoint.md
├── requirements.txt
└── README.md
```

## 11. Installation

Clone the repository:

```bash
git clone https://github.com/reyhaanezamzami-gif/pneumonia_xray.git

cd pneumonia_xray
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```cmd
.venv\Scripts\activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

**Important:** Dataset files and pretrained model weights are not included in the repository. They must be obtained separately and placed in the expected project directories.

## 12. Reproducibility

The following settings are used to support reproducibility:

- Random seed: `42`
- Input image size: `224 × 224`
- Stratified validation split
- Independent test set for final evaluation

## 13. Project Status

**Status: In Progress**

The project currently includes:

- Traditional machine learning baselines.
- HOG and LBP feature extraction.
- SVM and Random Forest classification.
- MobileNetV2 transfer learning.
- Independent test evaluation.
- Confusion matrix visualization.
- Grad-CAM implementation and visualization.

Future improvements may focus on model generalization, NORMAL-class performance, hyperparameter optimization, and interpretability.

## 14. Disclaimer

This project is intended for educational and research purposes only.

It is not a medical diagnostic tool and must not be used as a substitute for professional medical evaluation.