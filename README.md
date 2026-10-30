# Pneumonia Detection from Chest X-Ray Images

**A Machine Learning and Deep Learning Project for Medical Image Classification**

## Project Overview

This project explores the classification of chest X-ray images into two categories:

- **NORMAL** — No pneumonia detected.
- **PNEUMONIA** — Pneumonia detected.

The main objective is to compare traditional machine learning approaches with deep learning using transfer learning.

The project also explores model interpretability through Grad-CAM visualizations.

## Models and Methods

The following approaches are implemented and evaluated:

1. **HOG and LBP** — Hand-crafted image feature extraction.
2. **Support Vector Machine (SVM)** — Traditional classification baseline.
3. **Random Forest** — Traditional classification baseline.
4. **MobileNetV2** — Transfer learning for image classification.
5. **Grad-CAM** — Visualization of image regions influencing model predictions.

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)  
**Source:** Kaggle — `paultimothymooney/chest-xray-pneumonia`

The dataset contains chest X-ray images organized into NORMAL and PNEUMONIA classes.

The original validation folder is not used. Instead, a stratified validation split is created from the training data. The test set remains separate for final evaluation.

## Dataset Statistics

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,140 | 3,293 | 4,433 |
| Validation | 201 | 582 | 783 |
| Test | 234 | 390 | 624 |

## Baseline Results

Initial experiments with hand-crafted image features produced the following results:

| Model | Accuracy | Pneumonia Recall | F1-score |
|---|---:|---:|---:|
| SVM | 98.21% | 99.14% | 98.80% |
| Random Forest | 96.42% | 98.63% | 97.62% |

**Primary evaluation metric:** Recall for the PNEUMONIA class.

MobileNetV2 results will be added after the final evaluation.

## Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Particular attention is given to pneumonia recall because false negatives are important in medical image classification.

## Technology Stack

- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Project Structure

```text
pneumonia_xray/
├── data/
├── models/
├── outputs/
│   ├── figures/
│   ├── gradcam/
│   └── metrics/
├── src/
├── app.py
├── checkpoint.md
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/reyhaanezamzami-gif/pneumonia_xray.git
cd pneumonia_xray

python -m venv .venv
```

Activate the virtual environment on Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

**Dataset and pretrained model weights are not included in the repository.** They must be obtained separately and placed in the expected project directories.

## Reproducibility

- Random seed: `42`
- Input image size: `224 × 224`
- Stratified validation split
- Independent test set for final evaluation

## Project Status

**In Progress**

Traditional machine learning baselines have been evaluated. MobileNetV2 transfer learning and model interpretation are part of the ongoing development.

## Disclaimer

This project is intended for educational and research purposes only.

It is not a medical diagnostic tool and must not be used as a substitute for professional medical evaluation.
