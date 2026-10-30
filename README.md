# Pneumonia Detection from Chest X-Ray Images



## Project Overview



This project develops and evaluates machine learning and deep learning

models to classify chest X-ray images into two categories:



- `NORMAL`: no pneumonia detected

- `PNEUMONIA`: pneumonia detected



The project compares hand-crafted image features with a

MobileNetV2 transfer learning model.



## Dataset



Dataset: Chest X-Ray Images (Pneumonia)



Source: Kaggle â€” `paultimothymooney/chest-xray-pneumonia`



Expected directory structure:



&#x20;   data/chest\_xray/

&#x20;       train/

&#x20;           NORMAL/

&#x20;           PNEUMONIA/

&#x20;       val/

&#x20;           NORMAL/

&#x20;           PNEUMONIA/

&#x20;       test/

&#x20;           NORMAL/

&#x20;           PNEUMONIA/



The original validation folder will not be used for model validation.

A stratified validation split will be created from the training data.

The test set will remain separate for final evaluation.



## Models



1. HOG and LBP image features with traditional machine learning.

2. Support Vector Machine (SVM) baseline.

3. Random Forest baseline.

4. MobileNetV2 transfer learning model.

5. Manual Grad-CAM visualization for model interpretation.



## Evaluation



The project reports relevant classification metrics, including:



- Accuracy

- Precision

- Recall

- F1-score

- Confusion matrix



Recall for the `PNEUMONIA` class is the primary evaluation metric.



## Environment Setup



Python 3.10 is recommended.



Create and activate a virtual environment, then install dependencies:



&#x20;   python -m pip install -r requirements.txt



## Project Structure



- `data/`: dataset and data splits

- `src/`: configuration, preprocessing, features, models, and evaluation

- `models/`: saved trained models

- `outputs/`: figures, metrics, and Grad-CAM results

- `app.py`: Streamlit application

- `checkpoint.md`: project progress tracking



## Reproducibility



The project uses random seed `42` for reproducibility where supported.



## Disclaimer



This project is intended for educational and research purposes only.

It is not a medical diagnostic tool and must not be used as a substitute

for professional medical evaluation.


