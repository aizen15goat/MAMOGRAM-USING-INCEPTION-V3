# MAMOGRAM-USING-INCEPTION-V3
# Mammogram Classification using InceptionV3

This repository contains a PyTorch-based deep learning project for **classifying mammogram images** into **Benign** and **Malignant** categories. The model uses **InceptionV3** with a custom dense head and is trained on the **INbreast dataset**.

## Features

- Preprocessing and augmentation of mammogram images.
- Balanced dataset creation for improved accuracy.
- Custom InceptionV3 architecture with added dense layers, BatchNorm, and Dropout.
- Early stopping and learning rate scheduler to prevent overfitting.
- Evaluation metrics include:
  - Accuracy, F1-score, Sensitivity
  - Confusion Matrix
  - ROC curve and AUC
  - Grad-CAM visualizations for explainability

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- scikit-learn
- matplotlib
- pydicom (for reading DICOM images)
- PIL (Python Imaging Library)

Results

Test Accuracy: ~96-97%

F1 Score: ~0.97

ROC AUC: ~0.996

Confusion matrix and sensitivity available in evaluate_model.py.


## Usage

1. Clone the repository:

```bash
git clone <repo_url>
cd <repo_name>

python train_inceptionv3.py

python evaluate_model.py

