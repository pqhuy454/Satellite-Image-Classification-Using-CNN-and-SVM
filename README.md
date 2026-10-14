# Satellite Image Classification Using CNN and SVM

A computer vision project for classifying satellite images into five land-cover categories using **Convolutional Neural Networks (CNN)** and **Support Vector Machines (SVM)**. The project compares deep learning with traditional machine learning through image preprocessing, feature extraction, model training, and test-set evaluation.

## Project Overview

- **CNN:** Learns image features automatically using convolutional layers, batch normalization, pooling, and dropout.
- **SVM:** Uses handcrafted **Local Binary Patterns (LBP)** and **Histogram of Oriented Gradients (HOG)** features with an RBF kernel.
- **Data preprocessing:** Removes unreadable and duplicate images, resizes images to 128 × 128 pixels, and splits the dataset into training, validation, and test sets.
- **Evaluation:** Compares accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

### Classification Classes

The models classify satellite images into five categories:

`forest` · `lake` · `residential` · `river` · `road`

## Results

Both models were evaluated on the same **3,751 test images**.

| Metric | CNN | SVM |
|---|---:|---:|
| Accuracy | **93.15%** | 89.15% |
| Macro Precision | 93.23% | 89.66% |
| Macro Recall | **93.46%** | 88.74% |
| Macro F1-score | **93.27%** | 89.16% |
| Weighted F1-score | 93.12% | 89.13% |

CNN correctly classified **3,494 out of 3,751 images** and achieved an accuracy improvement of **4.00 percentage points** over SVM.

### Per-Class F1-score

| Class | CNN | SVM |
|---|---:|---:|
| Forest | **96.09%** | 95.34% |
| Lake | **94.72%** | 89.96% |
| Residential | **91.37%** | 86.38% |
| River | **92.22%** | 87.31% |
| Road | **91.94%** | 86.82% |

CNN achieved a higher F1-score across all five classes, with the largest difference for `residential`.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** Scikit-learn, SVM
- **Image Processing:** OpenCV, scikit-image
- **Data Processing:** NumPy, Pandas
- **Model Utilities:** Joblib

## Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1OnYOorFvlhxV29_s_RsIFRIYwJSe_I6N?usp=drive_link) and place the original images under `data/raw/` using the following structure:

```text
data/raw/
├── forest/
├── lake/
├── residential/
├── river/
└── road/
```

Each class folder should contain its corresponding images.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/pqhuy454/Satellite-Image-Classification-Using-CNN-and-SVM.git
cd Satellite-Image-Classification-Using-CNN-and-SVM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess the dataset

```bash
python src/clean_dataset.py
```

This step removes unreadable and duplicate images, resizes the remaining images, and creates training, validation, and test splits using a 70:15:15 ratio.

### 4. Train and evaluate CNN

```bash
python src/cnn_train.py
python src/cnn_predict.py
```

### 5. Train and evaluate SVM

```bash
python src/svm_prepare.py
python src/svm_train.py
python src/svm_predict.py
```

**Note:** Run the preprocessing step before training either model. CNN prediction requires the trained CNN model, while SVM prediction requires both the trained SVM model and the saved feature scaler.

## Key Takeaways

- Implemented end-to-end image classification pipelines for both CNN and SVM.
- Applied image preprocessing, data splitting, handcrafted feature extraction, and model evaluation.
- Compared deep learning and traditional machine learning using consistent test data and multiple classification metrics.
- Achieved **93.15% test accuracy with CNN**, outperforming the SVM baseline.