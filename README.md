# Signature Recognition using CNN and Feature Engineering (HOG/SIFT)

[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)]()
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-yellowgreen.svg)](https://scikit-learn.org/)

A comprehensive signature recognition system that compares three different approaches: **Convolutional Neural Networks (CNN)**, **Histogram of Oriented Gradients (HOG) + SVM**, and **Scale-Invariant Feature Transform (SIFT) + SVM**. This project demonstrates the effectiveness of deep learning versus traditional computer vision techniques for signature verification and classification.

## 🎯 Project Overview

This project implements and compares three different approaches for signature recognition:

1. **CNN (Deep Learning)**: Custom convolutional neural network optimized for T4 GPU with mixed precision training
2. **HOG + SVM**: Traditional computer vision approach using Histogram of Oriented Gradients features
3. **SIFT + SVM**: Keypoint-based feature extraction using Scale-Invariant Feature Transform

The system can distinguish between **genuine** and **forged** signatures across multiple signature classes, making it suitable for authentication and verification applications.

## 🏆 Key Results

Based on experimental evaluation:

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **CNN** | 78.79% | 79.43% | 78.79% | 75.85% |
| **HOG+SVM** | **78.18%** | **79.37%** | **78.18%** | **75.76%** |
| **SIFT+SVM** | 3.33% | 0.98% | 3.33% | 1.16% |

🏅 **Best Performing Method**: HOG+SVM shows competitive performance with CNN while being computationally more efficient.

## 📋 Features

- **Multi-method comparison**: Side-by-side evaluation of three different approaches
- **GPU optimization**: Mixed precision training for efficient CNN training on T4 GPU
- **Data augmentation**: Rotation, shift, shear, and zoom transformations
- **Comprehensive evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Genuine vs Forged analysis**: Specialized analysis for signature authenticity
- **Modular design**: Easy to extend and modify for different datasets
- **Visualization tools**: HOG feature visualization, training curves, and sample displays

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended: T4 or better)
- Google Colab (recommended) or local environment

### Required Dependencies

```bash
pip install tensorflow>=2.8.0
pip install opencv-python
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install tqdm
pip install kagglehub
pip install joblib
```

### For Google Colab Users

All dependencies are pre-installed in Google Colab. Simply upload the notebook and run!

## 📁 Dataset Structure

The project expects a signature dataset with the following structure:

```
dataset/
├── sign_data/
│   ├── train/
│   │   ├── person001/
│   │   │   ├── signature1.png
│   │   │   ├── signature2.png
│   │   │   └── ...
│   │   ├── person001_forg/
│   │   │   ├── forged1.png
│   │   │   ├── forged2.png
│   │   │   └── ...
│   │   ├── person002/
│   │   └── person002_forg/
│   └── test/
│       └── (similar structure)
```

### Supported Dataset Formats

- **Image formats**: PNG, JPG, JPEG
- **Naming convention**: Forged signatures should have `_forg` suffix
- **Image size**: Automatically resized to 128x128 pixels
- **Color**: Converted to grayscale for processing

## 🚀 Usage

### Quick Start

1. **Clone the repository**:
```bash
git clone <git-url>
cd signature-recognition
```

2. **Open in Google Colab** (Recommended):
   - Upload `CNN_Signature_FeatureEngineering(HOG OR SHIFT).ipynb`
   - Mount Google Drive or upload dataset
   - Run all cells

3. **Local execution**:
```python
from signature_recognition import SignatureRecognition

# Initialize the system
sig_recognition = SignatureRecognition(
    data_path="path/to/your/dataset",
    img_size=(128, 128),
    batch_size=32
)

# Load and preprocess data
sig_recognition.load_data()

# Train and evaluate CNN
sig_recognition.build_cnn_model()
sig_recognition.train_cnn_model(epochs=30)
sig_recognition.evaluate_cnn_model()

# Train and evaluate HOG+SVM
sig_recognition.train_hog_svm()

# Train and evaluate SIFT+SVM
sig_recognition.train_sift_svm()

# Compare all methods
comparison_results = sig_recognition.compare_methods()
```

### Testing New Signatures

```python
# Test a new signature image
predictions = test_signature_recognition(
    test_image_path="path/to/test/signature.png",
    models_dir="path/to/saved/models"
)

# Visualize HOG features
visualize_hog_features("path/to/test/signature.png")
```

## 🧠 Model Architecture

### CNN Architecture

```
Input (128x128x1)
    ↓
Conv2D(32) → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → Dropout(0.5) → Dense(num_classes)
```

### Feature Engineering

- **HOG Features**: 9 orientations, 8x8 pixels per cell, 2x2 cells per block
- **SIFT Features**: Keypoint detection with SURF detector, K-means clustering (K=100)
- **SVM Classifier**: RBF kernel with optimized hyperparameters

## 📊 Performance Analysis

### Training Configuration

- **Hardware**: Google Colab T4 GPU
- **Mixed Precision**: Enabled for faster training
- **Data Augmentation**: Rotation (10°), shift (10%), shear (10%), zoom (10%)
- **Optimization**: Adam optimizer with learning rate reduction
- **Early Stopping**: Patience of 10 epochs

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for detailed class-wise analysis
- Genuine vs Forged signature performance
- Training time comparison

## 📈 Visualizations

The project generates comprehensive visualizations:

- Class distribution analysis
- Sample signature displays (genuine vs forged)
- Training curves (loss and accuracy)
- Confusion matrices
- HOG feature visualizations
- Method comparison charts
- Genuine vs forged performance analysis

## 🔧 Customization

### Hyperparameter Tuning

```python
# CNN hyperparameters
epochs = 30
batch_size = 32
img_size = (128, 128)

# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# SIFT parameters
n_features = 500
n_clusters = 100
```

### Adding New Methods

The modular design allows easy integration of new classification methods:

```python
def train_new_method(self):
    # Implement your custom method
    # Follow the same pattern as existing methods
    pass
```

## 🎓 Educational Value

This project serves as an excellent learning resource for:

- **Computer Vision**: Traditional feature extraction techniques
- **Deep Learning**: CNN architecture and training optimization
- **Machine Learning**: SVM classification and performance evaluation
- **Comparative Analysis**: Understanding trade-offs between different approaches




## 🙏 Acknowledgments

- Google Colab for providing T4 GPU resources
- TensorFlow and scikit-learn communities
- OpenCV for computer vision utilities
- Dataset contributors and signature recognition research community

---

⭐ **Star this repository if you found it helpful!** 