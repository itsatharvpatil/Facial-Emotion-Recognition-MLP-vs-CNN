# FER-2013+ Emotion Recognition - MLP vs ResMLP vs CNN

A comprehensive comparison of **MLP**, **ResMLP**, and **CNN** architectures for facial emotion recognition using the **FER-2013+** (cleaned labels) dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Results

| Model   | Parameters | Test Accuracy | Top-3 Accuracy | Efficiency (Acc/M) | Best Epoch |
|---------|------------|---------------|----------------|--------------------|------------|
| **MLP** | 7.52M      | **51.8%**     | 81.2%          | 6.89               | 89         |
| ResMLP  | 9.20M      | 50.1%         | 81.8%          | 5.45               | 74         |
| **CNN** | **1.32M**  | **68.7%**     | **92.1%**      | **52.05**          | 52         |

> **CNN achieves ~17% higher accuracy** while using **~6x fewer parameters** than the MLP.

## 🚀 Project Highlights

- **Cleaned FER-2013+** dataset with better labels
- Strong data augmentation + **MixUp** (CNN)
- Proper regularization (BatchNorm, Dropout, Label Smoothing, Weight Decay)
- OneCycleLR + Cosine Annealing schedulers
- Detailed training curves, Top-K accuracy, and comprehensive comparison
- Full reproducibility with Google Drive persistence

## 🧠 Models Compared

### 1. FER_MLP
- Wide 4-layer fully connected network (4096 → 2048 → 1024)
- Heavy regularization (Dropout + BatchNorm)

### 2. ResMLP
- MLP with residual blocks for better gradient flow
- 1024-dim residual pathway

### 3. EmotionCNN (Best Performer)
- 3-block CNN (64 → 128 → 256 channels)
- MixUp augmentation + strong spatial feature learning
- **Significantly outperforms** both MLP variants

## ✨ Features

- Reproducible training with fixed seeds
- Class-weighted loss handling imbalance
- Top-1, Top-2, Top-3 accuracy evaluation
- Automatic best model saving + training history
- Beautiful Matplotlib visualizations
- Google Drive integration for persistence

## 🛠️ How to Run (Step by Step)

### Recommended Order:

1. **Run `fermlp.ipynb`** → Trains the MLP model
2. **Run `fercnn.ipynb`** → Trains the CNN model (with MixUp + OneCycleLR)
3. **Run `fer+cnnmlp.ipynb`** → Runs comparison and generates final visualizations

> All notebooks automatically save results to Google Drive:  
> `MyDrive/FER2013_Project/`

### Requirements
- Google Colab with **GPU** runtime (T4 recommended)
- Kaggle account (for dataset download — handled automatically)

## ✨ Key Features

- **Cleaned FER-2013+** dataset
- Strong data augmentation
- MixUp augmentation in CNN
- Proper schedulers (`OneCycleLR` for CNN, Cosine for MLP/ResMLP)
- Class-weighted loss for imbalance handling
- Top-1, Top-2, Top-3 accuracy
- Training history + beautiful plots
- Google Drive persistence


## 📈 Key Insights (After Training)

- CNN significantly outperforms both MLP variants due to spatial feature learning.
- Residual connections in pure MLPs give limited improvement.
- MixUp + OneCycleLR provides substantial boost to CNN.

## 📂 Google Drive Output

After training, you will find in your Drive:
- `mlp_best_model.pth`, `resmlp_best_model.pth`, `cnn_best_model.pth`
- Training curves (`*_training_curves.png`)
- `three_way_comparison.png`
- JSON result files

## 🚀 Future Work

- EfficientNet / MobileNet backbone
- Model ensemble
- Real-time webcam demo
- Confusion matrix analysis
- Quantization for deployment

---
