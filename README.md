# 🛡️ Malicious Sensor Node Detection

A statistical and machine learning-based approach to identifying compromised nodes in IoT sensor networks. The project explores multiple classification techniques including Random Forest, Support Vector Machines (SVM), XGBoost, and LSTM to detect malicious activity in real-time sensor data.

---

## 📄 Project Overview

The proliferation of IoT devices has made network integrity a crucial concern, especially when malicious sensor nodes attempt to disrupt or hijack the system. This project develops a robust classification system that leverages supervised learning and deep learning to detect such compromised nodes.

By combining classical models like Random Forest and SVM with advanced models like XGBoost and LSTM, we evaluate performance across multiple metrics while accounting for dataset imbalance and feature dimensionality.

---


## 📁 Folder Structure

```plaintext
📁 Malicious-Sensor-Node-Detection/
├── Project_Code.R # File contains the R code of the project
└── README.md # This File
```
---

## 📊 Dataset

**Source:** IEEE DataPort  
**Samples:** 10,000  
**Features:** 21 (e.g., packet rates, duplication, battery, SNR, CPU usage)  
**Target:** `Is_Malicious` (binary flag)

Key preprocessing:
- Removed non-informative columns (e.g., `IP Address`, `Timestamp`)
- Standardized features via Z-score normalization
- Addressed heavy class imbalance using SMOTE and the `ovun.sample()` method from the ROSE library

---

## 🧠 Statistical & Machine Learning Methods

### 🔹 Random Forest
- Ensemble of decision trees trained on feature subsets
- Feature importance analysis included
- Achieved **91.35% accuracy** with reduced model size for efficiency

### 🔹 Support Vector Machines (SVM)
- Kernels used: Radial, Linear, Polynomial, Sigmoid
- Best results achieved with Radial and Linear kernels
- Balanced performance across recall and F1-score (~91%)

### 🔹 XGBoost
- Gradient boosting with regularization
- Hyperparameter-tuned using cross-validation
- Achieved **92.1% accuracy**, **92.5% F1-score**

### 🔹 LSTM (Stacked)
- Deep learning model with two LSTM layers
- Implemented using TensorFlow and Keras
- Used `class_weight` balancing to handle class imbalance
- Achieved **99.83% accuracy** and **99.02% F1-score**

---

## 🔍 Feature Engineering

### Principal Component Analysis (PCA)
- Used to reduce dimensionality and identify informative features
- PC1 explained 44.9% of total variance
- Top predictors: `Packet_Duplication_Rate`, `Data_Throughput`

---

## ✅ Model Comparison

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Random Forest    | 91.35%   | 87.42%    | 94.22% | 91.80%   |
| SVM (Linear)     | 90.98%   | 86.50%    | 97.80% | 91.55%   |
| SVM (Radial)     | 90.55%   | 84.86%    | 98.70% | 91.26%   |
| XGBoost          | 92.10%   | 87.89%    | 97.65% | 92.51%   |
| Stacked LSTM     | 99.83%   | 99.91%    | 98.16% | 99.02%   |

---

## 🧪 Evaluation Metrics

- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **ROC Curve & AUC**
- **Training/Validation Loss (LSTM)**

---

## 🔮 Future Work

- Incorporate real-world network testbeds for validation
- Explore hybrid/ensemble model strategies
- Improve computational efficiency for real-time LSTM inference
- Apply explainable AI (XAI) to interpret predictions

---

## 👥 Team & Contributions

- **Vibha Bhavikatti**  
- **Koushal Parupudi**
- **Mohammad Arfan Ameen** 

---
