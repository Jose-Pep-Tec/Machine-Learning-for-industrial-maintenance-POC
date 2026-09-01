# Predictive Maintenance ML

Machine learning model for predicting industrial equipment failures using the AI4I 2020 dataset. Achieves 92.6% recall with XGBoost.

## Overview

This project compares four models (Logistic Regression, Random Forest, XGBoost, HistGradientBoosting) to detect machine failures. The best model (XGBoost) is deployed for real-time inference.

## Results

| Model | Recall | Precision | F1 | AUC |
|-------|--------|-----------|-----|-----|
| XGBoost | **92.6%** | 31.7% | 0.47 | **0.969** |
| HistGradientBoosting | 92.6% | 25.1% | 0.39 | 0.962 |
| RandomForest | 80.9% | 48.2% | 0.60 | 0.955 |
| LogisticRegression | 86.8% | 17.9% | 0.30 | 0.931 |

## Installation

pip install -r requirements.txt

#usage

# Train model
python src/train_model.py

# Run inference
python src/inference.py
