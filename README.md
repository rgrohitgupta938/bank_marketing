# Bank Marketing Term Deposit Prediction

## Problem Statement

This project aims to predict whether a customer will subscribe to a term deposit based on data collected from direct marketing campaigns (phone calls) conducted by a Portuguese banking institution.

The target variable:

- `y = 1` → Customer subscribes to term deposit
- `y = 0` → Customer does not subscribe

The dataset is imbalanced (≈ 88% No, 12% Yes), making evaluation beyond simple accuracy essential.

---

## Dataset Overview

- Source: UCI Bank Marketing Dataset
- Rows: 45,211
- Features: 16 input features + 1 target
- Feature Types: Categorical & Numerical

Key features include:

- Age
- Job
- Marital Status
- Education
- Default
- Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign interactions
- Duration
- Previous outcomes

---

## Feature Engineering

The preprocessing pipeline includes:

- Standard Scaling for numerical features
- One-Hot Encoding for categorical features
- Proper train-test split with stratification
- Class weighting to handle imbalance

Pipeline built using `ColumnTransformer`.

---

## Models Trained

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest
- XGBoost

---

## Model Performance Comparison

| Model            | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|------------------|----------|---------|-----------|---------|----------|---------|
| Logistic         | 0.8561   | 0.9040  | 0.4338    | 0.7524  | **0.5503** | **0.4973** |
| Decision Tree   | 0.8742   | 0.5770  | 0.4167    | 0.1890  | 0.2601   | 0.2207  |
| kNN             | 0.8934   | 0.7604  | 0.6224    | 0.2259  | 0.3315   | 0.3311  |
| Naive Bayes     | 0.8518   | 0.8113  | 0.3890    | 0.4669  | 0.4244   | 0.3420  |
| Random Forest   | 0.8959   | **0.9135** | 0.7312    | 0.1749  | 0.2822   | 0.3242  |
| XGBoost         | 0.8886   | 0.9059  | 0.5235    | 0.5369  | 0.5301   | 0.4670  |

---

## Key Insights

### 🔹 Best F1 Score → Logistic Regression

- Highest F1 (0.5503)
- Highest MCC (0.4973)
- Best Recall (0.7524)
- Strong at detecting actual subscribers

### 🔹 Best AUC → Random Forest

- Highest ranking performance (0.9135)
- However, low recall

### 🔹 Most Balanced Model → XGBoost

- Strong overall performance
- Balanced precision and recall

---

## Why Accuracy Is Not Enough

The dataset is highly imbalanced:

- 88% No
- 12% Yes

A model predicting all "No" would already achieve 88% accuracy.

Therefore, we focus on:

- F1 Score
- AUC
- MCC
  
---
