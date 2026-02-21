# AE_README.md

## Overview
This notebook implements a semi-supervised autoencoder (AE) model for anti-money laundering (AML) anomaly detection. It combines a denoising AE with IsolationForest, calibrated via LogisticRegression, to score customers as suspicious (1) or normal (0). The model prioritizes high true positives (TP) with controlled false positives (FP), achieving AUC-ROC ≈ 0.822. Procedures include data preprocessing, semi-supervised training, threshold tuning, and AML-friendly evaluations.

## Data Preparation
- **Load Data**: Read CSV (`customer_clusters.csv`), optional sampling (e.g., 2M rows max). Validate columns: `customer_id`, `label` (0=normal, 1=suspicious, NaN=unlabeled), `cluster` (0-6).
- **Feature Engineering**:
  - Numeric: Median impute, RobustScaler (fit on normals only).
  - Categorical: FeatureHasher (256 dims) for dense encoding.
  - Cluster: One-hot encode (7 buckets + missing).
  - Output: `X_all` (float32 matrix), `feature_names` list.
- **Splitting**: Seed normals for training, unlabeled for expansion.

## Modeling Steps
1. **AE Training (Semi-Supervised)**:
   - Train initial AE on seed normals (denoising with noise).
   - Score unlabeled data; add bottom 20% (likely normals) to expand training set.
   - Retrain AE on expanded normals.
2. **Anomaly Scoring**:
   - Compute AE reconstruction errors for all customers.
   - Fit IsolationForest on expanded normals; get anomaly scores.
   - Combine AE + IF scores via rank-based mean.
3. **Calibration**:
   - Use LogisticRegression on labeled subset to calibrate combined scores into probabilities (if labels available).
4. **Threshold Selection**:
   - Optimize on labeled data: Maximize TP subject to FPR ≤ 2% (or min precision 0.30).
   - Fallback: Maximize F2 score if constraints unmet.

## Evaluation
- **Metrics**: Confusion matrix (TP, FP, FN, TN), precision, recall, FPR. AUC-ROC, AUC-PR.
- **Plots**: Confusion matrix, ROC curve, PR curve.
- **Feature Importance**: Per-feature reconstruction error differences (Suspicious - Normal); plot top 5 interpretable features (numeric + cluster columns).
- **Segmented Analysis**:
  - Top 5% high-risk tables/plots by customer_type (SB vs IND).
  - Confusion matrices by type using global threshold.
- **Output**: CSV with `customer_id`, `predicted_label`, `risk_score`.

## Key Hyperparameters
- AE: 128 hidden, 32 bottleneck, 40 epochs, LR=5e-4, WD=1e-5.
- IF: 200 estimators, auto contamination.
- Threshold: MAX_FPR=0.02.

## Usage Notes
- Run cells sequentially; rebuild `feature_names` if mismatches occur.
- AML Focus: Interpretable features, ranked reviews, controlled FP.
- Dependencies: PyTorch, scikit-learn, pandas, matplotlib.

