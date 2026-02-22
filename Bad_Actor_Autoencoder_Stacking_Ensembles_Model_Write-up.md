# AE_README.md

## Files for Modeling
Both the two files `IMI_Modeling_Autoencoder_v3` and `v7` contains the model pipelines (the Denoising Autoencoder AE Stacking Ensembles ) we train and predict to produce the final `model_output.csv` predictions, though `v7` includes our PCA cluster feature as our training indicator.

## Overview
This notebook implements a semi-supervised autoencoder (AE) model for anti-money laundering (AML) anomaly detection. It combines a denoising AE with IsolationForest, calibrated via LogisticRegression, to score customers as suspicious (1) or normal (0). The model prioritizes high true positives (TP) with controlled false positives (FP), achieving AUC-ROC ≈ 0.822, AUC-PR = 0.21 and Precision around 30%. Procedures include data preprocessing, semi-supervised training, threshold tuning, and AML-friendly evaluations.

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

## Stacking Ensembles
### This is a useful modeling trick used in dealing with highly imbalanced AML data.

- Base Models: Autoencoder AE and Isolation Forest IF
- Ensemble Combination
   - Initial combination: AE and IF scores are averaged using rank-based percentiles to create a preliminary risk score
   - Stacking Meta-Model: A LogisticRegression model is trained on the labeled subset using the base models' scores as features
   -  It outputs calibrated probabilities as the final risk_score (predicting suspicious probability [0,1])
**The meta-model (LogisticRegression) learns to combine predictions from the base models (AE and IF), improving separation and TP/FP tradeoffs. Overall, this ensemble stacking pipeline: AE captures reconstruction-based anomalies, IF catches isolation-based patterns, and stacking optimizes their integration for AML detection.**

## Evaluation
- **Metrics**: Confusion matrix (TP, FP, FN, TN), precision, recall, FPR. AUC-ROC, AUC-PR.
- **Plots**: Confusion matrix, ROC curve, PR curve.
- **Feature Importance**: Per-feature reconstruction error differences (Suspicious - Normal); plot top 5 interpretable features (numeric + cluster columns).
- **Segmented Analysis**:
  - Top 5% high-risk tables/plots by customer_type (SB vs IND).
  - Confusion matrices by type using global threshold.
- **Output**: CSV with `customer_id`, `predicted_label`, `risk_score`.

## How Flagging Works for our "bad actor model"

This project produces a **customer-level AML risk score** and a **binary flag** indicating whether a customer should be prioritized for investigation. The final deliverable is a single file: `model_output.csv`.


### 1 What the model learns

We train an **Autoencoder (AE)** using customer-level behavioral features engineered from transaction activity (e.g., inflow/outflow consistency, spending intensity, structuring signals, velocity patterns, geographic mismatch) and a **cluster/segment feature** (7 behavioral clusters: 0–6).

The AE is trained primarily on **normal / unlabeled** customers to learn a baseline of “typical” behavior patterns.  
Customers whose behavior deviates from this baseline are expected to have higher reconstruction error.


### 2 How the AML risk score is computed

For each customer, the model outputs a continuous **anomaly score** based on **reconstruction error**:

- Input: feature vector `x`
- AE reconstruction: `x_hat`
- Reconstruction error (per customer):  
  \[
  \text{error}(x) = \text{mean}\left((x - \hat{x})^2\right)
  \]

This error is then converted into a normalized **risk_score ∈ [0, 1]**, where:
- **Higher risk_score** = more anomalous behavior relative to learned normal patterns
- Scores are comparable across customers and can be used for ranking and thresholding

### 3 How the model flags “bad actors” (predicted_label)

The binary prediction is generated by applying a threshold to the risk score:

- `predicted_label = 1`  → flagged for further investigation  
- `predicted_label = 0`  → not flagged  

Thresholding is operationally defined to match investigation capacity and imbalanced AML labels. In our final setting, we flag customers in the **top 1%** of risk scores (within each customer type where applicable), i.e., customers with the highest anomaly scores.

This approach ensures the model:
- focuses on a manageable set of highest-risk customers
- maximizes “hit rate” (precision) in a low-prevalence environment


### 4 Why clustering is included

Customers behave differently across segments (e.g., Individuals vs Small Business).  
We incorporate a **precomputed cluster feature (0–6)** to help the model learn different “normal baselines” across behavioral groups.

This typically improves **alert quality** (precision) by reducing false alarms caused by legitimate segment-specific behavior patterns.



### 5 Output file: `model_output.csv`

The final file contains one row per customer in the KYC population:

| Column | Type | Meaning |
|--------|------|---------|
| `customer_id` | text | Unique customer identifier from KYC files |
| `predicted_label` | int (0/1) | 1 = flagged for investigation, 0 = not flagged |
| `risk_score` | float in [0,1] | Continuous AML risk score used for ranking customers |

**Important requirement:** Every customer in the KYC input appears exactly once in `model_output.csv`.



### 6 How to interpret the outputs (AML analyst view)

- Use `risk_score` to **rank customers** for investigation.
- `predicted_label=1` indicates customers above the chosen threshold (top risk tail).
- Feature drivers can be interpreted using **reconstruction-error contribution**, i.e., which features are hardest for the model to reconstruct for suspicious customers (e.g., income mismatch, near-threshold activity, velocity volatility).


## Key Hyperparameters
- AE: 128 hidden, 32 bottleneck, 40 epochs, LR=5e-4, WD=1e-5.
- IF: 200 estimators, auto contamination.
- Threshold: MAX_FPR=0.02.

## Usage Notes
- Run cells sequentially; rebuild `feature_names` if mismatches occur.
- AML Focus: Interpretable features, ranked reviews, controlled FP.
- Dependencies: PyTorch, scikit-learn, pandas, matplotlib.

