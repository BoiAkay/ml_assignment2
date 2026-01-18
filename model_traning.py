# -*- coding: utf-8 -*-
"""
training_pipeline.py
--------------------
Script to train ML models on the Dry Bean dataset and save performance metrics.
Includes interactive progress tracking with persistent progress bar.
"""

import os
import sys  # NEW: Imported sys to synchronize output streams
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm  # Import for Progress Bar

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Constants
MODEL_DIR = 'model'
OUTPUT_METRICS = 'bean_model_performance.csv'
CORR_IMAGE = 'feature_heatmap.png'

# ---------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------
print(">>> Initializing Data Fetch...")

try:
    # Fetching ID 602: Dry Bean Dataset
    repo_data = fetch_ucirepo(id=602)
    features = repo_data.data.features
    targets = repo_data.data.targets

    # Flatten target if it's a DataFrame
    if isinstance(targets, pd.DataFrame):
        targets = targets.iloc[:, 0]

    print(f"    Dataset Ready: {features.shape[0]} samples, {features.shape[1]} features.")

except Exception as err:
    print(f"!!! Critical Error loading data: {err}")
    exit()

# ---------------------------------------------------------
# 2. EXPLORATORY ANALYSIS (Correlation)
# ---------------------------------------------------------
print("\n>>> Generating Feature Correlation Heatmap...")

plt.figure(figsize=(11, 9))
correlation_df = features.corr()

# Using 'viridis' colormap
sns.heatmap(
    correlation_df, 
    annot=True, 
    fmt=".2f", 
    cmap='viridis', 
    linewidths=0.5,
    cbar_kws={'shrink': .8}
)

plt.title("Feature Correlation Analysis - Dry Bean Data")
plt.tight_layout()
plt.savefig(CORR_IMAGE, dpi=300)
print(f"    Saved heatmap to '{CORR_IMAGE}'")

# ---------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------
# Encode string labels to integers
lbl_enc = LabelEncoder()
y_encoded = lbl_enc.fit_transform(targets)

# Stratified Split (80/20)
X_trn, X_tst, y_trn, y_tst = train_test_split(
    features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scaling
std_scaler = StandardScaler()
X_trn_sc = std_scaler.fit_transform(X_trn)
X_tst_sc = std_scaler.transform(X_tst)

# ---------------------------------------------------------
# 3.1 TEST SET HEATMAP
# ---------------------------------------------------------
print("\n>>> Generating Test Set Heatmap...")
test_merged_df = pd.DataFrame(X_tst_sc, columns=features.columns)
test_merged_df['Target_Class'] = y_tst

plt.figure(figsize=(12, 10))
sns.heatmap(
    test_merged_df.corr(), 
    annot=True, 
    fmt=".2f", 
    cmap='magma', 
    linewidths=0.5
)
plt.title("Test Set Correlation (Scaled Features + Target)")
plt.tight_layout()
plt.savefig('test_data_correlation.png', dpi=300)
print("    Saved test set heatmap.")


# ---------------------------------------------------------
# 4. MODELING UTILITIES
# ---------------------------------------------------------

def manage_model_training(algo_name, file_name, clf, x_data, y_data):
    """
    Checks for existing model file; trains and saves if missing.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    full_path = os.path.join(MODEL_DIR, file_name)

    if os.path.exists(full_path):
        # Notify user (via tqdm.write in main loop)
        return joblib.load(full_path), "LOADED from disk"
    else:
        clf.fit(x_data, y_data)
        joblib.dump(clf, full_path)
        return clf, "TRAINED and saved"

def calculate_performance(algo_name, model, x_val, y_val):
    """Generates a dictionary of performance metrics."""
    preds = model.predict(x_val)
    
    # Calculate AUC safely
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_val)
            auc_val = roc_auc_score(y_val, probs, multi_class='ovr', average='weighted', labels=model.classes_)
        else:
            auc_val = 0.0
    except:
        auc_val = 0.0

    return {
        "Algorithm": algo_name,
        "Accuracy": accuracy_score(y_val, preds),
        "AUC": auc_val,
        "Precision": precision_score(y_val, preds, average='weighted'),
        "Recall": recall_score(y_val, preds, average='weighted'),
        "F1-Score": f1_score(y_val, preds, average='weighted'),
        "MCC": matthews_corrcoef(y_val, preds)
    }

# ---------------------------------------------------------
# 5. EXECUTION PIPELINE (Interactive)
# ---------------------------------------------------------

# Dictionary of classifiers
# Removed verbose=1 from LogisticRegression to prevent progress bar conflicts
classifiers = [
    ("Logistic Regression", "log_reg.pkl", LogisticRegression(max_iter=1500, random_state=42)),
    ("Decision Tree", "dt_clf.pkl", DecisionTreeClassifier(random_state=42)),
    ("K-Nearest Neighbors", "knn.pkl", KNeighborsClassifier(n_neighbors=5)),
    ("Gaussian NB", "gnb.pkl", GaussianNB()),
    ("Random Forest", "rf_clf.pkl", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("XGBoost", "xgb.pkl", XGBClassifier(eval_metric='mlogloss', random_state=42))
]

performance_log = []

print("\n>>> Starting Model Evaluation Loop...")

# NEW: Progress Bar Implementation
# leave=True: Ensures the bar stays visible after completion
# file=sys.stdout: Ensures the bar plays nicely with print statements
# dynamic_ncols=True: Adjusts to terminal width to prevent line wrapping
with tqdm(classifiers, desc="Initializing Models", unit="model", leave=True, file=sys.stdout, dynamic_ncols=True) as pbar:
    for name, f_name, obj in pbar:
        # Update text to show current model
        pbar.set_description(f"Processing {name}")
        
        # Train or Load
        trained_model, status = manage_model_training(name, f_name, obj, X_trn_sc, y_trn)
        
        # Print status above the bar using write()
        pbar.write(f"   > {name}: {status}")
        
        # Calculate Metrics
        metrics = calculate_performance(name, trained_model, X_tst_sc, y_tst)
        performance_log.append(metrics)

# ---------------------------------------------------------
# 6. EXPORT RESULTS
# ---------------------------------------------------------
final_df = pd.DataFrame(performance_log)
final_df = final_df.sort_values(by="Accuracy", ascending=False)

col_order = ["Algorithm", "Accuracy", "AUC", "Precision", "Recall", "F1-Score", "MCC"]
final_df = final_df[col_order]

final_df.to_csv(OUTPUT_METRICS, index=False)

print(f"\n{'-'*60}")
print(f"Pipeline Complete. Metrics saved to '{OUTPUT_METRICS}'")
print(f"{'-'*60}")
print(final_df)