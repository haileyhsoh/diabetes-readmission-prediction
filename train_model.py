"""
Full training pipeline for Diabetes Hospital Readmission Prediction.

Executes: data loading → ID mapping → cleaning → stratified splits →
AutoGluon training → evaluation → artefact saving.

Usage:
    python train_model.py                        # medium_quality smoke test (default)
    python train_model.py --preset best_quality --time_limit 3600   # full run
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Script started", flush=True)

# ── CLI args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="medium_quality",
                    help="AutoGluon preset: medium_quality | best_quality")
parser.add_argument("--time_limit", type=int, default=600,
                    help="Training time limit in seconds")
args = parser.parse_args()

DATA_DIR = "data_diabetes_hospital_readmission_1999-2008"
TARGET = "readmitted"

# ============================================================
# 1. DATA LOADING & ID MAPPING
# ============================================================
print("=" * 60, flush=True)
print("STEP 1: DATA LOADING & ID MAPPING", flush=True)
print("=" * 60, flush=True)

df = pd.read_csv(f"{DATA_DIR}/diabetic_data.csv")
print(f"Loaded diabetic_data.csv: {df.shape[0]:,} rows × {df.shape[1]} columns", flush=True)

# Parse the multi-table IDS_mapping.csv
mappings = {}
current_key = None
current_data = []
with open(f"{DATA_DIR}/IDS_mapping.csv", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line == ",":
            if current_key and current_data:
                mappings[current_key] = dict(current_data)
            current_key = None
            current_data = []
            continue
        if line.endswith(",description"):
            current_key = line.split(",")[0]
            continue
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            current_data.append((int(parts[0].strip()), parts[1].strip()))
if current_key and current_data:
    mappings[current_key] = dict(current_data)

for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(df[col].astype(str))
        print(f"  Mapped {col} → {df[col].nunique()} unique values", flush=True)

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 60, flush=True)
print("STEP 2: DATA CLEANING", flush=True)
print("=" * 60, flush=True)

# 2a. Identify '?' placeholders
missing_counts = (df == "?").sum()
missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("Columns with '?' placeholders:", flush=True)
for col_name, count in missing_cols.items():
    print(f"  {col_name}: {count:,} ({count / len(df) * 100:.1f}%)", flush=True)

df.replace("?", np.nan, inplace=True)

# 2b. Drop columns with >40% missing — too sparse to impute reliably
high_missing = ["weight", "payer_code", "medical_specialty"]
df.drop(columns=high_missing, inplace=True)
print(f"\nDropped high-missing columns (>40% NaN): {high_missing}", flush=True)

# 2c. Fill remaining missing values
df["race"] = df["race"].fillna(df["race"].mode()[0])
print(f"Filled missing race with mode: {df['race'].mode()[0]}", flush=True)

for col in ["diag_1", "diag_2", "diag_3"]:
    n = df[col].isna().sum()
    if n > 0:
        df[col] = df[col].fillna("Unknown")
        print(f"Filled {n} missing {col} with 'Unknown'", flush=True)

# 2d. Remove deceased/hospice patients — they cannot be readmitted,
#     so including them would bias the model toward predicting "NO"
expired_hospice = [
    "Expired",
    "Hospice / home",
    "Hospice / medical facility",
    "Expired at home. Medicaid only, hospice.",
    "Expired in a medical facility. Medicaid only, hospice.",
    "Expired, place unknown. Medicaid only, hospice.",
]
n_before = len(df)
df = df[~df["discharge_disposition_id"].isin(expired_hospice)]
n_removed_expired = n_before - len(df)
print(f"\nRemoved {n_removed_expired:,} expired/hospice patients "
      f"(cannot be readmitted — including them biases toward 'NO')", flush=True)

# 2e. Remove duplicate patients (keep first encounter)
n_before = len(df)
df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
n_removed_dupes = n_before - len(df)
print(f"Removed {n_removed_dupes:,} duplicate patient encounters (kept first)", flush=True)

# 2f. Drop identifier columns (not predictive features)
df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)
print(f"\nFinal cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} columns", flush=True)

# ============================================================
# 3. STRATIFIED DATA SPLITTING
# ============================================================
print("\n" + "=" * 60, flush=True)
print("STEP 3: STRATIFIED DATA SPLITTING", flush=True)
print("=" * 60, flush=True)

# 5% unseen sample (stratified on target)
working_df, unseen_df = train_test_split(
    df, test_size=0.05, stratify=df[TARGET], random_state=42
)

# 80/20 train/test from the remaining 95% (stratified)
train_df, test_df = train_test_split(
    working_df, test_size=0.20, stratify=working_df[TARGET], random_state=42
)

unseen_df.to_csv("unseen_data.csv", index=False)

print(f"Full cleaned dataset: {len(df):>8,} rows", flush=True)
print(f"Unseen sample (5%):   {len(unseen_df):>8,} rows  → saved to unseen_data.csv", flush=True)
print(f"Working data (95%):   {len(working_df):>8,} rows", flush=True)
print(f"  Training set (80%): {len(train_df):>8,} rows", flush=True)
print(f"  Test set (20%):     {len(test_df):>8,} rows", flush=True)

print("\nTarget distribution verification:", flush=True)
for name, split in [("Full", df), ("Unseen", unseen_df),
                     ("Train", train_df), ("Test", test_df)]:
    dist = split[TARGET].value_counts(normalize=True).sort_index()
    print(f"  {name:>7}: " + " | ".join(
        [f"{k}: {v:.3f}" for k, v in dist.items()]
    ), flush=True)

# Save splits so notebook and inference can use them
train_df.to_csv("_train_split.csv", index=False)
test_df.to_csv("_test_split.csv", index=False)

# ============================================================
# 4. AUTOGLUON TRAINING
# ============================================================
print("\n" + "=" * 60, flush=True)
print(f"STEP 4: AUTOGLUON TRAINING (preset={args.preset}, "
      f"time_limit={args.time_limit}s)", flush=True)
print("=" * 60, flush=True)

from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=TARGET,
    eval_metric="f1_macro",
    path="ag_models/"
).fit(
    train_data=train_df,
    presets=args.preset,
    time_limit=args.time_limit,
    verbosity=2
)

print("\nTraining complete!", flush=True)

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n" + "=" * 60, flush=True)
print("STEP 5: MODEL EVALUATION", flush=True)
print("=" * 60, flush=True)

leaderboard = predictor.leaderboard(data=test_df, silent=False)
leaderboard.to_csv("leaderboard.csv", index=False)
print("Leaderboard saved to leaderboard.csv", flush=True)

y_true = test_df[TARGET]
y_pred = predictor.predict(test_df)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
print(f"\nAccuracy:         {acc:.4f}", flush=True)
print(f"F1 Score (macro): {f1:.4f}", flush=True)
print("\nClassification Report:", flush=True)
print(classification_report(y_true, y_pred), flush=True)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=predictor.class_labels)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=predictor.class_labels,
            yticklabels=predictor.class_labels, ax=ax)
ax.set_title("Confusion Matrix — Hold-Out Test Set",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Confusion matrix saved to confusion_matrix.png", flush=True)

# ── Quick smoke test: can we load the model and run inference? ──
print("\n" + "=" * 60, flush=True)
print("STEP 6: INFERENCE SMOKE TEST", flush=True)
print("=" * 60, flush=True)

loaded = TabularPredictor.load("ag_models/")
unseen = pd.read_csv("unseen_data.csv")
preds = loaded.predict(unseen)
print(f"Inference on unseen_data.csv ({len(unseen)} rows) succeeded.", flush=True)
print(f"Prediction distribution:\n{preds.value_counts().to_string()}", flush=True)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60, flush=True)
print("PIPELINE COMPLETE", flush=True)
print("=" * 60, flush=True)
print(f"Best model:         {predictor.model_best}", flush=True)
print(f"Models trained:     {len(leaderboard)}", flush=True)
print(f"Test accuracy:      {acc:.4f}", flush=True)
print(f"Test F1 (macro):    {f1:.4f}", flush=True)
print(f"\nArtefacts saved:", flush=True)
print(f"  ag_models/           — trained model artefacts", flush=True)
print(f"  leaderboard.csv      — model ranking", flush=True)
print(f"  unseen_data.csv      — 5% held-out data for graders", flush=True)
print(f"  confusion_matrix.png — evaluation chart", flush=True)
print(f"  requirements.txt     — environment specification", flush=True)
