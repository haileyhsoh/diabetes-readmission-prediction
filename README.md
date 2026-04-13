# Diabetes Hospital Readmission Prediction

NVIDIA Omniverse AI Engineering Internship Assignment — predicting hospital readmission outcomes for diabetic patients using AutoGluon.

## Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a diabetic patient will be readmitted to hospital:
- **NO** — not readmitted
- **>30** — readmitted after 30 days
- **<30** — readmitted within 30 days

**Dataset:** 101,766 clinical encounter records (1999–2008) with 50 features including demographics, admission details, diagnoses (ICD-9), lab results, and 23 diabetes medications.

**Approach:** AutoGluon `TabularPredictor` with `best_quality` preset — trains multiple model families (LightGBM, XGBoost, CatBoost, Neural Nets, Random Forests) with 8-fold bagging and multi-layer stacking, then combines them in a weighted ensemble.

**Best Model:** `WeightedEnsemble_L3` — 3-layer stacked ensemble  
**Test Accuracy:** 0.6070  
**Test F1 (Macro):** 0.4144

## Environment Setup

This project uses **uv** as the Python package manager with **Python 3.12**.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
uv pip install -r requirements.txt
```

## Repository Structure

```
├── notebook.ipynb          # Full pipeline: data prep, EDA, training, evaluation
├── train_model.py          # Standalone training script (CLI configurable)
├── inference.py            # Standalone inference script for graders
├── requirements.txt        # Pinned dependencies (uv pip freeze)
├── leaderboard.csv         # AutoGluon model leaderboard (28 models ranked)
├── unseen_data.csv         # 5% stratified sample reserved for grader testing
├── confusion_matrix.png    # Confusion matrix visualisation
├── ag_models/              # Trained AutoGluon model artefacts (see note below)
├── data_diabetes_hospital_readmission_1999-2008/
│   ├── diabetic_data.csv   # Primary dataset (101,766 rows × 50 columns)
│   └── IDS_mapping.csv     # Lookup tables for coded ID columns
└── README.md               # This file
```

> **Note on `ag_models/`:** The trained model directory is ~6 GB (too large for GitHub). Download it from [Google Drive](https://drive.google.com/file/d/1ZIwbVNs7dJG4amVK5YS2ieMIF5eKuzV9/view?usp=sharing), unzip it, and place the `ag_models/` folder in the repository root. Alternatively, retrain from scratch by running `python train_model.py --preset best_quality --time_limit 3600`.

## Running the Training Script

To reproduce the full training pipeline from scratch:

```bash
source .venv/bin/activate

# Medium quality (~10 min, good for testing)
python train_model.py --preset medium_quality --time_limit 600

# Best quality (~45–60 min, produces the submitted model)
python train_model.py --preset best_quality --time_limit 3600
```

This executes: data loading → ID mapping → cleaning → stratified splitting → AutoGluon training → evaluation → artefact saving.

## Running Inference (For Graders)

**Option A — Command line (recommended):**

```bash
source .venv/bin/activate
python inference.py --input unseen_data.csv
```

This loads the trained model from `ag_models/`, runs predictions on the input CSV, and saves results to `predictions.csv` with predicted classes and probability scores.

**Option B — Jupyter notebook:**

Open `notebook.ipynb` and run Section 8 ("Inference on Unseen Data"), which demonstrates the same inference workflow.

**Option C — Python one-liner:**

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

predictor = TabularPredictor.load("ag_models/")
predictions = predictor.predict(pd.read_csv("unseen_data.csv"))
print(predictions.value_counts())
```

## Model Performance

### Leaderboard (Top 5 Models on Hold-Out Test Set)

| Rank | Model | Test F1 (Macro) | Stack Level |
|------|-------|----------------|-------------|
| 1 | WeightedEnsemble_L3 | 0.4144 | 3 |
| 2 | NeuralNetFastAI_BAG_L2 | 0.4143 | 2 |
| 3 | NeuralNetTorch_BAG_L2 | 0.4050 | 2 |
| 4 | NeuralNetFastAI_BAG_L1 | 0.4018 | 1 |
| 5 | NeuralNetTorch_r79_BAG_L2 | 0.3967 | 2 |

28 models were trained in total. See `leaderboard.csv` for the full ranking.

### Classification Report

```
              precision    recall  f1-score   support

         <30       0.36      0.06      0.11      1194
         >30       0.47      0.36      0.41      4223
          NO       0.66      0.82      0.73      7883

    accuracy                           0.61     13300
   macro avg       0.50      0.41      0.41     13300
weighted avg       0.57      0.61      0.57     13300
```

### Key Observations

- The model performs well on "NO" (not readmitted) — F1 = 0.73
- Moderate performance on ">30" (readmitted after 30 days) — F1 = 0.41
- Weak on "<30" (readmitted within 30 days) — F1 = 0.11, expected given only ~9% prevalence
- These results are consistent with published benchmarks on this dataset

## Data Cleaning Decisions

| Step | Action | Justification |
|------|--------|---------------|
| Missing values | Replaced `?` with NaN | Dataset uses `?` as missing indicator |
| `weight` (96.9% missing) | Dropped | Too sparse to impute meaningfully |
| `medical_specialty` (49.1%) | Dropped | High cardinality + half missing |
| `payer_code` (39.6%) | Dropped | Not clinically predictive of readmission |
| `race` (2.2% missing) | Mode imputation | Low missing rate; preserves distribution |
| `diag_1/2/3` (0.02–1.4%) | Filled with "Unknown" | Valid sentinel for categorical codes |
| Expired/hospice patients | Removed (2,413) | Cannot be readmitted — biases model toward "NO" |
| Duplicate encounters | Kept first per patient | Prevents data leakage from repeated patients |

## Dependencies

- **Python:** 3.12.4
- **AutoGluon:** 1.5.0
- **Key packages:** pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.8, seaborn 0.13.2, scipy 1.16.3

Full pinned versions in `requirements.txt` (299 packages).
