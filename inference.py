"""
Inference script for Diabetes Hospital Readmission Prediction.

Loads a trained AutoGluon model and generates predictions on unseen data.
Designed for graders to run with a single command.

Usage:
    python inference.py                            # default: unseen_data.csv
    python inference.py --input custom_data.csv    # specify input file
    python inference.py --input data.csv --output results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor


def load_and_predict(input_path: str, model_path: str = "ag_models/",
                     output_path: str = "predictions.csv") -> pd.DataFrame:
    """
    Load a trained AutoGluon predictor and run inference on the given CSV.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file containing patient records.
    model_path : str
        Path to the directory containing saved AutoGluon model artefacts.
    output_path : str
        Path where the predictions CSV will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus prediction and probability columns.
    """
    if not Path(model_path).exists():
        print(f"ERROR: Model directory '{model_path}' not found.")
        print("Please ensure ag_models/ is present, or download it per the README.")
        sys.exit(1)

    if not Path(input_path).exists():
        print(f"ERROR: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    predictor = TabularPredictor.load(model_path)

    print(f"Reading input data from {input_path} ...")
    data = pd.read_csv(input_path)
    print(f"  Loaded {len(data):,} rows × {data.shape[1]} columns")

    print("Generating predictions ...")
    predictions = predictor.predict(data)
    probabilities = predictor.predict_proba(data)

    results = data.copy()
    results["predicted_readmitted"] = predictions.values
    for col in probabilities.columns:
        results[f"prob_{col}"] = probabilities[col].values

    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"Prediction distribution:\n{predictions.value_counts().to_string()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on diabetes readmission data"
    )
    parser.add_argument("--input", default="unseen_data.csv",
                        help="Path to input CSV file (default: unseen_data.csv)")
    parser.add_argument("--model", default="ag_models/",
                        help="Path to AutoGluon model directory (default: ag_models/)")
    parser.add_argument("--output", default="predictions.csv",
                        help="Path for output predictions CSV (default: predictions.csv)")
    args = parser.parse_args()

    load_and_predict(args.input, args.model, args.output)
