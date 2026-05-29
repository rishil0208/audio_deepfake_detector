"""
train_baseline.py

Runs the complete training pipeline for the baseline model:
  1. Trains on datasets/train.csv
  2. Evaluates on datasets/val.csv (validation check)
  3. Saves model to saved_models/baseline.pkl

Run with:
    python scripts/train_baseline.py
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.baseline_model import BaselineDetector


def evaluate_on_split(detector, csv_path: str, split_name: str):
    """
    Evaluates the trained model on a CSV split (val or test).
    Prints accuracy and a simple breakdown.
    """
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, classification_report

    print(f"\n── Evaluating on {split_name} set ──")

    df     = pd.read_csv(csv_path)
    y_true = []
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            file_path = PROJECT_ROOT / row['filepath']
            result    = detector.predict(str(file_path))

            y_true.append(int(row['label']))
            y_pred.append(1 if result['label'] == 'FAKE' else 0)

        except Exception:
            pass   # skip bad files

    if not y_true:
        print("  No predictions made — check your CSV paths")
        return

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n  {split_name} accuracy: {accuracy*100:.1f}%")
    print(f"\n  Detailed report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['REAL', 'FAKE'],
        digits=3
    ))


def main():
    # ── Paths ──────────────────────────────────────────────────────────────────
    TRAIN_CSV  = PROJECT_ROOT / "datasets" / "train.csv"
    VAL_CSV    = PROJECT_ROOT / "datasets" / "val.csv"
    MODEL_PATH = PROJECT_ROOT / "saved_models" / "baseline.pkl"

    # ── Check that training data exists ───────────────────────────────────────
    if not TRAIN_CSV.exists():
        print("ERROR: datasets/train.csv not found!")
        print("       Run this first: python scripts/preprocess.py")
        sys.exit(1)

    # ── Create output directory ────────────────────────────────────────────────
    (PROJECT_ROOT / "saved_models").mkdir(exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    detector = BaselineDetector()
    detector.train(str(TRAIN_CSV), str(MODEL_PATH))

    # ── Validate ──────────────────────────────────────────────────────────────
    evaluate_on_split(detector, str(VAL_CSV), "Validation")

    # ── Quick demo prediction ─────────────────────────────────────────────────
    print("\n── Quick demo predictions ──")
    import glob
    real_files = glob.glob(str(PROJECT_ROOT / "datasets" / "real" / "*.wav"))
    fake_files = glob.glob(str(PROJECT_ROOT / "datasets" / "fake" / "*.wav"))

    if real_files:
        result = detector.predict(real_files[0])
        status = "✓" if result["label"] == "REAL" else "✗"
        print(f"  {status} Real file → predicted: {result['label']}  ({result['confidence']}% confidence)")

    if fake_files:
        result = detector.predict(fake_files[0])
        status = "✓" if result["label"] == "FAKE" else "✗"
        print(f"  {status} Fake file → predicted: {result['label']}  ({result['confidence']}% confidence)")

    print("\n✓ Baseline training complete!")
    print(f"  Model saved at: {MODEL_PATH}")
    print("  Next: run python scripts/evaluate.py for full metrics")


if __name__ == "__main__":
    main()