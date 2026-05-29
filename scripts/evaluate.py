"""
evaluate.py

Runs full evaluation on the test set and saves all plots to assets/plots/.

Generates:
  - Confusion matrix (assets/plots/confusion_matrix.png)
  - ROC curve       (assets/plots/roc_curve.png)
  - Sample spectrograms side-by-side (assets/plots/sample_spectrograms.png)

Prints to console:
  - Accuracy, Precision, Recall, F1-score, AUC

Run with:
    python scripts/evaluate.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.baseline_model import BaselineDetector
from backend.ml.feature_extractor import extract_mel_spectrogram, save_spectrogram_image
from backend.utils.audio_preprocessor import load_and_preprocess

PLOTS_DIR  = PROJECT_ROOT / "assets" / "plots"
MODEL_PATH = PROJECT_ROOT / "saved_models" / "baseline.pkl"
TEST_CSV   = PROJECT_ROOT / "datasets" / "test.csv"


def run_predictions(detector, csv_path: str):
    """
    Runs predictions on every file in the CSV.
    Returns (y_true, y_pred, y_proba) arrays.
    """
    df      = pd.read_csv(csv_path)
    y_true  = []
    y_pred  = []
    y_proba = []   # raw probabilities for ROC curve

    print(f"  Running predictions on {len(df)} test files...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Predicting"):
        try:
            file_path = PROJECT_ROOT / row['filepath']
            result    = detector.predict(str(file_path))

            y_true.append(int(row['label']))
            y_pred.append(1 if result['label'] == 'FAKE' else 0)
            y_proba.append(result['raw_prob']['fake'] / 100)  # convert % → 0-1

        except Exception as e:
            pass

    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def print_metrics(y_true, y_pred, y_proba):
    """Prints all evaluation metrics to the console."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)

    print("\n" + "═" * 40)
    print("  BASELINE MODEL — TEST RESULTS")
    print("═" * 40)
    print(f"  Accuracy:   {acc*100:.1f}%")
    print(f"  Precision:  {prec*100:.1f}%")
    print(f"  Recall:     {rec*100:.1f}%")
    print(f"  F1-score:   {f1*100:.1f}%")
    print(f"  ROC-AUC:    {auc:.3f}")
    print("═" * 40)

    # Explain each metric briefly
    print(f"\n  Accuracy {acc*100:.1f}%: {acc*100:.1f}% of all files correctly classified")
    print(f"  Precision {prec*100:.1f}%: when model says FAKE, it is right {prec*100:.1f}% of the time")
    print(f"  Recall {rec*100:.1f}%: the model catches {rec*100:.1f}% of all fake files")
    print(f"  AUC {auc:.3f}: ranking quality (0.5=random, 1.0=perfect)")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def plot_confusion_matrix(y_true, y_pred):
    """
    Saves a confusion matrix heatmap.

    The confusion matrix is a 2×2 grid:
    ┌──────────────────┬──────────────────┐
    │ True Neg (TN)    │ False Pos (FP)   │
    │ Predicted Real,  │ Predicted Fake,  │
    │ Actually Real    │ Actually Real    │
    ├──────────────────┼──────────────────┤
    │ False Neg (FN)   │ True Pos (TP)    │
    │ Predicted Real,  │ Predicted Fake,  │
    │ Actually Fake    │ Actually Fake    │
    └──────────────────┴──────────────────┘
    You want TN and TP to be large. FP and FN to be small.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot      = True,     # show numbers inside cells
        fmt        = 'd',      # integer format
        cmap       = 'Blues',  # blue colour scale
        xticklabels = ['Predicted REAL', 'Predicted FAKE'],
        yticklabels = ['Actually REAL', 'Actually FAKE'],
        ax         = ax,
        linewidths = 0.5
    )
    ax.set_title('Confusion Matrix — Baseline Model', fontsize=13, pad=12)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    save_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_roc_curve(y_true, y_proba):
    """
    Saves the ROC (Receiver Operating Characteristic) curve.

    The ROC curve plots:
    - X axis: False Positive Rate (how often we wrongly flag real as fake)
    - Y axis: True Positive Rate  (how often we correctly catch fakes)

    A perfect classifier hugs the top-left corner.
    A random classifier follows the diagonal.
    AUC = Area Under the Curve (bigger = better).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc          = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the ROC curve
    ax.plot(fpr, tpr, color='#185FA5', lw=2, label=f'Baseline model (AUC = {auc:.3f})')

    # Plot the random-chance diagonal
    ax.plot([0, 1], [0, 1], color='#888780', lw=1.2,
            linestyle='--', label='Random chance (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Baseline Model', fontsize=13, pad=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = PLOTS_DIR / "roc_curve.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_sample_spectrograms():
    """
    Creates a side-by-side figure showing 3 real and 3 fake spectrograms.
    This is the most visually impressive output — shows reviewers the
    actual difference the model is learning from.
    """
    import glob

    real_files = glob.glob(str(PROJECT_ROOT / "datasets" / "real" / "*.wav"))[:3]
    fake_files = glob.glob(str(PROJECT_ROOT / "datasets" / "fake" / "*.wav"))[:3]

    if not real_files or not fake_files:
        print("  ⚠  Not enough files for spectrogram comparison")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('Mel Spectrograms: Real vs AI-Generated Voice',
                 fontsize=14, y=0.98)

    for col, wav_path in enumerate(real_files):
        try:
            audio   = load_and_preprocess(wav_path)
            mel     = extract_mel_spectrogram(audio)
            axes[0, col].imshow(mel, aspect='auto', origin='lower', cmap='magma')
            axes[0, col].set_title(f'Real #{col+1}', fontsize=10, color='#27500A')
            axes[0, col].axis('off')
        except Exception:
            axes[0, col].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[0, col].axis('off')

    for col, wav_path in enumerate(fake_files):
        try:
            audio   = load_and_preprocess(wav_path)
            mel     = extract_mel_spectrogram(audio)
            axes[1, col].imshow(mel, aspect='auto', origin='lower', cmap='magma')
            axes[1, col].set_title(f'AI-Generated #{col+1}', fontsize=10, color='#791F1F')
            axes[1, col].axis('off')
        except Exception:
            axes[1, col].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[1, col].axis('off')

    # Add row labels
    fig.text(0.01, 0.72, 'REAL', va='center', rotation='vertical',
             fontsize=12, color='#27500A', fontweight='bold')
    fig.text(0.01, 0.27, 'FAKE', va='center', rotation='vertical',
             fontsize=12, color='#791F1F', fontweight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    save_path = PLOTS_DIR / "sample_spectrograms.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def main():
    print("═" * 50)
    print("  Audio Deepfake Detector — Evaluation")
    print("═" * 50)

    # ── Check files exist ──────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("       Run: python scripts/train_baseline.py first")
        sys.exit(1)

    if not TEST_CSV.exists():
        print(f"\nERROR: Test CSV not found at {TEST_CSV}")
        print("       Run: python scripts/preprocess.py first")
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n  Loading trained model...")
    detector = BaselineDetector()
    detector.load(str(MODEL_PATH))

    # ── Run predictions ────────────────────────────────────────────────────────
    y_true, y_pred, y_proba = run_predictions(detector, str(TEST_CSV))

    if len(y_true) == 0:
        print("ERROR: No predictions made. Check your test CSV file paths.")
        sys.exit(1)

    # ── Print metrics ──────────────────────────────────────────────────────────
    metrics = print_metrics(y_true, y_pred, y_proba)

    # ── Save plots ─────────────────────────────────────────────────────────────
    print("\n  Saving visualisation plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    plot_sample_spectrograms()

    print(f"\n  All plots saved to: {PLOTS_DIR}/")
    print("\n✓ Evaluation complete! Check assets/plots/ for your graphs.")


if __name__ == "__main__":
    main()