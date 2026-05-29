"""
metrics.py

GET /metrics

Returns all training history and evaluation data for the frontend dashboard.
Reads from:
  - saved_models/training_history.json  (created by train_cnn.py)
  - assets/plots/                       (PNG files — returned as base64)

Response shape (abbreviated):
{
  "cnn": {
    "epochs_trained": 23,
    "best_val_acc": 0.843,
    "train_loss": [...],
    "val_loss":   [...],
    "train_acc":  [...],
    "val_acc":    [...]
  },
  "baseline": {
    "available": true
  },
  "plots": {
    "confusion_matrix_b64": "...",
    "roc_curve_b64":        "...",
    "training_curves_b64":  "..."
  },
  "dataset": {
    "total_samples": 800,
    "real_count":    400,
    "fake_count":    400,
    "train_count":   560,
    "val_count":     120,
    "test_count":    120
  }
}
"""

import json
import base64
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / "saved_models" / "training_history.json"
PLOTS_DIR    = PROJECT_ROOT / "assets" / "plots"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def image_to_base64(image_path: Path) -> str:
    """
    Reads a PNG file from disk and returns it as a base64 string.
    The frontend can use this directly in an <img> tag:
      <img src={`data:image/png;base64,${base64string}`} />
    """
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_dataset_stats() -> dict:
    """
    Reads the CSV manifest files and counts real/fake samples per split.
    Returns a dict with counts for the dashboard stat cards.
    """
    import pandas as pd

    stats = {
        "total_samples": 0,
        "real_count":    0,
        "fake_count":    0,
        "train_count":   0,
        "val_count":     0,
        "test_count":    0
    }

    for split in ["train", "val", "test"]:
        csv_path = DATASETS_DIR / f"{split}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                count = len(df)
                stats[f"{split}_count"]  = count
                stats["total_samples"]  += count
                stats["real_count"]     += int((df["label"] == 0).sum())
                stats["fake_count"]     += int((df["label"] == 1).sum())
            except Exception:
                pass

    return stats


def get_cnn_history() -> dict:
    """
    Loads training_history.json and returns its contents.
    Returns a dict with 'available': False if the file doesn't exist yet.
    """
    if not HISTORY_PATH.exists():
        return {
            "available":     False,
            "message":       "CNN not trained yet. Run: python scripts/train_cnn.py"
        }

    try:
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)

        return {
            "available":      True,
            "epochs_trained": history.get("epochs_trained", 0),
            "best_val_acc":   round(history.get("best_val_acc", 0), 4),
            "best_val_loss":  round(history.get("best_val_loss", 0), 4),
            "train_loss":     history.get("train_loss", []),
            "val_loss":       history.get("val_loss", []),
            "train_acc":      [round(a * 100, 2) for a in history.get("train_acc", [])],
            "val_acc":        [round(a * 100, 2) for a in history.get("val_acc", [])],
            "hyperparams":    history.get("hyperparams", {})
        }
    except Exception as e:
        return {"available": False, "message": str(e)}


@router.get("/metrics")
async def get_metrics():
    """
    Returns all training metrics, plot images, and dataset statistics.

    This endpoint is called by the frontend Dashboard page.
    It reads files from disk rather than re-computing anything,
    so it's very fast (just file reads + JSON serialisation).
    """
    # Gather all data
    cnn_history   = get_cnn_history()
    dataset_stats = get_dataset_stats()

    # Load plot images as base64
    plots = {
        "confusion_matrix_b64": image_to_base64(PLOTS_DIR / "confusion_matrix.png"),
        "roc_curve_b64":        image_to_base64(PLOTS_DIR / "roc_curve.png"),
        "training_curves_b64":  image_to_base64(PLOTS_DIR / "training_curves.png"),
        "spectrograms_b64":     image_to_base64(PLOTS_DIR / "sample_spectrograms.png")
    }

    # Check which models are saved on disk
    baseline_available = (PROJECT_ROOT / "saved_models" / "baseline.pkl").exists()
    cnn_available      = (PROJECT_ROOT / "saved_models" / "best_cnn.pt").exists()

    return JSONResponse(content={
        "cnn":      cnn_history,
        "baseline": {"available": baseline_available},
        "plots":    plots,
        "dataset":  dataset_stats,
        "models_saved": {
            "cnn":      cnn_available,
            "baseline": baseline_available
        }
    })