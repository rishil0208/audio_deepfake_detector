"""
preprocess.py

What this does:
1. Scans datasets/real/ and datasets/fake/ for all .wav files
2. Preprocesses each file using audio_preprocessor.py
3. Saves preprocessed arrays as .npy files (fast to load during training)
4. Creates train.csv, val.csv, test.csv with columns: filepath, label

The CSV files are what your model reads during training.
Each row = one audio clip. Like this:
  filepath,                         label
  datasets/processed/real_0001.npy, 0
  datasets/processed/fake_0001.npy, 1

Label 0 = real, Label 1 = fake

Why do we split into train/val/test?
- train: model learns from these (70%)
- val:   we check the model during training to avoid overfitting (15%)
- test:  final honest evaluation after training is done (15%)
  (never used during training — keeps the evaluation honest)
"""

import sys
import os
from pathlib import Path

# Add the project root to Python's search path so we can import our modules
# This is needed because preprocess.py is in scripts/ but audio_preprocessor.py
# is in backend/utils/ — Python needs to know where to look
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from backend.utils.audio_preprocessor import preprocess_and_save

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR   = PROJECT_ROOT / "datasets"
REAL_DIR      = DATASET_DIR / "real"
FAKE_DIR      = DATASET_DIR / "fake"
PROCESSED_DIR = DATASET_DIR / "processed"    # where .npy files go

# ── Settings ───────────────────────────────────────────────────────────────────
TEST_SIZE  = 0.15    # 15% of data for testing
VAL_SIZE   = 0.15    # 15% of remaining data for validation
SEED       = 42      # random seed — keeps the split the same every time you run
MAX_FILES  = 500     # max files per class (keeps training fast for demo)
                     # set to None to use all files


def main():
    print("=" * 55)
    print("  Audio Deepfake Detector — Preprocessing Pipeline")
    print("=" * 55)
    
    # ── Step 1: Find all audio files ──────────────────────────────────────
    print("\n[1/4] Scanning for audio files...")
    
    real_files = sorted(REAL_DIR.glob("*.wav"))
    fake_files = sorted(FAKE_DIR.glob("*.wav"))
    
    if len(real_files) == 0:
        print("ERROR: No files found in datasets/real/")
        print("       Run: python scripts/download_dataset.py first")
        sys.exit(1)
    
    if len(fake_files) == 0:
        print("ERROR: No files found in datasets/fake/")
        print("       Run: python scripts/download_dataset.py first")
        sys.exit(1)
    
    # Cap at MAX_FILES per class for speed
    if MAX_FILES:
        real_files = real_files[:MAX_FILES]
        fake_files = fake_files[:MAX_FILES]
    
    print(f"  Found {len(real_files)} real files")
    print(f"  Found {len(fake_files)} fake files")
    
    # ── Step 2: Preprocess all files ──────────────────────────────────────
    print(f"\n[2/4] Preprocessing audio files → saving to {PROCESSED_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []     # will collect (processed_path, label) for each file
    
    # Process REAL files (label = 0)
    print("  Processing real audio...")
    for wav_path in tqdm(real_files, desc="  Real"):
        # Create the output path: datasets/processed/real_0001.npy
        out_path = PROCESSED_DIR / f"real_{wav_path.stem}.npy"
        
        success = preprocess_and_save(str(wav_path), str(out_path))
        
        if success:
            # Store relative path (makes the CSV portable across computers)
            records.append({
                "filepath": str(out_path.relative_to(PROJECT_ROOT)),
                "label": 0    # 0 = real
            })
    
    # Process FAKE files (label = 1)
    print("  Processing fake audio...")
    for wav_path in tqdm(fake_files, desc="  Fake"):
        out_path = PROCESSED_DIR / f"fake_{wav_path.stem}.npy"
        
        success = preprocess_and_save(str(wav_path), str(out_path))
        
        if success:
            records.append({
                "filepath": str(out_path.relative_to(PROJECT_ROOT)),
                "label": 1    # 1 = fake
            })
    
    print(f"\n  Successfully processed: {len(records)} files")
    
    # ── Step 3: Create DataFrame and split ────────────────────────────────
    print("\n[3/4] Splitting into train / validation / test sets...")
    
    df = pd.DataFrame(records)
    
    # Check for balance
    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"  Real samples: {real_count}")
    print(f"  Fake samples: {fake_count}")
    
    # First split: separate out test set
    # stratify=df["label"] means: keep the same real/fake ratio in each split
    df_train_val, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]    # important! keeps class balance
    )
    
    # Second split: separate train from validation
    # The val_size is relative to the remaining data after removing test
    actual_val_size = VAL_SIZE / (1 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=actual_val_size,
        random_state=SEED,
        stratify=df_train_val["label"]
    )
    
    print(f"\n  Split summary:")
    print(f"    Train: {len(df_train)} samples  ({len(df_train)/len(df)*100:.0f}%)")
    print(f"    Val:   {len(df_val)} samples  ({len(df_val)/len(df)*100:.0f}%)")
    print(f"    Test:  {len(df_test)} samples  ({len(df_test)/len(df)*100:.0f}%)")
    
    # ── Step 4: Save CSV files ────────────────────────────────────────────
    print("\n[4/4] Saving CSV manifest files...")
    
    train_csv = DATASET_DIR / "train.csv"
    val_csv   = DATASET_DIR / "val.csv"
    test_csv  = DATASET_DIR / "test.csv"
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,     index=False)
    df_test.to_csv(test_csv,   index=False)
    
    print(f"  ✓ Saved: {train_csv}")
    print(f"  ✓ Saved: {val_csv}")
    print(f"  ✓ Saved: {test_csv}")
    
    # Show a preview of what the CSV looks like
    print(f"\n  Preview of train.csv (first 3 rows):")
    print(df_train.head(3).to_string(index=False))
    
    print("\n" + "=" * 55)
    print("  Preprocessing complete! You're ready for Day 2.")
    print("=" * 55)


if __name__ == "__main__":
    main()