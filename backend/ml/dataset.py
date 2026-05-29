"""
dataset.py

PyTorch Dataset class for the audio deepfake CNN.

A Dataset is like a smart list that:
  - Knows how many items it has (__len__)
  - Can return any single item on request (__getitem__)

The DataLoader wraps the Dataset and:
  - Shuffles the order each epoch (prevents the model learning order)
  - Groups items into batches (e.g. 32 at a time)
  - Loads items in parallel (num_workers) for speed

Data augmentation:
  During TRAINING we randomly modify spectrograms so the model
  sees slightly different versions each epoch. This prevents
  overfitting (memorising training data).

  SpecAugment techniques used:
    - Time masking:      blank out a vertical strip (a time window)
    - Frequency masking: blank out a horizontal strip (a freq band)
    - Gaussian noise:    add tiny random values to all pixels
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.feature_extractor import extract_mel_spectrogram
from backend.utils.audio_preprocessor import load_and_preprocess


class AudioSpectrogramDataset(Dataset):
    """
    Loads audio files from a CSV manifest and returns
    mel spectrogram tensors + labels.

    Parameters:
        csv_path (str):  path to train.csv / val.csv / test.csv
        augment (bool):  if True, apply random augmentations
                         → always True for training, False for val/test

    Each CSV row has:
        filepath: path to .npy preprocessed audio (relative to project root)
        label:    0 = real, 1 = fake
    """

    def __init__(self, csv_path: str, augment: bool = False):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment

        # Count classes
        n_real = (self.df['label'] == 0).sum()
        n_fake = (self.df['label'] == 1).sum()
        print(f"  Dataset loaded: {len(self.df)} samples "
              f"({n_real} real, {n_fake} fake), augment={augment}")

    def __len__(self) -> int:
        """How many samples are in this dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns the idx-th sample as (spectrogram_tensor, label_tensor).

        Called by DataLoader internally. You never call this directly.

        Returns:
            tensor: torch.FloatTensor of shape (1, 128, 128)
                    → 1 channel (grayscale), 128×128 pixels
            label:  torch.FloatTensor of shape (1,)
                    → [0.0] for real, [1.0] for fake
        """
        row       = self.df.iloc[idx]
        label     = float(row['label'])

        # ── Load the preprocessed audio ────────────────────────────────────
        file_path = PROJECT_ROOT / row['filepath']

        try:
            # .npy files load ~50x faster than .wav files
            audio = np.load(str(file_path))
        except Exception:
            # Fallback: try loading as raw audio
            try:
                audio = load_and_preprocess(str(file_path))
            except Exception as e:
                # Return a silent spectrogram on failure (don't crash training)
                print(f"  Warning: could not load {file_path}: {e}")
                silent = np.zeros(128 * 128, dtype=np.float32).reshape(128, 128)
                return torch.FloatTensor(silent).unsqueeze(0), torch.FloatTensor([label])

        # ── Extract mel spectrogram ────────────────────────────────────────
        # Returns shape (128, 128), values 0.0–1.0
        mel = extract_mel_spectrogram(audio)

        # ── Data augmentation (training only) ─────────────────────────────
        if self.augment:
            mel = self._augment(mel)

        # ── Convert to PyTorch tensor ──────────────────────────────────────
        # unsqueeze(0) adds the channel dimension: (128, 128) → (1, 128, 128)
        # The CNN expects shape (batch, channels, height, width)
        tensor = torch.FloatTensor(mel).unsqueeze(0)

        # Label must also be a tensor with shape (1,) for BCELoss
        label_tensor = torch.FloatTensor([label])

        return tensor, label_tensor

    def _augment(self, mel: np.ndarray) -> np.ndarray:
        """
        Applies random SpecAugment-style transformations.

        Each transformation happens with a certain probability.
        On any given sample, 0, 1, 2, or all 3 might apply.

        Why augment?
          If the model always sees the exact same 560 spectrograms,
          it memorises them. Augmentation makes each epoch slightly
          different, forcing generalisation.

        Parameters:
            mel: numpy array (128, 128) — original spectrogram

        Returns:
            numpy array (128, 128) — augmented version
        """
        mel = mel.copy()   # don't modify the original

        # ── Time masking ──────────────────────────────────────────────────
        # Randomly blank out a vertical band (range of time frames)
        # Simulates the model not seeing part of the audio
        if np.random.random() < 0.4:
            t_mask_width = np.random.randint(5, 25)    # how wide to blank
            t_start      = np.random.randint(0, 128 - t_mask_width)
            mel[:, t_start : t_start + t_mask_width] = 0.0

        # ── Frequency masking ─────────────────────────────────────────────
        # Randomly blank out a horizontal band (range of frequency bins)
        # Simulates the model not seeing part of the spectrum
        if np.random.random() < 0.4:
            f_mask_width = np.random.randint(5, 25)
            f_start      = np.random.randint(0, 128 - f_mask_width)
            mel[f_start : f_start + f_mask_width, :] = 0.0

        # ── Gaussian noise ────────────────────────────────────────────────
        # Add a tiny amount of random noise to every pixel
        # Prevents the model from being overly sensitive to exact values
        if np.random.random() < 0.3:
            noise = np.random.randn(*mel.shape).astype(np.float32) * 0.015
            mel   = mel + noise
            mel   = np.clip(mel, 0.0, 1.0)   # keep values in valid range

        # ── Brightness jitter ─────────────────────────────────────────────
        # Multiply all values by a random scale factor
        # Simulates recording at different volumes
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.85, 1.15)
            mel   = np.clip(mel * scale, 0.0, 1.0)

        return mel


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    TRAIN_CSV = PROJECT_ROOT / "datasets" / "train.csv"
    if not TRAIN_CSV.exists():
        print("ERROR: Run preprocess.py first")
        sys.exit(1)

    print("Testing AudioSpectrogramDataset...")

    dataset = AudioSpectrogramDataset(str(TRAIN_CSV), augment=True)
    print(f"\nDataset size: {len(dataset)}")

    # Get one sample
    tensor, label = dataset[0]
    print(f"Spectrogram tensor shape: {tensor.shape}  (should be [1, 128, 128])")
    print(f"Label tensor:             {label}          (should be [0.] or [1.])")
    print(f"Tensor value range:       [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch_tensors, batch_labels = next(iter(loader))
    print(f"\nDataLoader batch shape:  {batch_tensors.shape}  (should be [8, 1, 128, 128])")
    print(f"DataLoader labels shape: {batch_labels.shape}   (should be [8, 1])")

    print("\n✓ Dataset working correctly!")