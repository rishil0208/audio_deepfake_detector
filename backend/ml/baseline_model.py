"""
baseline_model.py

The RandomForest-based baseline detector.

Why a baseline model at all?
  The CNN (Day 3) is your main model, but it takes time to train and is
  harder to explain. The baseline gives you:
  1. A working predictor in < 5 minutes of training
  2. A performance comparison: "CNN achieves X%, vs baseline Y%"
  3. Proof that the problem is solvable with simple features

Architecture:
  Audio file
    → load_and_preprocess()       (Day 1 code)
    → extract_all_features()      (Day 2 code)
    → RandomForestClassifier      (sklearn)
    → prediction: REAL or FAKE + confidence %
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Add project root to path so we can import our modules ─────────────────────
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.audio_preprocessor import load_and_preprocess
from backend.ml.feature_extractor import extract_all_features


class BaselineDetector:
    """
    A complete audio deepfake detector using hand-crafted features + RandomForest.

    Usage:
        detector = BaselineDetector()

        # Training:
        detector.train("datasets/train.csv", "saved_models/baseline.pkl")

        # Prediction:
        result = detector.predict("path/to/audio.wav")
        # Returns: {"label": "FAKE", "confidence": 87.3, "model": "baseline"}

        # Loading a saved model:
        detector.load("saved_models/baseline.pkl")
    """

    def __init__(self):
        # The classifier: 200 decision trees, working together
        # n_estimators=200 → 200 trees (more = better, but slower)
        # max_depth=20      → each tree can ask 20 questions max
        # class_weight='balanced' → handles unequal real/fake counts automatically
        # n_jobs=-1         → use ALL CPU cores to train in parallel
        # random_state=42   → same random seed = reproducible results
        self.model = RandomForestClassifier(
            n_estimators  = 200,
            max_depth     = 20,
            class_weight  = 'balanced',
            n_jobs        = -1,
            random_state  = 42,
            min_samples_leaf = 2    # each tree leaf needs at least 2 samples
        )

        # StandardScaler normalises features to mean=0, std=1
        # Why? RandomForest doesn't strictly need this, but it helps
        # when you later compare with the CNN output in combined models
        self.scaler = StandardScaler()

        self.is_trained = False
        self.feature_dim = 84   # 80 MFCC + 4 waveform stats

    # ── Feature building ───────────────────────────────────────────────────────

    def build_features(self, audio_path: str) -> np.ndarray:
        """
        Loads an audio file and returns its 84-dimensional feature vector.

        This is the complete pipeline for a SINGLE file:
          .wav/.mp3/.npy → preprocess → extract_features → (84,) array

        Parameters:
            audio_path: path to audio file (any format) OR .npy preprocessed file

        Returns:
            features: numpy array of shape (84,)
        """
        # Handle both .npy (already preprocessed) and raw audio files
        if str(audio_path).endswith('.npy'):
            audio = np.load(audio_path)
        else:
            audio = load_and_preprocess(str(audio_path))

        features = extract_all_features(audio)
        return features

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, train_csv_path: str, save_path: str, show_progress: bool = True):
        """
        Trains the RandomForest on all files in the training CSV.

        Steps:
        1. Read train.csv → list of (filepath, label) pairs
        2. For each file: load audio → extract 84 features
        3. Build feature matrix X: shape (N_samples, 84)
        4. Build label vector y:  shape (N_samples,) — 0 or 1
        5. Fit scaler on X (learn mean/std of each feature)
        6. Scale X → X_scaled
        7. Fit RandomForest on (X_scaled, y)
        8. Save model + scaler to disk

        Parameters:
            train_csv_path: path to datasets/train.csv
            save_path:      where to save the trained model (e.g. saved_models/baseline.pkl)
        """
        import pandas as pd
        from tqdm import tqdm

        print("═" * 50)
        print("  Training Baseline Model (RandomForest)")
        print("═" * 50)

        # ── Step 1: Read CSV ────────────────────────────────────────────────
        df = pd.read_csv(train_csv_path)
        print(f"\n  Training samples:  {len(df)}")
        print(f"  Real (label=0):    {(df['label']==0).sum()}")
        print(f"  Fake (label=1):    {(df['label']==1).sum()}")

        # ── Step 2 & 3: Extract features for every file ────────────────────
        print(f"\n  Extracting features ({self.feature_dim} per file)...")

        features_list = []    # will become the X matrix
        labels_list   = []    # will become the y vector
        failed        = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting"):
            try:
                # Build the absolute path (CSV stores relative paths)
                file_path = PROJECT_ROOT / row['filepath']
                features  = self.build_features(str(file_path))

                features_list.append(features)
                labels_list.append(int(row['label']))

            except Exception as e:
                failed += 1
                # Skip bad files silently (a few failures are normal)

        if failed > 0:
            print(f"  ⚠  Skipped {failed} files (could not load/process)")

        # ── Step 4: Build matrices ──────────────────────────────────────────
        X = np.array(features_list)   # shape: (N, 84)
        y = np.array(labels_list)     # shape: (N,)

        print(f"\n  Feature matrix shape: {X.shape}")
        print(f"  Labels shape:         {y.shape}")

        # Sanity check: are there NaN or Inf values?
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("  ⚠  Found NaN/Inf values — replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Step 5 & 6: Scale features ─────────────────────────────────────
        # Fit the scaler on training data ONLY (important!)
        # Then apply the same transformation to validation/test data
        print("\n  Fitting feature scaler...")
        X_scaled = self.scaler.fit_transform(X)

        # ── Step 7: Train the model ─────────────────────────────────────────
        print("  Training RandomForest (200 trees)...")
        print("  (This takes 30 seconds to 3 minutes depending on dataset size)")
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Quick training accuracy check
        train_acc = self.model.score(X_scaled, y)
        print(f"\n  Training accuracy: {train_acc*100:.1f}%")
        print("  (High training accuracy is expected — validation accuracy matters more)")

        # ── Step 8: Save model + scaler ─────────────────────────────────────
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'model':       self.model,
            'scaler':      self.scaler,
            'feature_dim': self.feature_dim
        }
        joblib.dump(save_data, save_path)
        print(f"\n  ✓ Model saved to: {save_path}")
        print("═" * 50)

    # ── Loading a saved model ──────────────────────────────────────────────────

    def load(self, model_path: str):
        """
        Loads a previously trained model from disk.

        Call this before calling predict() if you haven't called train() yet.

        Parameters:
            model_path: path to the .pkl file (e.g. saved_models/baseline.pkl)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Run train_baseline.py first to create it."
            )

        save_data    = joblib.load(model_path)
        self.model   = save_data['model']
        self.scaler  = save_data['scaler']
        self.is_trained = True
        print(f"  ✓ Baseline model loaded from: {model_path}")

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> dict:
        """
        Predicts whether an audio clip is REAL or FAKE.

        Returns a dictionary with:
          label:      "REAL" or "FAKE"
          confidence: 0–100 float — how confident the model is
          model:      "baseline" — which model made this prediction
          raw_prob:   [prob_real, prob_fake] — raw probabilities

        Parameters:
            audio_path: path to any audio file or .npy preprocessed file
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )

        # Extract features
        features = self.build_features(audio_path)

        # Reshape to (1, 84) — the model expects a 2D matrix, not a 1D vector
        features_2d = features.reshape(1, -1)

        # Apply the same scaling as during training
        features_scaled = self.scaler.transform(features_2d)

        # Get probabilities for each class
        # predict_proba returns [[prob_real, prob_fake]]
        probabilities = self.model.predict_proba(features_scaled)[0]
        # probabilities[0] = P(real), probabilities[1] = P(fake)

        prob_fake = float(probabilities[1])
        prob_real = float(probabilities[0])

        # Decide label: fake if P(fake) > 0.5
        label = "FAKE" if prob_fake > 0.5 else "REAL"
        confidence = max(prob_fake, prob_real) * 100  # convert to percentage

        return {
            "label":      label,
            "confidence": round(confidence, 1),
            "model":      "baseline",
            "raw_prob":   {
                "real": round(prob_real * 100, 1),
                "fake": round(prob_fake * 100, 1)
            }
        }