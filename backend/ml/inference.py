"""
inference.py

The single entry point for all predictions.
Called by the FastAPI backend (/predict endpoint) in Day 4.

Loads both models once at startup (not on every request).
Runs predictions using the CNN by default, baseline as fallback.

Usage:
    engine = InferenceEngine()
    engine.load_models()
    result = engine.predict("path/to/audio.wav")
    # result = {
    #   "prediction":       "FAKE",
    #   "confidence":       92.3,
    #   "model_used":       "CNN",
    #   "spectrogram_b64":  "<base64 PNG string>",
    #   "waveform_data":    [0.1, 0.2, -0.1, ...],   # 200 points
    #   "processing_ms":    143
    # }
"""

import sys
import time
import base64
import io
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.cnn_model         import AudioCNN
from backend.ml.baseline_model    import BaselineDetector
from backend.ml.feature_extractor import extract_mel_spectrogram
from backend.utils.audio_preprocessor import load_and_preprocess


class InferenceEngine:
    """
    Manages both models and runs predictions.

    Keeps models loaded in memory so every API request doesn't
    have to reload them from disk (that would be very slow).
    """

    def __init__(self):
        self.cnn_model        = None
        self.baseline_model   = None
        self.device           = self._get_device()
        self.cnn_loaded       = False
        self.baseline_loaded  = False

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def load_models(self, cnn_path: str = None, baseline_path: str = None):
        """
        Loads both models from disk.
        Call this once when the FastAPI server starts.
        """
        cnn_path      = cnn_path      or str(PROJECT_ROOT / "saved_models" / "best_cnn.pt")
        baseline_path = baseline_path or str(PROJECT_ROOT / "saved_models" / "baseline.pkl")

        # ── Load CNN ───────────────────────────────────────────────────────
        if Path(cnn_path).exists():
            try:
                self.cnn_model = AudioCNN().to(self.device)
                checkpoint = torch.load(cnn_path, map_location=self.device)
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.cnn_model.eval()
                self.cnn_loaded = True
                print(f"  ✓ CNN loaded from {cnn_path}")
            except Exception as e:
                print(f"  ✗ CNN load failed: {e}")
        else:
            print(f"  ⚠  CNN model not found at {cnn_path}")
            print(f"     Run: python scripts/train_cnn.py")

        # ── Load Baseline ──────────────────────────────────────────────────
        if Path(baseline_path).exists():
            try:
                self.baseline_model = BaselineDetector()
                self.baseline_model.load(baseline_path)
                self.baseline_loaded = True
            except Exception as e:
                print(f"  ✗ Baseline load failed: {e}")
        else:
            print(f"  ⚠  Baseline model not found at {baseline_path}")

        if not self.cnn_loaded and not self.baseline_loaded:
            raise RuntimeError(
                "Neither model could be loaded. "
                "Run train_baseline.py and train_cnn.py first."
            )

    def predict(self, audio_path: str) -> dict:
        """
        Runs prediction on an audio file.
        Prefers CNN, falls back to baseline if CNN not loaded.

        Returns complete JSON-ready dict for the API response.
        """
        t_start = time.time()

        # ── Load + preprocess audio ────────────────────────────────────────
        try:
            audio = load_and_preprocess(str(audio_path))
        except Exception as e:
            raise ValueError(f"Could not process audio: {e}")

        # ── Choose model and predict ───────────────────────────────────────
        if self.cnn_loaded:
            label, confidence, model_used = self._predict_cnn(audio)
        elif self.baseline_loaded:
            result     = self.baseline_model.predict(str(audio_path))
            label      = result['label']
            confidence = result['confidence']
            model_used = 'baseline'
        else:
            raise RuntimeError("No models loaded")

        # ── Generate spectrogram base64 image ─────────────────────────────
        # The frontend displays this as an <img src="data:image/png;base64,...">
        mel      = extract_mel_spectrogram(audio)
        spec_b64 = self._spectrogram_to_base64(mel)

        # ── Downsample waveform to 200 points for frontend visualisation ───
        waveform_200 = self._downsample_waveform(audio, n_points=200)

        processing_ms = int((time.time() - t_start) * 1000)

        return {
            "prediction":      label,               # "REAL" or "FAKE"
            "confidence":      round(confidence, 1), # 0–100
            "model_used":      model_used,
            "spectrogram_b64": spec_b64,            # base64 PNG string
            "waveform_data":   waveform_200,        # list of 200 floats
            "processing_ms":   processing_ms
        }

    def _predict_cnn(self, audio: np.ndarray) -> tuple:
        """Runs the CNN on a preprocessed audio array."""
        mel    = extract_mel_spectrogram(audio)
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        # shape: (1, 1, 128, 128) — batch=1, channel=1, H=128, W=128

        self.cnn_model.eval()
        with torch.no_grad():
            prob_fake = float(self.cnn_model(tensor).cpu().numpy()[0][0])

        label      = "FAKE" if prob_fake > 0.5 else "REAL"
        confidence = max(prob_fake, 1 - prob_fake) * 100
        return label, confidence, "CNN"

    def _spectrogram_to_base64(self, mel: np.ndarray) -> str:
        """Converts a mel spectrogram array to a base64-encoded PNG string."""
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode('utf-8')

    def _downsample_waveform(self, audio: np.ndarray, n_points: int = 200) -> list:
        """Reduces 48000 audio samples to n_points for frontend chart."""
        indices    = np.linspace(0, len(audio) - 1, n_points).astype(int)
        downsampled = audio[indices]
        return [round(float(v), 4) for v in downsampled]

    @property
    def status(self) -> dict:
        """Returns model load status — used by /health endpoint."""
        return {
            "cnn_loaded":      self.cnn_loaded,
            "baseline_loaded": self.baseline_loaded,
            "device":          str(self.device)
        }


# ── Singleton: one engine instance shared across all API requests ─────────────
# Day 4's FastAPI app.py will import this and call engine.load_models() at startup
engine = InferenceEngine()


# ── Test when run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    print("Testing InferenceEngine...")
    engine.load_models()

    print(f"\nModel status: {engine.status}")

    # Test on demo clips if available
    demo_clips = glob.glob(str(PROJECT_ROOT / "assets" / "demo_clips" / "*.wav"))
    if demo_clips:
        print(f"\nTesting on {len(demo_clips)} demo clips:")
        for clip in sorted(demo_clips):
            expected   = "REAL" if "real" in Path(clip).stem else "FAKE"
            result     = engine.predict(clip)
            status     = "✓" if result['prediction'] == expected else "✗"
            print(f"  {status} {Path(clip).name}:")
            print(f"     Predicted: {result['prediction']} "
                  f"({result['confidence']}% confidence) "
                  f"via {result['model_used']} "
                  f"in {result['processing_ms']}ms")
    else:
        print("No demo clips found. Run generate_samples.py first.")