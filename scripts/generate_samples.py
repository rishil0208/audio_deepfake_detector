"""
generate_samples.py

Creates 4 demo audio clips for the live demo:
  assets/demo_clips/real_01.wav  → real-sounding speech
  assets/demo_clips/real_02.wav  → different real-sounding pattern
  assets/demo_clips/fake_01.wav  → AI-sounding speech (GAN artifacts)
  assets/demo_clips/fake_02.wav  → different fake pattern

These are synthesised mathematically — they don't need to sound like
real speech. They just need to produce DIFFERENT spectrograms so the
model predicts correctly, making the live demo impressive.

Run with:
    python scripts/generate_samples.py
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR  = PROJECT_ROOT / "assets" / "demo_clips"
SR        = 16000
DURATION  = 3.0
N_SAMPLES = int(SR * DURATION)


def generate_real_sample_1() -> np.ndarray:
    """
    Simulates a mid-pitched voice with natural formant structure.
    Fundamental ~150Hz, irregular vibrato, natural dynamics.
    """
    t = np.linspace(0, DURATION, N_SAMPLES)
    audio = np.zeros(N_SAMPLES)

    fundamental = 150.0
    for h in range(1, 10):
        amplitude = 0.8 / h
        wobble = fundamental * h + 3 * np.sin(2 * np.pi * 4.5 * t + np.random.rand())
        audio += amplitude * np.sin(2 * np.pi * wobble * t + np.random.rand() * 2 * np.pi)

    # Natural amplitude envelope — syllable-like dynamics
    env = np.ones(N_SAMPLES)
    for _ in range(6):
        s = np.random.randint(2000, N_SAMPLES - 5000)
        l = np.random.randint(1000, 4000)
        env[s:s+l] *= np.random.uniform(0.05, 0.3)
    audio *= env

    # Background noise — real recordings always have this
    audio += np.random.randn(N_SAMPLES) * 0.025

    audio /= np.abs(audio).max() + 1e-8
    audio *= 0.88
    return audio.astype(np.float32)


def generate_real_sample_2() -> np.ndarray:
    """
    Higher-pitched voice pattern with more consonant-like bursts.
    """
    t = np.linspace(0, DURATION, N_SAMPLES)
    audio = np.zeros(N_SAMPLES)

    fundamental = 210.0
    for h in range(1, 8):
        amplitude = 0.7 / h**0.9
        wobble = fundamental * h + 5 * np.sin(2 * np.pi * 5.2 * t)
        audio += amplitude * np.sin(2 * np.pi * wobble * t)

    # Burst-like energy — consonants
    for _ in range(8):
        burst_start = np.random.randint(0, N_SAMPLES - 2000)
        burst_len   = np.random.randint(200, 1500)
        audio[burst_start:burst_start + burst_len] += (
            np.random.randn(burst_len) * 0.15
        )

    audio += np.random.randn(N_SAMPLES) * 0.02
    audio /= np.abs(audio).max() + 1e-8
    audio *= 0.88
    return audio.astype(np.float32)


def generate_fake_sample_1() -> np.ndarray:
    """
    GAN-style voice: too-perfect harmonics, frame boundary clicks,
    unnaturally flat amplitude.
    """
    t = np.linspace(0, DURATION, N_SAMPLES)
    audio = np.zeros(N_SAMPLES)

    fundamental = 160.0
    for h in range(1, 7):
        amplitude = 0.75 / h   # no randomness — too perfect
        audio += amplitude * np.sin(2 * np.pi * fundamental * h * t)

    # Frame-boundary artifacts at every 256 samples
    FRAME = 256
    for i in range(0, N_SAMPLES, FRAME):
        end = min(i + FRAME, N_SAMPLES)
        if i > 0:
            audio[i:i+3] += np.random.choice([-0.08, 0.08])
        audio[i:end] *= 0.92   # slightly unnatural level

    # Very little noise — GAN doesn't model background well
    audio += np.random.randn(N_SAMPLES) * 0.005

    audio = np.clip(audio, -0.82, 0.82)
    audio /= np.abs(audio).max() + 1e-8
    audio *= 0.88
    return audio.astype(np.float32)


def generate_fake_sample_2() -> np.ndarray:
    """
    Different GAN artifact pattern: horizontal stripe artifacts and
    spectral smearing typical of MelGAN vocoders.
    """
    t = np.linspace(0, DURATION, N_SAMPLES)
    audio = np.zeros(N_SAMPLES)

    fundamental = 130.0
    # Suspiciously regular harmonics
    for h in range(1, 6):
        audio += (0.8 / h) * np.sin(2 * np.pi * fundamental * h * t)

    # Horizontal stripe artifact: add a constant low-energy sinusoid
    # at a very specific frequency (generator filter artefact)
    audio += 0.04 * np.sin(2 * np.pi * 3200 * t)   # 3.2kHz stripe
    audio += 0.03 * np.sin(2 * np.pi * 6400 * t)   # 6.4kHz stripe

    # Flat energy — no natural dynamics
    audio *= 0.85

    # Minimal noise — sharp high-freq cutoff
    audio += np.random.randn(N_SAMPLES) * 0.006

    audio /= np.abs(audio).max() + 1e-8
    audio *= 0.88
    return audio.astype(np.float32)


def main():
    print("═" * 50)
    print("  Generating demo audio clips")
    print("═" * 50)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    clips = [
        ("real_01.wav", generate_real_sample_1, "Real voice #1"),
        ("real_02.wav", generate_real_sample_2, "Real voice #2"),
        ("fake_01.wav", generate_fake_sample_1, "AI voice #1 (GAN artifacts)"),
        ("fake_02.wav", generate_fake_sample_2, "AI voice #2 (MelGAN stripe artifacts)"),
    ]

    for filename, generator_fn, description in clips:
        save_path = DEMO_DIR / filename
        audio     = generator_fn()
        sf.write(save_path, audio, SR)
        print(f"  ✓ {description} → {save_path}")

    print(f"\n  4 demo clips saved to: {DEMO_DIR}/")
    print("\n── Quick test with baseline model ──")

    # Test predictions on the demo clips
    model_path = PROJECT_ROOT / "saved_models" / "baseline.pkl"
    if model_path.exists():
        from backend.ml.baseline_model import BaselineDetector
        detector = BaselineDetector()
        detector.load(str(model_path))

        for filename, _, description in clips:
            expected = "REAL" if "real" in filename else "FAKE"
            result   = detector.predict(str(DEMO_DIR / filename))
            status   = "✓" if result['label'] == expected else "✗"
            print(f"  {status} {description}: {result['label']} ({result['confidence']}%)")
    else:
        print("  (Skipping predictions — train_baseline.py not run yet)")

    print("\n✓ Demo clips ready!")


if __name__ == "__main__":
    main()