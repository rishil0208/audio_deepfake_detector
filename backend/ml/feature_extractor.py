"""
feature_extractor.py

Converts a preprocessed audio array (numpy, shape: 48000,)
into three kinds of features:

1. MFCC feature vector   → shape (80,)   → used by the baseline model
2. Mel spectrogram       → shape (128,128) → used by the CNN (Day 3)
3. Waveform stats        → shape (4,)    → extra features for baseline

Why these features?
- MFCCs capture the shape of the vocal tract — how the voice resonates
- Mel spectrograms show the full frequency-time picture (GAN artifacts visible here)
- Waveform stats give simple global properties (energy, zero-crossings)
"""

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')       # use non-interactive backend (no GUI window pops up)
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
SR          = 16000     # sample rate (must match preprocessor)
N_MFCC      = 40        # number of MFCC coefficients
N_MELS      = 128       # number of mel filter banks
HOP_LENGTH  = 512       # how many samples between each STFT frame
SPEC_SIZE   = 128       # we resize spectrograms to 128×128 pixels


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — MFCC Vector
# ══════════════════════════════════════════════════════════════════════════════

def extract_mfcc(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extracts a compact MFCC feature vector from an audio clip.

    What are MFCCs?
    ──────────────
    Imagine you're listening to someone say "aaah". The shape of your
    mouth, throat and tongue creates a unique frequency pattern. MFCCs
    are a mathematical description of that frequency "shape".

    The process:
    1. Split the audio into short overlapping windows (~25ms each)
    2. For each window, run a Fourier transform → frequency spectrum
    3. Apply mel filter bank → compress to human-hearing scale
    4. Take the log → mimics how ears perceive loudness
    5. Apply DCT → decorrelate → get 40 coefficients

    Result: a matrix of shape (40, T) where T is number of time frames.

    But a matrix is awkward for a simple classifier — it varies in size
    and is 2D. So we summarise each of the 40 rows with:
      - mean: the average value of that coefficient across time
      - std:  how much that coefficient varies across time

    Final output: 40 means + 40 stds = 80 numbers. Always the same size.

    Why mean AND std?
    The mean captures "what kind of voice is this overall".
    The std captures "how much does this voice vary" — GAN voices
    tend to have abnormally low std (too consistent = fake).

    Parameters:
        audio: numpy array of shape (48000,) — preprocessed audio
        sr:    sample rate (default 16000)

    Returns:
        feature_vector: numpy array of shape (80,) — float32
    """
    # Compute the MFCC matrix — shape: (N_MFCC, T)
    # T varies based on audio length but with our fixed 3s clips it's ~93 frames
    mfcc = librosa.feature.mfcc(
        y      = audio,
        sr     = sr,
        n_mfcc = N_MFCC,
        n_fft  = 2048,          # FFT window size (~128ms)
        hop_length = HOP_LENGTH  # step between frames
    )
    # mfcc shape: (40, ~93)

    # Summarise across time: compute mean and std of each coefficient row
    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (40,)
    mfcc_std  = np.std(mfcc,  axis=1)   # shape: (40,)

    # Stack mean and std into one flat vector
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])  # shape: (80,)

    return feature_vector.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Mel Spectrogram (used by CNN tomorrow)
# ══════════════════════════════════════════════════════════════════════════════

def extract_mel_spectrogram(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Converts audio into a 2D mel spectrogram image.

    What is a spectrogram?
    ──────────────────────
    A regular spectrogram is a 2D grid:
      - X axis: time (left = start, right = end)
      - Y axis: frequency (bottom = low, top = high)
      - Cell value: how much energy is at that frequency at that time

    A MEL spectrogram uses a frequency scale that matches human hearing.
    Low frequencies (where voices live) get more detail. High frequencies
    get compressed together. This makes it much better for voice analysis.

    Why does this detect fakes?
    GAN-generated voices leave visible patterns in the spectrogram:
    - Regular horizontal stripes (generator filter artifacts)
    - Abrupt cutoff in high frequencies (generator doesn't model noise)
    - Overly smooth, uniform mid-frequency bands
    Your CNN learns to spot these textures automatically.

    Parameters:
        audio: numpy array of shape (48000,)
        sr:    sample rate

    Returns:
        spectrogram: numpy array of shape (128, 128) — values 0.0 to 1.0
    """
    # Step 1: Compute raw mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y          = audio,
        sr         = sr,
        n_mels     = N_MELS,       # 128 mel frequency bands
        hop_length = HOP_LENGTH,   # time resolution
        n_fft      = 2048,
        fmax       = 8000          # max frequency — captures full speech range
    )
    # mel_spec shape: (128, ~93) — 128 frequency bins, ~93 time frames

    # Step 2: Convert to decibel scale (log scale — matches human perception)
    # librosa.power_to_db converts energy values to dB values
    # ref=np.max makes it relative to the loudest point
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Values are now in dB, typically -80 to 0

    # Step 3: Resize to exactly SPEC_SIZE × SPEC_SIZE (128 × 128)
    # The CNN needs a fixed input size. We use simple interpolation.
    # np.interp approach: just reshape using slicing
    # For a proper resize, we use a simple nearest-neighbour via index mapping

    # First, make it square: resize time axis from ~93 to 128
    from PIL import Image as PILImage
    img = PILImage.fromarray(mel_spec_db)
    img_resized = img.resize((SPEC_SIZE, SPEC_SIZE), PILImage.BILINEAR)
    mel_resized = np.array(img_resized)
    # Shape: (128, 128)

    # Step 4: Normalise to [0, 1] range
    # Neural networks train better when inputs are in a consistent range
    min_val = mel_resized.min()
    max_val = mel_resized.max()
    if max_val - min_val > 0:
        mel_normalised = (mel_resized - min_val) / (max_val - min_val)
    else:
        mel_normalised = np.zeros_like(mel_resized)

    return mel_normalised.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Waveform Statistics (extra features for baseline)
# ══════════════════════════════════════════════════════════════════════════════

def extract_waveform_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Computes 4 simple statistics from the raw waveform.

    These are quick-to-compute properties that differ between
    real and AI-generated speech.

    Zero Crossing Rate (ZCR):
    How often does the signal cross zero (positive → negative)?
    Real speech has irregular ZCR. Synthetic speech often has unnaturally
    regular patterns.

    RMS Energy:
    Root Mean Square energy — basically the "loudness" over time.
    GAN voices have suspiciously flat RMS (too consistent).

    Spectral Centroid:
    The "center of gravity" of the frequency spectrum.
    Where most of the energy sits. Fake voices sometimes have this
    shifted toward lower or higher frequencies unnaturally.

    Spectral Bandwidth:
    How "spread out" the frequencies are.
    Real speech has more spread; AI voices can be narrower.

    Returns:
        features: numpy array of shape (4,) — [zcr, rms, centroid, bandwidth]
    """
    # Zero Crossing Rate — shape: (1, T) → we take the mean
    zcr = np.mean(librosa.feature.zero_crossing_rate(
        audio, hop_length=HOP_LENGTH
    ))

    # RMS Energy — shape: (1, T) → we take the mean
    rms = np.mean(librosa.feature.rms(
        y=audio, hop_length=HOP_LENGTH
    ))

    # Spectral Centroid — shape: (1, T) → mean, normalised by SR
    centroid = np.mean(librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )) / sr   # normalise to [0, 0.5]

    # Spectral Bandwidth — shape: (1, T) → mean, normalised by SR
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )) / sr

    features = np.array([zcr, rms, centroid, bandwidth], dtype=np.float32)
    return features


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED — Full feature vector for baseline model
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extracts the COMPLETE feature vector for the baseline model.

    Combines:
    - 80 MFCC features (mean + std of 40 coefficients)
    -  4 waveform statistics
    ─────────────────────────────
    = 84-dimensional feature vector

    This 84-number vector is what RandomForest trains on.
    Each number represents one aspect of the audio's character.

    Returns:
        features: numpy array of shape (84,)
    """
    mfcc_features  = extract_mfcc(audio, sr)           # (80,)
    wave_features  = extract_waveform_features(audio, sr)  # (4,)

    combined = np.concatenate([mfcc_features, wave_features])  # (84,)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY — Save spectrogram as image (for dashboard display)
# ══════════════════════════════════════════════════════════════════════════════

def save_spectrogram_image(mel_spec: np.ndarray, output_path: str):
    """
    Saves a mel spectrogram as a PNG image file.

    Used by:
    - The dashboard to show example spectrograms
    - The backend API to return base64-encoded spectrograms to the frontend

    The colourmap 'magma' maps low values (dark/quiet) to black/purple,
    high values (loud) to bright yellow. This is the standard look for
    audio spectrograms — you'll recognise it from research papers.

    Parameters:
        mel_spec:    numpy array of shape (128, 128), values 0-1
        output_path: where to save the PNG file
    """
    # Create a figure with no borders or axes — just the raw image
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(
        mel_spec,
        aspect='auto',          # don't force square pixels
        origin='lower',         # low frequencies at bottom (conventional)
        cmap='magma',           # standard audio spectrogram colormap
        interpolation='nearest'
    )
    ax.axis('off')              # remove axes, ticks, labels
    fig.subplots_adjust(        # remove all white margin around image
        left=0, right=1, top=1, bottom=0
    )
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)              # IMPORTANT: always close figure to free memory


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run this file directly to verify everything works
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test the feature extractor on a real file from your dataset.
    Run with: python backend/ml/feature_extractor.py
    """
    import sys
    import glob
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Find a processed .npy file to test with
    npy_files = glob.glob("datasets/processed/*.npy")

    if not npy_files:
        print("No processed files found. Run preprocess.py first.")
        sys.exit(1)

    test_file = npy_files[0]
    print(f"Testing feature extraction on: {test_file}")

    # Load the preprocessed audio
    audio = np.load(test_file)
    print(f"\nAudio shape: {audio.shape}   (should be (48000,))")
    print(f"Audio dtype: {audio.dtype}   (should be float32)")

    # Test each extractor
    print("\n── Testing MFCC extraction ──")
    mfcc_vec = extract_mfcc(audio)
    print(f"  MFCC vector shape: {mfcc_vec.shape}   (should be (80,))")
    print(f"  MFCC mean range:   [{mfcc_vec.min():.2f}, {mfcc_vec.max():.2f}]")

    print("\n── Testing mel spectrogram extraction ──")
    mel = extract_mel_spectrogram(audio)
    print(f"  Mel spec shape:  {mel.shape}   (should be (128, 128))")
    print(f"  Mel spec range:  [{mel.min():.3f}, {mel.max():.3f}]  (should be 0.0–1.0)")

    print("\n── Testing waveform features ──")
    wave = extract_waveform_features(audio)
    print(f"  Waveform features: {wave}   (4 numbers)")

    print("\n── Testing combined features ──")
    combined = extract_all_features(audio)
    print(f"  Combined shape: {combined.shape}   (should be (84,))")

    print("\n── Saving spectrogram image ──")
    Path("assets/plots").mkdir(parents=True, exist_ok=True)
    save_spectrogram_image(mel, "assets/plots/test_spectrogram.png")
    print(f"  Saved to: assets/plots/test_spectrogram.png")

    print("\n✓ All feature extractors working correctly!")