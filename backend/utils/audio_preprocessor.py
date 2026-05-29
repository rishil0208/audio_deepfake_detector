"""
audio_preprocessor.py

What this does:
Takes a raw .wav file and returns a clean numpy array, always:
  - Mono (single channel, not stereo)
  - Resampled to exactly 16,000 Hz
  - Silence trimmed from both ends
  - Amplitude normalised to [-1, 1]
  - Exactly 3 seconds long (padded with zeros if short, truncated if long)

Why do we need this?
Every ML model expects its input to be exactly the same shape and scale.
If file A is 2.3 seconds, file B is 5 seconds, and file C was recorded
at 44100Hz — the model can't handle that variety. We standardise everything.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SR  = 16000          # target sample rate in Hz
DURATION   = 3.0            # target duration in seconds
N_SAMPLES  = int(TARGET_SR * DURATION)   # = 48000 samples


def load_and_preprocess(file_path: str) -> np.ndarray:
    """
    Loads an audio file and returns a clean, standardised numpy array.
    
    Parameters:
        file_path: path to the audio file (.wav, .mp3, .flac, etc.)
    
    Returns:
        audio: numpy array of shape (48000,) — always exactly 48000 samples
               values between -1.0 and 1.0
    
    Raises:
        ValueError: if the file cannot be read or is completely silent
    """
    file_path = str(file_path)   # convert Path object to string if needed
    
    # ── Step 1: Load the audio ─────────────────────────────────────────────
    # librosa.load() reads an audio file and returns:
    #   audio: numpy array of audio samples
    #   sr:    the sample rate (how many samples per second)
    #
    # We tell librosa:
    #   sr=TARGET_SR → resample to 16000 Hz automatically
    #   mono=True    → convert stereo to single channel by averaging
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio file '{file_path}': {e}")
    
    # At this point: audio is a 1D numpy array at 16kHz
    # Length varies — we'll fix that in later steps
    
    # ── Step 2: Trim silence ───────────────────────────────────────────────
    # Many recordings have silence at the beginning and end
    # (person waiting before speaking, recording after they finish)
    # We trim it so the model sees actual speech content
    #
    # top_db=20 means: trim anything quieter than 20dB below the peak
    # (i.e. cut off the quiet parts at the edges)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # If trimming left nothing (completely silent file), use original
    if len(audio) == 0:
        raise ValueError(f"Audio file '{file_path}' is completely silent")
    
    # ── Step 3: Normalise amplitude ────────────────────────────────────────
    # Different recordings have different volumes.
    # A quiet recording might have max amplitude 0.1
    # A loud recording might have max amplitude 0.9
    # 
    # We normalise so the loudest point is always 0.9
    # This ensures the model sees consistent signal levels
    max_amplitude = np.abs(audio).max()
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.9
    
    # ── Step 4: Fix length to exactly N_SAMPLES ────────────────────────────
    # All inputs must be the same length for our CNN.
    # If the audio is too long: cut it from the middle
    #   (the middle of speech is usually more informative than the edges)
    # If the audio is too short: pad it with zeros
    #   (adding silence doesn't hurt the model)
    
    if len(audio) > N_SAMPLES:
        # Take from the centre of the audio
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start : start + N_SAMPLES]
    
    elif len(audio) < N_SAMPLES:
        # Pad with zeros at the end
        padding = N_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
    
    # audio is now exactly N_SAMPLES long
    return audio.astype(np.float32)


def preprocess_and_save(input_path: str, output_path: str) -> bool:
    """
    Preprocesses an audio file and saves the result as a .npy file.
    
    .npy is numpy's own binary format — much faster to load than .wav
    because it skips all the audio decoding. Perfect for training loops
    that load thousands of files repeatedly.
    
    Parameters:
        input_path:  path to the original .wav file
        output_path: where to save the .npy processed array
    
    Returns:
        True if successful, False if the file had an error
    """
    try:
        audio = load_and_preprocess(input_path)
        np.save(output_path, audio)
        return True
    except Exception as e:
        print(f"  ✗ Skipping {Path(input_path).name}: {e}")
        return False


def verify_preprocessing(audio: np.ndarray) -> dict:
    """
    Runs sanity checks on a preprocessed audio array.
    Returns a dict with the results — useful for debugging.
    """
    return {
        "length":          len(audio),
        "expected_length": N_SAMPLES,
        "length_ok":       len(audio) == N_SAMPLES,
        "max_amplitude":   float(np.abs(audio).max()),
        "min_amplitude":   float(audio.min()),
        "mean_amplitude":  float(np.abs(audio).mean()),
        "is_silent":       float(np.abs(audio).max()) < 0.001,
        "dtype":           str(audio.dtype),
    }


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    If you run this file directly with:
        python backend/utils/audio_preprocessor.py
    
    It will test itself on any .wav file it finds in datasets/
    """
    import glob
    
    # Find any .wav file to test with
    wav_files = glob.glob("datasets/**/*.wav", recursive=True)
    
    if not wav_files:
        print("No .wav files found. Run download_dataset.py first.")
    else:
        test_file = wav_files[0]
        print(f"Testing with: {test_file}")
        
        audio = load_and_preprocess(test_file)
        info  = verify_preprocessing(audio)
        
        print(f"\nPreprocessing results:")
        for key, val in info.items():
            status = "✓" if (key != "is_silent" or not val) else "✗ PROBLEM"
            print(f"  {status} {key}: {val}")