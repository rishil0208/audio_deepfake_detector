"""
predict.py

POST /predict
  Accepts: multipart/form-data with an audio file field named 'file'
  Returns: JSON prediction result

POST /predict/demo
  Accepts: JSON body with {"sample_id": "real_01" | "real_02" | "fake_01" | "fake_02"}
  Returns: Same JSON prediction result using preloaded demo clips

Why do we need file format handling?
  Users might upload .mp3, .m4a, .flac, .ogg, .wav — any audio format.
  Our model only works on .wav files at 16kHz.
  pydub handles the conversion automatically.

Why do we save to a temp file?
  FastAPI gives us an UploadFile object (a stream).
  Our inference engine needs a file path on disk.
  So we: read stream → save to /tmp/ → run inference → delete temp file.

Error handling strategy:
  - Return 400 for bad requests (wrong file type, too short, corrupted)
  - Return 422 for missing required fields
  - Return 500 for unexpected server errors
  - NEVER let an unhandled exception crash the server
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT   = Path(__file__).parent.parent.parent
DEMO_CLIPS_DIR = PROJECT_ROOT / "assets" / "demo_clips"

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB in bytes

# Minimum audio duration we can process (in seconds)
MIN_DURATION = 0.5


class DemoRequest(BaseModel):
    """
    Pydantic model for the /predict/demo request body.

    Pydantic automatically validates the JSON body.
    If 'sample_id' is missing or wrong type, FastAPI returns a
    422 error automatically — you don't write any validation code.
    """
    sample_id: str   # "real_01", "real_02", "fake_01", or "fake_02"


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Converts any audio format to WAV using pydub.

    pydub uses ffmpeg under the hood — it supports virtually every
    audio format ever created.

    Parameters:
        input_path:  path to the original file (.mp3, .flac, etc.)
        output_path: where to save the converted .wav file

    Returns:
        True if conversion succeeded, False if it failed
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"  Audio conversion failed: {e}")
        return False


def validate_audio_file(file: UploadFile) -> tuple:
    """
    Checks that the uploaded file is a valid audio file we can process.

    Returns:
        (is_valid: bool, error_message: str)
    """
    # Check file extension
    if not file.filename:
        return False, "No filename provided"

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"File type '{ext}' not supported. "
            f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    return True, ""


async def save_upload_to_temp(file: UploadFile) -> str:
    """
    Reads an uploaded file stream and saves it to a temporary file.

    Returns the path to the saved temp file.
    The caller is responsible for deleting it after use.

    Why tempfile?
    We can't save user uploads permanently — disk would fill up.
    tempfile creates files in the OS temp directory (/tmp on Mac/Linux,
    %TEMP% on Windows) which gets cleaned up automatically.
    """
    # Get the file extension to preserve it in the temp file name
    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"

    # Create a temporary file with delete=False so we can open it later
    # delete=False means it stays on disk until we manually delete it
    tmp_file = tempfile.NamedTemporaryFile(
        suffix = suffix,
        delete = False
    )
    tmp_path = tmp_file.name

    try:
        # Read the upload in chunks to handle large files
        # without loading the entire thing into RAM at once
        CHUNK_SIZE  = 8192   # 8KB per chunk
        total_bytes = 0

        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break   # end of file
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE:
                tmp_file.close()
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            tmp_file.write(chunk)

        tmp_file.close()
        return tmp_path

    except HTTPException:
        raise
    except Exception as e:
        tmp_file.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")


@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Main prediction endpoint.

    Accepts an audio file upload and returns a prediction.

    The '= File(...)' syntax tells FastAPI:
    - Expect a file in the multipart form data
    - The field name in the form is 'file'
    - The '...' means it is required (not optional)

    Steps:
    1. Validate the file extension
    2. Save to a temp file
    3. Convert to WAV if needed
    4. Run inference
    5. Clean up temp files
    6. Return JSON result
    """
    tmp_path     = None
    wav_tmp_path = None

    try:
        # ── Step 1: Validate ───────────────────────────────────────────────
        is_valid, error_msg = validate_audio_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # ── Step 2: Save upload to temp file ───────────────────────────────
        tmp_path = await save_upload_to_temp(file)

        # ── Step 3: Convert to WAV if needed ───────────────────────────────
        file_ext = Path(file.filename).suffix.lower()

        if file_ext == ".wav":
            # Already WAV — use directly
            audio_path = tmp_path
        else:
            # Convert to WAV first
            wav_tmp_path = tmp_path.replace(file_ext, ".wav")
            success      = convert_to_wav(tmp_path, wav_tmp_path)

            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not decode the audio file. "
                        f"Please ensure ffmpeg is installed for non-WAV formats. "
                        f"Accepted without ffmpeg: .wav"
                    )
                )
            audio_path = wav_tmp_path

        # ── Step 4: Run inference ──────────────────────────────────────────
        from backend.ml.inference import engine

        if not engine.cnn_loaded and not engine.baseline_loaded:
            raise HTTPException(
                status_code=503,
                detail="No models are loaded. Train the models first."
            )

        result = engine.predict(audio_path)

        # ── Step 5: Clean up temp files ────────────────────────────────────
        # Always runs even if an exception was raised above
        # (the finally block below handles cleanup)

        # ── Step 6: Build and return response ─────────────────────────────
        return JSONResponse(content={
            "prediction":      result["prediction"],          # "REAL" or "FAKE"
            "confidence":      result["confidence"],          # 0.0 – 100.0
            "model_used":      result["model_used"],          # "CNN" or "baseline"
            "spectrogram_b64": result["spectrogram_b64"],     # base64 PNG
            "waveform_data":   result["waveform_data"],       # 200 float values
            "processing_ms":   result["processing_ms"],       # milliseconds
            "filename":        file.filename or "unknown"
        })

    except HTTPException:
        # Re-raise HTTP exceptions (our own errors) — don't swallow them
        raise

    except Exception as e:
        # Catch all unexpected errors so the server never crashes
        print(f"  Unexpected error in /predict: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {str(e)}"
        )

    finally:
        # ── Always clean up temp files ─────────────────────────────────────
        # This block runs whether the try block succeeded or raised an error
        for path in [tmp_path, wav_tmp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass   # non-critical if cleanup fails


@router.post("/predict/demo")
async def predict_demo(request: DemoRequest):
    """
    Runs prediction on one of the preloaded demo clips.

    Used by the frontend's "Try a demo" buttons.
    Much faster than uploading because the file is already on disk.

    Valid sample_ids: real_01, real_02, fake_01, fake_02
    """
    # Map sample_id to filename
    valid_ids = {
        "real_01": "real_01.wav",
        "real_02": "real_02.wav",
        "fake_01": "fake_01.wav",
        "fake_02": "fake_02.wav"
    }

    sample_id = request.sample_id.lower().strip()
    if sample_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown sample_id: '{request.sample_id}'. "
                f"Valid options: {list(valid_ids.keys())}"
            )
        )

    audio_path = DEMO_CLIPS_DIR / valid_ids[sample_id]

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Demo clip not found: {audio_path.name}. "
                f"Run: python scripts/generate_samples.py"
            )
        )

    try:
        from backend.ml.inference import engine
        result = engine.predict(str(audio_path))

        return JSONResponse(content={
            "prediction":      result["prediction"],
            "confidence":      result["confidence"],
            "model_used":      result["model_used"],
            "spectrogram_b64": result["spectrogram_b64"],
            "waveform_data":   result["waveform_data"],
            "processing_ms":   result["processing_ms"],
            "filename":        valid_ids[sample_id],
            "is_demo":         True,
            "expected_label":  "REAL" if "real" in sample_id else "FAKE"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Demo prediction failed: {str(e)}"
        )