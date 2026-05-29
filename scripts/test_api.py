"""
test_api.py

Tests all 4 API endpoints automatically.
Run this AFTER starting the server in a separate terminal.

Usage:
    Terminal 1: uvicorn backend.app:app --reload --port 8000
    Terminal 2: python scripts/test_api.py
"""

import sys
import time
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_URL     = "http://localhost:8000"

# Colours for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")


def test_root():
    print(f"\n{BOLD}── Test 1: Root endpoint GET /{RESET}")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "message" in data
        ok(f"Status 200, message: {data['message']}")
    except requests.ConnectionError:
        fail("Cannot connect to server. Is it running?")
        fail("Start it with: uvicorn backend.app:app --reload --port 8000")
        sys.exit(1)
    except AssertionError as e:
        fail(str(e))


def test_health():
    print(f"\n{BOLD}── Test 2: Health endpoint GET /health{RESET}")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()

        ok(f"Status: {data['status']}")
        ok(f"CNN loaded:      {data['models']['cnn_loaded']}")
        ok(f"Baseline loaded: {data['models']['baseline_loaded']}")
        ok(f"Device:          {data['models']['device']}")

        if not data['models']['cnn_loaded']:
            info("CNN not loaded — run: python scripts/train_cnn.py")
        if not data['models']['baseline_loaded']:
            info("Baseline not loaded — run: python scripts/train_baseline.py")

    except AssertionError as e:
        fail(str(e))
    except Exception as e:
        fail(f"Unexpected error: {e}")


def test_metrics():
    print(f"\n{BOLD}── Test 3: Metrics endpoint GET /metrics{RESET}")
    try:
        r = requests.get(f"{BASE_URL}/metrics", timeout=10)
        assert r.status_code == 200
        data = r.json()

        ok(f"Response received ({len(str(data))} chars)")

        # Check CNN history
        if data["cnn"]["available"]:
            ok(f"CNN epochs trained:  {data['cnn']['epochs_trained']}")
            ok(f"CNN best val acc:    {data['cnn']['best_val_acc']*100:.1f}%")
        else:
            info(f"CNN history not available: {data['cnn'].get('message','')}")

        # Check dataset stats
        ds = data["dataset"]
        ok(f"Dataset: {ds['total_samples']} total "
           f"({ds['real_count']} real, {ds['fake_count']} fake)")
        ok(f"Splits: train={ds['train_count']} "
           f"val={ds['val_count']} test={ds['test_count']}")

        # Check plot images
        plots_loaded = sum(1 for v in data["plots"].values() if v)
        ok(f"Plot images available: {plots_loaded}/4")

    except AssertionError as e:
        fail(str(e))
    except Exception as e:
        fail(f"Unexpected error: {e}")


def test_predict_demo():
    print(f"\n{BOLD}── Test 4: Demo prediction POST /predict/demo{RESET}")

    for sample_id, expected_label in [
        ("real_01", "REAL"),
        ("fake_01", "FAKE")
    ]:
        try:
            r = requests.post(
                f"{BASE_URL}/predict/demo",
                json    = {"sample_id": sample_id},
                timeout = 30
            )

            if r.status_code == 404:
                info(f"Demo clip not found — run: python scripts/generate_samples.py")
                continue

            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
            data = r.json()

            predicted  = data["prediction"]
            confidence = data["confidence"]
            model      = data["model_used"]
            ms         = data["processing_ms"]

            correct = predicted == expected_label
            status  = ok if correct else fail

            status(
                f"Sample '{sample_id}': predicted {predicted} "
                f"({confidence}% confidence) via {model} in {ms}ms"
            )

            # Check spectrogram was returned
            if data["spectrogram_b64"]:
                ok(f"  Spectrogram image: {len(data['spectrogram_b64'])} base64 chars")
            else:
                info(f"  No spectrogram image returned")

            # Check waveform
            if data["waveform_data"]:
                ok(f"  Waveform: {len(data['waveform_data'])} data points")

        except AssertionError as e:
            fail(str(e))
        except Exception as e:
            fail(f"Error testing {sample_id}: {e}")


def test_predict_upload():
    print(f"\n{BOLD}── Test 5: File upload prediction POST /predict{RESET}")

    # Use a demo clip as the test upload
    demo_clip = PROJECT_ROOT / "assets" / "demo_clips" / "real_01.wav"

    if not demo_clip.exists():
        info("No demo clips found — run generate_samples.py first")
        info("Skipping upload test")
        return

    try:
        with open(demo_clip, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/predict",
                files   = {"file": ("real_01.wav", f, "audio/wav")},
                timeout = 30
            )

        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()

        ok(f"Upload prediction: {data['prediction']} ({data['confidence']}%)")
        ok(f"Processing time: {data['processing_ms']}ms")
        ok(f"Spectrogram returned: {'yes' if data['spectrogram_b64'] else 'no'}")

    except AssertionError as e:
        fail(str(e))
    except Exception as e:
        fail(f"Upload test error: {e}")


def test_error_handling():
    print(f"\n{BOLD}── Test 6: Error handling{RESET}")

    # Test invalid sample_id
    r = requests.post(
        f"{BASE_URL}/predict/demo",
        json = {"sample_id": "invalid_id"},
        timeout = 5
    )
    if r.status_code == 400:
        ok("Invalid sample_id correctly returns 400")
    else:
        fail(f"Expected 400 for invalid sample_id, got {r.status_code}")

    # Test missing file
    r = requests.post(f"{BASE_URL}/predict", timeout=5)
    if r.status_code in [400, 422]:
        ok("Missing file correctly returns 400/422")
    else:
        fail(f"Expected 400/422 for missing file, got {r.status_code}")


def main():
    print(f"\n{BOLD}{'═'*50}")
    print("  Audio Deepfake Detector — API Test Suite")
    print(f"{'═'*50}{RESET}")
    print(f"  Testing server at: {BASE_URL}")

    test_root()
    test_health()
    test_metrics()
    test_predict_demo()
    test_predict_upload()
    test_error_handling()

    print(f"\n{BOLD}{'═'*50}")
    print("  Tests complete!")
    print(f"{'═'*50}{RESET}\n")


if __name__ == "__main__":
    main()