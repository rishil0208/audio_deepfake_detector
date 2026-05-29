"""
app.py

The main FastAPI application.

This file:
1. Creates the FastAPI app instance
2. Adds CORS middleware (allows browser requests from React)
3. Loads ML models at startup (once, not on every request)
4. Registers all route files (health, metrics, predict)
5. Provides a root endpoint for quick sanity checks

To start the server:
    cd audio-deepfake-detector
    uvicorn backend.app:app --reload --port 8000

The --reload flag means: restart automatically when you save any Python file.
Very useful during development.

After starting, open:
    http://localhost:8000/docs   ← interactive API documentation (auto-generated!)
    http://localhost:8000/health ← quick health check
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to Python path so all imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Lifespan: code that runs at server startup and shutdown ───────────────────
# The @asynccontextmanager + lifespan pattern is the modern FastAPI way
# to run setup code once when the server starts.
# We use it to load ML models into memory so they're ready for requests.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts (before handling any requests).
    'yield' separates startup code (above) from shutdown code (below).

    Why load models at startup instead of on first request?
    - First request would be very slow (10–30 seconds for model load)
    - All subsequent requests benefit from the cached model in memory
    - The server is ready to handle requests immediately after starting
    """
    print("\n" + "═" * 50)
    print("  Audio Deepfake Detector API — Starting up")
    print("═" * 50)

    # Load both ML models into memory
    try:
        from backend.ml.inference import engine
        engine.load_models()
        print(f"\n  Models loaded on device: {engine.status['device']}")
        print(f"  CNN available:           {engine.status['cnn_loaded']}")
        print(f"  Baseline available:      {engine.status['baseline_loaded']}")
    except Exception as e:
        print(f"\n  ⚠  Model loading error: {e}")
        print("     Server will start but predictions may fail.")
        print("     Ensure you have run train_baseline.py and train_cnn.py")

    print("\n  ✓ Server ready at http://localhost:8000")
    print("  ✓ API docs at    http://localhost:8000/docs")
    print("═" * 50 + "\n")

    yield   # server runs here — handles requests

    # Shutdown code (after yield) — runs when server stops
    print("\nServer shutting down...")


# ── Create the FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title       = "Audio Deepfake Detector API",
    description = (
        "Detects whether an audio clip is real human speech "
        "or AI-generated (deepfake) using CNN and Random Forest models."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,   # run our startup/shutdown code
    docs_url    = "/docs",    # Swagger UI at /docs
    redoc_url   = "/redoc"    # ReDoc UI at /redoc
)


# ── CORS Middleware ────────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
#
# Without this, the browser will BLOCK requests from:
#   http://localhost:5173  (React dev server)
# to:
#   http://localhost:8000  (FastAPI server)
#
# Because they run on different ports, they are considered "different origins."
# The browser treats this as a security risk by default.
#
# This middleware tells the browser: "yes, allow requests from these origins."
# allow_origins=["*"] means allow ALL origins — fine for local development.
# In production you would list specific allowed domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],    # allow all origins (for local dev)
    allow_credentials = True,
    allow_methods     = ["*"],    # allow GET, POST, PUT, DELETE, etc.
    allow_headers     = ["*"],    # allow all headers
)


# ── Register route files ───────────────────────────────────────────────────────
# Each router is a group of related endpoints.
# include_router() adds all its endpoints to the main app.

from backend.routes.health  import router as health_router
from backend.routes.metrics import router as metrics_router
from backend.routes.predict import router as predict_router

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(predict_router)


# ── Root endpoint ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """
    The root URL — just confirms the API is running.
    Useful as a quick check: curl http://localhost:8000/
    """
    return JSONResponse(content={
        "message": "Audio Deepfake Detector API is running",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health"
    })