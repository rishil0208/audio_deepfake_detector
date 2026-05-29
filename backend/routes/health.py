"""
health.py

GET /health

Returns the current status of the server and models.
Called by the frontend on page load to show the status badge.

Response example:
{
  "status": "ok",
  "models": {
    "cnn_loaded": true,
    "baseline_loaded": true,
    "device": "cpu"
  },
  "version": "1.0.0"
}
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# APIRouter is like a mini-app — it holds a group of related endpoints.
# app.py will import this router and register it with the main app.
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Returns server and model status.

    The 'async' keyword means this function is non-blocking.
    FastAPI can handle other requests while this one is waiting.
    For simple functions like this it doesn't matter much,
    but it's good practice for all FastAPI handlers.
    """
    # Import here (not at top) to avoid circular import issues
    # The engine is a singleton defined in inference.py
    from backend.ml.inference import engine

    return JSONResponse(content={
        "status":  "ok",
        "models":  engine.status,
        "version": "1.0.0",
        "endpoints": [
            "POST /predict",
            "POST /predict/demo",
            "GET  /metrics",
            "GET  /health"
        ]
    })