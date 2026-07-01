"""
api/main.py
FraudGuard API — FastAPI server with:
  • Real-time SSE streaming to browser dashboard
  • Model loading from shared volume (no MLflow registry queries)
  • /predict endpoint that scores AND broadcasts to dashboard
  • /api/test for manual scenario testing
  • /static serving of the web dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/mlartifacts/production")
THRESHOLD  = float(os.getenv("FRAUD_THRESHOLD", "0.45"))
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
# ─────────────────────────────────────────────────────────────────────────────

_model = None


# ── Event Bus (SSE) ───────────────────────────────────────────────────────────

class EventBus:
    """Fan-out broadcaster for Server-Sent Events."""
    def __init__(self):
        self._subs: list[asyncio.Queue] = []

    async def publish(self, event: dict) -> None:
        dead = []
        for q in self._subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._unsub(q)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subs.append(q)
        return q

    def _unsub(self, q: asyncio.Queue) -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            pass

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._unsub(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


event_bus = EventBus()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load model from MODEL_PATH.
    Tries sklearn pipeline first, then raw XGBoost file.
    """
    path = MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model folder not found: {path}\n"
            "Run training first:  python training/train.py"
        )

    mlmodel_path = os.path.join(path, "MLmodel")

    if os.path.isfile(mlmodel_path):
        with open(mlmodel_path) as f:
            content = f.read()

        if "sklearn" in content:
            # Try sklearn pipeline
            try:
                import mlflow.sklearn
                m = mlflow.sklearn.load_model(path)
                log.info("Loaded sklearn pipeline from MLflow format")
                return m
            except Exception:
                pass

            # Try joblib directly
            pkl = os.path.join(path, "pipeline.pkl")
            if os.path.isfile(pkl):
                import joblib
                m = joblib.load(pkl)
                log.info("Loaded sklearn pipeline from joblib")
                return m

        if "xgboost" in content:
            try:
                import mlflow.xgboost
                m = mlflow.xgboost.load_model(path)
                log.info("Loaded XGBoost model from MLflow format")
                return m
            except Exception:
                pass

    # Fallback: raw XGBoost file
    for fname in ("model.xgb", "model.json", "model.ubj"):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(fpath)
            log.info(f"Loaded raw XGBoost model: {fname}")
            return m

    raise FileNotFoundError(
        f"No loadable model found in {path}. "
        "Ensure training completed successfully."
    )


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    log.info(f"Loading model from: {MODEL_PATH}")
    try:
        _model = load_model()
        log.info("✅  Model loaded and ready")
    except Exception as e:
        log.error(f"❌  Model load failed: {e}")
        log.warning("    /predict will return 503. Run training/train.py first.")
    yield
    log.info("API shutdown")


app = FastAPI(
    title="FraudGuard API",
    description="Real-time credit card fraud detection with live dashboard.",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    V1: float = 0.0;  V2: float = 0.0;  V3: float = 0.0;  V4: float = 0.0
    V5: float = 0.0;  V6: float = 0.0;  V7: float = 0.0;  V8: float = 0.0
    V9: float = 0.0;  V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(..., description="Transaction amount in USD")


class PredictionResponse(BaseModel):
    transaction_id:    str
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    latency_ms:        float


class TestRequest(BaseModel):
    scenario:  str
    features:  dict
    amount:    Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "threshold":    THRESHOLD,
        "model_path":   MODEL_PATH,
        "subscribers":  event_bus.subscriber_count,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    txn_id:     str,
    features:   TransactionFeatures,
    true_label: Optional[int] = None,   # 0=legit, 1=fraud (from real dataset)
) -> PredictionResponse:

    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training/train.py then restart the API.",
        )

    t0  = time.perf_counter()
    df  = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
    proba = float(_model.predict_proba(df)[0][1])
    latency_ms = (time.perf_counter() - t0) * 1000

    is_fraud = proba >= THRESHOLD

    if is_fraud:
        log.warning(
            f"🚨 FRAUD  id={txn_id}  p={proba:.4f}  "
            f"amount={features.Amount:.2f}  latency={latency_ms:.1f}ms"
        )

    # ── Broadcast to SSE dashboard ────────────────────────────────────────────
    event = {
        "type":              "transaction",
        "transaction_id":   txn_id,
        "amount":           round(features.Amount, 2),
        "fraud_probability": round(proba, 4),
        "is_fraud":          is_fraud,
        "threshold":         THRESHOLD,
        "latency_ms":        round(latency_ms, 2),
        "true_label":        true_label,   # None if not from real dataset
    }
    await event_bus.publish(event)

    return PredictionResponse(
        transaction_id=    txn_id,
        fraud_probability= round(proba, 4),
        is_fraud=          is_fraud,
        threshold_used=    THRESHOLD,
        latency_ms=        round(latency_ms, 2),
    )


@app.post("/api/test")
async def test_scenario(req: TestRequest) -> dict:
    """
    Manual test endpoint for the dashboard's Test panel.
    Does NOT broadcast to the SSE feed — keeps test results private.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_dict = req.features
    if req.amount is not None:
        features_dict["Amount"] = req.amount

    # Fill missing features with 0
    row = {col: features_dict.get(col, 0.0) for col in FEATURE_COLS}

    t0  = time.perf_counter()
    df  = pd.DataFrame([row])[FEATURE_COLS]
    proba = float(_model.predict_proba(df)[0][1])
    latency_ms = (time.perf_counter() - t0) * 1000

    is_fraud = proba >= THRESHOLD

    return {
        "scenario":          req.scenario,
        "fraud_probability": round(proba, 4),
        "is_fraud":          is_fraud,
        "threshold_used":    THRESHOLD,
        "latency_ms":        round(latency_ms, 2),
    }


@app.get("/api/stream")
async def sse_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint — browser connects here for live updates."""
    queue = event_bus.subscribe()

    async def generate():
        try:
            # Initial connection ping
            yield "data: {\"type\":\"ping\",\"msg\":\"Connected to FraudGuard stream\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield "data: {\"type\":\"ping\"}\n\n"
        finally:
            event_bus.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "Connection":      "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
