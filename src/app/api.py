"""FastAPI service exposing all trained oil-price models.

Run with:

    uvicorn src.app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.advanced_model import TemporalFusionTransformer
from src.models.deep_model import build_deep_model

logger = logging.getLogger("oil_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class Settings:
    def __init__(self) -> None:
        root = _project_root()
        self.model_dir = Path(os.environ.get("MODEL_DIR", root / "models"))
        self.processed_dir = Path(
            os.environ.get("PROCESSED_DIR", root / "data" / "processed")
        )
        self.metrics_dir = Path(
            os.environ.get("METRICS_DIR", root / "outputs" / "metrics")
        )
        self.seq_len_default = int(os.environ.get("SEQ_LEN", "30"))


settings = Settings()


# ---------------------------------------------------------------------------
# Model registry (populated at startup)
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns: list[str] = []
        self.seq_len: int = settings.seq_len_default
        self.rf = None
        self.deep: dict[str, torch.nn.Module] = {}
        self.tft: TemporalFusionTransformer | None = None
        self.tft_quantiles: list[float] = []
        self.loaded: list[str] = []
        self.metrics_summary: dict = {}

    def load(self) -> None:
        # Feature names + scaler
        feat_path = settings.processed_dir / "feature_names.json"
        if feat_path.exists():
            self.feature_columns = json.loads(feat_path.read_text())
            logger.info("Loaded %d feature columns", len(self.feature_columns))

        # Random Forest baseline
        rf_path = settings.model_dir / "baseline_rf.pkl"
        if rf_path.exists():
            bundle = joblib.load(rf_path)
            self.rf = bundle["pipeline"]
            if not self.feature_columns:
                self.feature_columns = bundle.get("feature_columns", [])
            self.loaded.append("rf")
            logger.info("Loaded RandomForest from %s", rf_path)

        # Deep models
        for name in ("bilstm", "cnn_gru"):
            path = settings.model_dir / f"deep_{name}.pt"
            if not path.exists():
                continue
            ckpt = torch.load(path, map_location=self.device)
            model = build_deep_model(name, n_features=ckpt["n_features"]).to(self.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.deep[name] = model
            self.seq_len = int(ckpt.get("seq_len", self.seq_len))
            if not self.feature_columns:
                self.feature_columns = ckpt.get("feature_columns", [])
            self.loaded.append(name)
            logger.info("Loaded deep model %s from %s", name, path)

        # TFT
        tft_path = settings.model_dir / "advanced_tft.pt"
        if tft_path.exists():
            ckpt = torch.load(tft_path, map_location=self.device)
            quantiles = tuple(ckpt.get("quantiles", [0.1, 0.5, 0.9]))
            tft = TemporalFusionTransformer(
                n_features=ckpt["n_features"], quantiles=quantiles
            ).to(self.device)
            tft.load_state_dict(ckpt["state_dict"])
            tft.eval()
            self.tft = tft
            self.tft_quantiles = list(quantiles)
            self.seq_len = int(ckpt.get("seq_len", self.seq_len))
            if not self.feature_columns:
                self.feature_columns = ckpt.get("feature_columns", [])
            self.loaded.append("tft")
            logger.info("Loaded TFT from %s", tft_path)

        # Metrics summary
        comparison_csv = settings.metrics_dir / "model_comparison.csv"
        if comparison_csv.exists():
            import pandas as pd

            self.metrics_summary = (
                pd.read_csv(comparison_csv).to_dict(orient="records")
            )

        # Naive model is always available — it doesn't need any state.
        self.loaded.append("naive")

    # ----- Predictions ------------------------------------------------------
    def predict_naive(self, sequence: np.ndarray) -> float:
        """Naive persistence on a *raw* (unscaled) sequence — uses the last
        observation's price column as the next-day prediction."""
        if "WTI_price" in self.feature_columns:
            idx = self.feature_columns.index("WTI_price")
            return float(sequence[-1, idx])
        # Fallback to the first feature
        return float(sequence[-1, 0])

    def predict_rf(self, sequence: np.ndarray) -> float:
        if self.rf is None:
            raise HTTPException(status_code=503, detail="Random Forest model not loaded.")
        last_row = sequence[-1].reshape(1, -1)
        return float(self.rf.predict(last_row)[0])

    def predict_deep(self, name: str, sequence: np.ndarray) -> float:
        model = self.deep.get(name)
        if model is None:
            raise HTTPException(status_code=503, detail=f"Deep model {name!r} not loaded.")
        x = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model(x).cpu().numpy().reshape(-1)
        return float(pred[0])

    def predict_tft(self, sequence: np.ndarray) -> dict[str, float]:
        if self.tft is None:
            raise HTTPException(status_code=503, detail="TFT model not loaded.")
        x = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.tft(x).cpu().numpy().reshape(-1)
        out = {f"q{int(q*100)}": float(p) for q, p in zip(self.tft_quantiles, preds)}
        return out


registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    model: str = Field(..., description="naive | rf | bilstm | cnn_gru | tft")
    sequence: list[list[float]] = Field(
        ...,
        description="Last `seq_len` rows of feature vectors (shape: seq_len x n_features).",
    )


class ConfidenceInterval(BaseModel):
    q10: float
    q50: float
    q90: float


class PredictionResponse(BaseModel):
    model: str
    predicted_price: float
    unit: str = "USD/barrel"
    confidence_interval: Optional[ConfidenceInterval] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    seq_len: int
    n_features: int


class ModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401 (FastAPI signature)
    try:
        registry.load()
        logger.info("Models loaded: %s", registry.loaded)
    except Exception:  # pragma: no cover
        logger.exception("Failed to load some models — API will respond with 503 where needed.")
    yield


app = FastAPI(
    title="Oil Price Prediction API",
    description="Serve baseline, deep-learning and TFT models for next-day WTI price.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded,
        seq_len=registry.seq_len,
        n_features=len(registry.feature_columns),
    )


@app.get("/models")
def list_models() -> dict:
    info = [
        ModelInfo(name="naive", type="heuristic", loaded=True),
        ModelInfo(name="rf", type="random_forest", loaded=registry.rf is not None),
        ModelInfo(name="bilstm", type="deep_lstm", loaded="bilstm" in registry.deep),
        ModelInfo(name="cnn_gru", type="deep_cnn_gru", loaded="cnn_gru" in registry.deep),
        ModelInfo(name="tft", type="transformer", loaded=registry.tft is not None),
    ]
    return {
        "feature_columns": registry.feature_columns,
        "seq_len": registry.seq_len,
        "models": [m.model_dump() for m in info],
        "metrics_summary": registry.metrics_summary,
    }


def _validate_sequence(seq: list[list[float]]) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="sequence must be a 2D array")
    expected_features = len(registry.feature_columns)
    if expected_features and arr.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Each sequence row must have {expected_features} features, "
                f"got {arr.shape[1]}."
            ),
        )
    if arr.shape[0] != registry.seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"sequence must have exactly {registry.seq_len} time-steps.",
        )
    return arr


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    arr = _validate_sequence(req.sequence)
    name = req.model.lower()

    if name == "naive":
        price = registry.predict_naive(arr)
    elif name == "rf":
        price = registry.predict_rf(arr)
    elif name in {"bilstm", "cnn_gru"}:
        price = registry.predict_deep(name, arr)
    elif name == "tft":
        q = registry.predict_tft(arr)
        return PredictionResponse(
            model=name,
            predicted_price=q.get("q50", float("nan")),
            confidence_interval=ConfidenceInterval(
                q10=q.get("q10", float("nan")),
                q50=q.get("q50", float("nan")),
                q90=q.get("q90", float("nan")),
            ),
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {req.model!r}. Use one of: naive, rf, bilstm, cnn_gru, tft.",
        )

    return PredictionResponse(model=name, predicted_price=price)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("src.app.api:app", host=host, port=port, reload=False)
