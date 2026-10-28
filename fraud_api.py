"""FastAPI service for predictions from the saved credit-card fraud model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))


class PredictionRequest(BaseModel):
    """One raw transaction keyed by the model's feature names."""

    transaction: dict[str, float] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Fraud prediction returned by the service."""

    fraud_probability: float
    prediction: int
    threshold: float


@lru_cache(maxsize=1)
def load_model_payload(model_path: str = str(MODEL_PATH)) -> dict[str, Any]:
    """Load the artifact produced by ``creditcardfraud.py`` once per process."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Train the model first."
        )

    payload = joblib.load(path)
    required_keys = {"model", "features", "threshold"}
    if not required_keys.issubset(payload):
        missing = ", ".join(sorted(required_keys - set(payload)))
        raise ValueError(f"Model artifact is missing required keys: {missing}")
    return payload


def predict_transaction(
    transaction: dict[str, float], payload: dict[str, Any]
) -> PredictionResponse:
    """Validate a transaction and return the model's fraud prediction."""

    features = list(payload["features"])
    provided = set(transaction)
    expected = set(features)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing features: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected features: {', '.join(extra)}")
        raise ValueError("; ".join(details))

    frame = pd.DataFrame([[transaction[feature] for feature in features]], columns=features)
    probability = float(payload["model"].predict_proba(frame)[0, 1])
    threshold = float(payload["threshold"])
    return PredictionResponse(
        fraud_probability=round(probability, 6),
        prediction=int(probability >= threshold),
        threshold=threshold,
    )


def create_app(model_path: str = str(MODEL_PATH)) -> FastAPI:
    """Create an application configured to use a specific model artifact."""

    app = FastAPI(
        title="Credit Card Fraud Detection API",
        version="1.0.0",
        description="Predicts whether a transaction is fraudulent.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            load_model_payload(model_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            payload = load_model_payload(model_path)
            return predict_transaction(request.transaction, payload)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503 if isinstance(exc, FileNotFoundError) else 422,
                                detail=str(exc)) from exc

    return app


app = create_app()
