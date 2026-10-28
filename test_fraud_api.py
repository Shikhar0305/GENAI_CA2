from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from fraud_api import create_app, load_model_payload


def make_model_artifact(path: Path) -> dict:
    rng = np.random.default_rng(42)
    features = ["V1", "V2", "Amount"]
    x = rng.normal(size=(80, len(features)))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(pd.DataFrame(x, columns=features), y)
    payload = {"model": model, "features": features, "threshold": 0.5}
    joblib.dump(payload, path)
    return payload


def test_model_artifact_reproduces_predictions(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    payload = make_model_artifact(artifact)
    loaded = load_model_payload(str(artifact))
    sample = pd.DataFrame([[0.8, 0.7, 10.0]], columns=payload["features"])

    expected = payload["model"].predict_proba(sample)[:, 1]
    actual = loaded["model"].predict_proba(sample)[:, 1]

    np.testing.assert_allclose(actual, expected)


def test_predict_endpoint_returns_prediction(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    make_model_artifact(artifact)
    client = TestClient(create_app(str(artifact)))

    response = client.post(
        "/predict",
        json={"transaction": {"V1": 0.8, "V2": 0.7, "Amount": 10.0}},
    )

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == 0.5


def test_predict_endpoint_rejects_wrong_features(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    make_model_artifact(artifact)
    client = TestClient(create_app(str(artifact)))

    response = client.post(
        "/predict",
        json={"transaction": {"V1": 0.8, "V2": 0.7, "Wrong": 10.0}},
    )

    assert response.status_code == 422
    assert "missing features: Amount" in response.json()["detail"]
    assert "unexpected features: Wrong" in response.json()["detail"]
