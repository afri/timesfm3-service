import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.model_manager import model_manager


class DummyOutput:
    def __init__(self, forecast, quantiles=None):
        self.forecast = forecast
        self.quantiles = quantiles


class DummyModel:
    def predict_batch(self, contexts, horizon, return_quantiles=False, use_symmetric_averaging=False, **kwargs):
        outputs = []
        for ctx in contexts:
            f = np.arange(horizon, dtype=np.float32) + float(ctx[-1])
            q = None
            if return_quantiles:
                # Shape (horizon, 9)
                q = np.tile(f[:, None], (1, 9)) + np.linspace(0.1, 0.9, 9, dtype=np.float32)
            outputs.append(DummyOutput(forecast=f, quantiles=q))
        return outputs


# Attach test model
model_manager.model = DummyModel()
model_manager.backend = "timesfm3"

client = TestClient(app)


def test_docs_endpoint():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "scalar" in response.text.lower()


def test_openapi_json_endpoint():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/v1/forecast" in data["paths"]


def test_root_redirect_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "device" in data
    assert "version" in data


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_info_endpoint():
    response = client.get("/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == settings.MODEL_ID
    assert data["max_horizon"] == settings.MAX_HORIZON
    assert data["backend"] == "timesfm3"


def test_models_list_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == settings.MODEL_ID
    assert data["data"][0]["object"] == "model"


def test_models_retrieve_endpoint():
    response = client.get(f"/v1/models/{settings.MODEL_ID}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == settings.MODEL_ID
    assert data["object"] == "model"

    # Test 404 for unknown model
    response_404 = client.get("/v1/models/non-existent-model")
    assert response_404.status_code == 404



def test_forecast_univariate_single_series():
    payload = {
        "series": [10.0, 11.0, 12.5, 13.0, 14.2, 15.0, 16.5, 17.0],
        "horizon": 10,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "point_forecast" in data
    assert len(data["point_forecast"]) == 10
    assert data["quantiles"] is None
    assert data["horizon"] == 10
    assert data["model_id"] == settings.MODEL_ID
    assert data["inference_time_ms"] >= 0


def test_forecast_univariate_with_quantiles():
    payload = {
        "series": [10.0, 11.0, 12.5, 13.0, 14.2, 15.0, 16.5, 17.0],
        "horizon": 6,
        "return_quantiles": True,
        "use_symmetric_averaging": True,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["point_forecast"]) == 6
    assert data["quantiles"] is not None
    assert len(data["quantiles"]) == 6
    for q_step in data["quantiles"]:
        assert len(q_step) == 9  # 9 deciles


def test_forecast_batch_series_with_covariates():
    payload = {
        "series": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ],
        "horizon": 5,
        "return_quantiles": True,
        "past_only_covariates": [
            [[0.1, 0.2, 0.3, 0.4, 0.5]],
            [[1.1, 1.2, 1.3, 1.4, 1.5]],
        ],
        "past_future_covariates": [
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
            [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]],
        ],
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["point_forecast"]) == 2
    assert len(data["point_forecast"][0]) == 5
    assert len(data["point_forecast"][1]) == 5
    assert data["quantiles"] is not None
    assert len(data["quantiles"]) == 2
    assert len(data["quantiles"][0]) == 5
    assert len(data["quantiles"][0][0]) == 9


def test_forecast_default_horizon():
    payload = {
        "series": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["point_forecast"]) == settings.DEFAULT_HORIZON

    # Verify unversioned /forecast is no longer routed
    unversioned_resp = client.post("/forecast", json=payload)
    assert unversioned_resp.status_code == 404


def test_forecast_empty_series_validation():
    payload = {
        "series": [],
        "horizon": 12,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 422


def test_forecast_exceeds_max_horizon():
    payload = {
        "series": [1.0, 2.0, 3.0],
        "horizon": settings.MAX_HORIZON + 100,
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 400
    assert "exceeds max allowed horizon" in response.json()["detail"]


def test_forecast_covariate_length_mismatch_returns_400():
    payload = {
        "series": [1.0, 2.0, 3.0, 4.0, 5.0],
        "horizon": 4,
        # Only 4 future steps provided instead of 5 + 4 = 9
        "past_future_covariates": [0.1, 0.2, 0.3, 0.4],
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 400
    assert "Invalid forecast request" in response.json()["detail"]
    assert "past_future_covariates length (4)" in response.json()["detail"]


def test_forecast_empty_covariates_returns_200():
    payload = {
        "series": [1.0, 2.0, 3.0, 4.0, 5.0],
        "horizon": 4,
        "past_only_covariates": [],
        "past_future_covariates": [],
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    assert len(response.json()["point_forecast"]) == 4

