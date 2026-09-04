import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

# Ensure mock mode during test runs
settings.MOCK_MODE = True

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
    assert data["mock_mode"] is True


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
    assert data["horizon"] == 10
    assert data["model_id"] == settings.MODEL_ID
    assert data["inference_time_ms"] >= 0


def test_forecast_batch_series():
    payload = {
        "series": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ],
        "horizon": 5,
        "quantiles": [0.1, 0.5, 0.9],
    }
    response = client.post("/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["point_forecast"]) == 2
    assert len(data["point_forecast"][0]) == 5
    assert len(data["point_forecast"][1]) == 5
    assert data["quantiles"] is not None


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
