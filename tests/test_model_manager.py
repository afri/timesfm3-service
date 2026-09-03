import pytest
from app.model_manager import TimesFMModelManager


def test_model_manager_singleton():
    mgr1 = TimesFMModelManager()
    mgr2 = TimesFMModelManager()
    assert mgr1 is mgr2


def test_mock_forecast_univariate():
    mgr = TimesFMModelManager()
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    horizon = 8
    point, quantiles, elapsed = mgr.forecast(series=series, horizon=horizon)

    assert len(point) == horizon
    assert isinstance(point[0], float)
    assert elapsed >= 0.0
    assert quantiles is not None


def test_mock_forecast_batch():
    mgr = TimesFMModelManager()
    series = [
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
        [100.0, 200.0, 300.0],
    ]
    horizon = 4
    point, quantiles, elapsed = mgr.forecast(series=series, horizon=horizon)

    assert len(point) == 3
    for p in point:
        assert len(p) == horizon
