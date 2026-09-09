import numpy as np
import pytest
from app.model_manager import TimesFMModelManager


class DummyOutput:
    def __init__(self, forecast, quantiles=None):
        self.forecast = forecast
        self.quantiles = quantiles


class DummyModel:
    def __init__(self):
        self.last_kwargs = {}

    def predict_batch(self, contexts, horizon, return_quantiles=False, use_symmetric_averaging=False, **kwargs):
        self.last_kwargs = {
            "contexts": contexts,
            "horizon": horizon,
            "return_quantiles": return_quantiles,
            "use_symmetric_averaging": use_symmetric_averaging,
            **kwargs,
        }
        outputs = []
        for ctx in contexts:
            f = np.arange(horizon, dtype=np.float32) + float(ctx[-1])
            q = None
            if return_quantiles:
                # Shape (horizon, 9)
                q = np.tile(f[:, None], (1, 9)) + np.linspace(0.1, 0.9, 9, dtype=np.float32)
            outputs.append(DummyOutput(forecast=f, quantiles=q))
        return outputs


def test_model_manager_singleton():
    mgr1 = TimesFMModelManager()
    mgr2 = TimesFMModelManager()
    assert mgr1 is mgr2


def test_forecast_univariate_default_quantiles():
    mgr = TimesFMModelManager()
    mgr.model = DummyModel()
    mgr.backend = "timesfm3"

    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    horizon = 8
    point, quantiles, elapsed = mgr.forecast(series=series, horizon=horizon)

    assert len(point) == horizon
    assert isinstance(point[0], float)
    assert elapsed >= 0.0
    assert quantiles is None


def test_forecast_univariate_with_quantiles():
    mgr = TimesFMModelManager()
    dummy = DummyModel()
    mgr.model = dummy
    mgr.backend = "timesfm3"

    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    horizon = 8
    point, quantiles, elapsed = mgr.forecast(
        series=series,
        horizon=horizon,
        return_quantiles=True,
        use_symmetric_averaging=True,
    )

    assert len(point) == horizon
    assert quantiles is not None
    assert len(quantiles) == horizon
    for q_step in quantiles:
        assert len(q_step) == 9  # 9 deciles (0.1 to 0.9)
    assert dummy.last_kwargs["use_symmetric_averaging"] is True
    assert dummy.last_kwargs["return_quantiles"] is True


def test_forecast_batch_with_covariates():
    mgr = TimesFMModelManager()
    dummy = DummyModel()
    mgr.model = dummy
    mgr.backend = "timesfm3"

    series = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
    ]
    past_only = [
        [[0.5, 0.6, 0.7, 0.8, 0.9]],
        [[1.5, 1.6, 1.7, 1.8, 1.9]],
    ]
    past_future = [
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]],
    ]
    horizon = 4
    point, quantiles, elapsed = mgr.forecast(
        series=series,
        horizon=horizon,
        return_quantiles=True,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
    )

    assert len(point) == 2
    for p in point:
        assert len(p) == horizon
    assert quantiles is not None
    assert len(quantiles) == 2
    for q_series in quantiles:
        assert len(q_series) == horizon
        for q_step in q_series:
            assert len(q_step) == 9

    assert "past_only_covariates" in dummy.last_kwargs
    assert "past_future_covariates" in dummy.last_kwargs


def test_forecast_single_series_1d_covariates():
    mgr = TimesFMModelManager()
    dummy = DummyModel()
    mgr.model = dummy
    mgr.backend = "timesfm3"

    series = [1.0, 2.0, 3.0, 4.0, 5.0]  # context_len = 5
    horizon = 3
    # 1D covariates
    past_only = [0.1, 0.2, 0.3, 0.4, 0.5]  # length 5
    past_future = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # length 5 + 3 = 8

    point, quantiles, elapsed = mgr.forecast(
        series=series,
        horizon=horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
    )

    assert len(point) == horizon
    assert "past_only_covariates" in dummy.last_kwargs
    # Should be reshaped to (1, 5) inside the list
    assert dummy.last_kwargs["past_only_covariates"][0].shape == (1, 5)
    assert dummy.last_kwargs["past_future_covariates"][0].shape == (1, 8)


def test_forecast_empty_covariates_normalized_to_none():
    mgr = TimesFMModelManager()
    dummy = DummyModel()
    mgr.model = dummy
    mgr.backend = "timesfm3"

    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    horizon = 4
    point, quantiles, elapsed = mgr.forecast(
        series=series,
        horizon=horizon,
        past_only_covariates=[],
        past_future_covariates=[[]],
    )

    assert len(point) == horizon
    assert "past_only_covariates" not in dummy.last_kwargs
    assert "past_future_covariates" not in dummy.last_kwargs


def test_forecast_covariate_length_validation():
    mgr = TimesFMModelManager()
    dummy = DummyModel()
    mgr.model = dummy
    mgr.backend = "timesfm3"

    series = [1.0, 2.0, 3.0, 4.0, 5.0]  # context_len = 5
    horizon = 4

    # past_only has length 3 instead of 5
    with pytest.raises(ValueError, match="does not match context length"):
        mgr.forecast(
            series=series,
            horizon=horizon,
            past_only_covariates=[0.1, 0.2, 0.3],
        )

    # past_future has length 4 instead of 5 + 4 = 9 (future only)
    with pytest.raises(ValueError, match="does not match expected length context_len \\+ horizon"):
        mgr.forecast(
            series=series,
            horizon=horizon,
            past_future_covariates=[0.1, 0.2, 0.3, 0.4],
        )

