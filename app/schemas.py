from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Forecasting request schema."""

    series: Union[List[float], List[List[float]], List[List[List[float]]]] = Field(
        ...,
        description=(
            "Input time series. Can be a single 1D series [y1, y2, ...], "
            "a batch of 1D series [[...], [...]], or multivariate series."
        ),
        examples=[[10.5, 11.2, 12.0, 11.8, 12.5, 13.1, 13.0, 13.8, 14.2, 14.9]],
    )
    horizon: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Forecast horizon (number of future steps to predict). Defaults to server default.",
    )
    frequency: Optional[Union[int, str]] = Field(
        default=0,
        description="Frequency indicator: 0 for high-frequency/unspecified, 1 for daily, 2 for weekly, etc.",
    )
    quantiles: Optional[List[float]] = Field(
        default=None,
        description="Optional specific quantiles to return (e.g. [0.1, 0.5, 0.9]).",
    )
    past_covariates: Optional[Dict[str, List[Union[float, List[float]]]]] = Field(
        default=None,
        description="Dynamic covariates observed in the past context window.",
    )
    future_covariates: Optional[Dict[str, List[Union[float, List[float]]]]] = Field(
        default=None,
        description="Dynamic covariates known for the future horizon.",
    )

    @field_validator("series")
    @classmethod
    def validate_series_not_empty(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Input series cannot be empty")
        return v


class ForecastResponse(BaseModel):
    """Forecasting response schema."""

    point_forecast: Union[List[float], List[List[float]], List[List[List[float]]]] = Field(
        ...,
        description="Point predictions for the requested horizon (median/mean).",
    )
    quantiles: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Probabilistic quantile forecasts (e.g. q10, q20 ... q90) if computed.",
    )
    horizon: int = Field(..., description="Number of forecasted steps.")
    model_id: str = Field(..., description="Model identifier used for inference.")
    inference_time_ms: float = Field(..., description="Time taken to compute forecast in milliseconds.")


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field(..., description="'ok', 'degraded', or 'initializing'.")
    model_loaded: bool = Field(..., description="Whether the model weights are loaded and ready.")
    model_id: str = Field(..., description="Configured model identifier.")
    device: str = Field(..., description="Active compute device (e.g. 'cpu', 'cuda:0', 'mps').")
    mock_mode: bool = Field(..., description="True if running with mock inference for testing.")
    version: str = Field(..., description="Webservice API version.")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_id: str
    backend: str
    device: str
    max_context: int
    max_horizon: int
    default_horizon: int
    normalize_inputs: bool
    mock_mode: bool
    version: str
