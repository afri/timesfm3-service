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
    return_quantiles: bool = Field(
        default=False,
        description="Whether to return probabilistic quantile forecasts (9 deciles: 0.1 to 0.9).",
    )
    use_symmetric_averaging: bool = Field(
        default=False,
        description="Whether to use symmetric averaging across forward and backward prediction.",
    )
    past_only_covariates: Optional[
        Union[
            List[float],
            List[List[float]],
            List[List[List[float]]],
        ]
    ] = Field(
        default=None,
        description=(
            "Dynamic covariates known only in the past context window. "
            "Shape per series: (num_channels, context_len) or 1D (context_len)."
        ),
    )
    past_future_covariates: Optional[
        Union[
            List[float],
            List[List[float]],
            List[List[List[float]]],
        ]
    ] = Field(
        default=None,
        description=(
            "Dynamic covariates known for both past context and future horizon. "
            "Shape per series: (num_channels, context_len + horizon) or 1D (context_len + horizon)."
        ),
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
    quantiles: Optional[
        Union[
            List[List[float]],
            List[List[List[float]]],
            List[List[List[List[float]]]],
        ]
    ] = Field(
        default=None,
        description=(
            "Probabilistic quantile forecasts for the 9 model deciles [0.1, 0.2, ..., 0.9]. "
            "Shape is (horizon, 9) for a 1D series, or corresponding shape for batch/multivariate."
        ),
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
    version: str


class ModelCard(BaseModel):
    """Individual model descriptor (OpenAI compatible)."""

    id: str = Field(..., description="Unique model identifier.")
    object: str = Field(default="model", description="The object type, always 'model'.")
    created: int = Field(default=1700000000, description="Unix timestamp of model creation.")
    owned_by: str = Field(default="google", description="Organization or entity that owns the model.")
    max_context: Optional[int] = Field(default=None, description="Maximum input context window length.")
    max_horizon: Optional[int] = Field(default=None, description="Maximum forecasting horizon steps.")
    backend: Optional[str] = Field(default=None, description="Active compute backend.")


class ModelListResponse(BaseModel):
    """List of available models (OpenAI compatible)."""

    object: str = Field(default="list", description="The object type, always 'list'.")
    data: List[ModelCard] = Field(..., description="Array of model objects.")


