from fastapi import APIRouter, HTTPException, status
from app.config import settings
from app.model_manager import model_manager
from app.schemas import ForecastRequest, ForecastResponse

router = APIRouter(tags=["Forecasting"])


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Generate time series forecasts",
    description="Generate zero-shot forecasts for univariate or multivariate time series inputs.",
)
def forecast(request: ForecastRequest) -> ForecastResponse:
    target_horizon = request.horizon or settings.DEFAULT_HORIZON

    if target_horizon > settings.MAX_HORIZON:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested horizon {target_horizon} exceeds max allowed horizon {settings.MAX_HORIZON}",
        )

    try:
        point, quantiles, elapsed_ms = model_manager.forecast(
            series=request.series,
            horizon=target_horizon,
            frequency=request.frequency,
            quantiles=request.quantiles,
            past_covariates=request.past_covariates,
            future_covariates=request.future_covariates,
        )

        return ForecastResponse(
            point_forecast=point,
            quantiles=quantiles,
            horizon=target_horizon,
            model_id=settings.MODEL_ID,
            inference_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting failed: {str(e)}",
        )
