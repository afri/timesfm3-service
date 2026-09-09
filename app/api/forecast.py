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
            return_quantiles=request.return_quantiles,
            use_symmetric_averaging=request.use_symmetric_averaging,
            past_only_covariates=request.past_only_covariates,
            past_future_covariates=request.past_future_covariates,
        )

        return ForecastResponse(
            point_forecast=point,
            quantiles=quantiles,
            horizon=target_horizon,
            model_id=settings.MODEL_ID,
            inference_time_ms=round(elapsed_ms, 2),
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid forecast request: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting failed: {str(e)}",
        )

