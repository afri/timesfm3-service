from fastapi import APIRouter
from app.config import settings
from app.model_manager import model_manager
from app.schemas import ModelInfoResponse

router = APIRouter(tags=["Metadata"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Returns metadata about the active TimesFM model, configuration limits, and backend.",
)
@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="Active model details",
    description="Returns details of the configured forecasting foundation model.",
)
def get_model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_id=settings.MODEL_ID,
        backend=model_manager.backend,
        device=model_manager.device,
        max_context=settings.MAX_CONTEXT,
        max_horizon=settings.MAX_HORIZON,
        default_horizon=settings.DEFAULT_HORIZON,
        normalize_inputs=settings.NORMALIZE_INPUTS,
        mock_mode=model_manager.mock_mode or (model_manager.backend == "mock"),
        version=settings.APP_VERSION,
    )
