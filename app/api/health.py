from fastapi import APIRouter
from app.config import settings
from app.model_manager import model_manager
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service status, active device, and whether the TimesFM model is loaded.",
)
@router.get(
    "/healthz",
    response_model=HealthResponse,
    include_in_schema=False,
)
def get_health() -> HealthResponse:
    is_loaded = model_manager.is_loaded
    status = "ok" if is_loaded else "ready"
    return HealthResponse(
        status=status,
        model_loaded=is_loaded,
        model_id=settings.MODEL_ID,
        device=model_manager.device,
        version=settings.APP_VERSION,
    )
