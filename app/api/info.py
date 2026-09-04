from fastapi import APIRouter, HTTPException
from app.config import settings
from app.model_manager import model_manager
from app.schemas import ModelCard, ModelInfoResponse, ModelListResponse

router = APIRouter(tags=["Metadata"])


def _build_model_card() -> ModelCard:
    return ModelCard(
        id=settings.MODEL_ID,
        object="model",
        owned_by="google",
        max_context=settings.MAX_CONTEXT,
        max_horizon=settings.MAX_HORIZON,
        backend=model_manager.backend,
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model and service information",
    description="Returns metadata about the active TimesFM model, configuration limits, and backend.",
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


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Returns an array of available models in OpenAI-compatible collection format.",
)
def list_models() -> ModelListResponse:
    return ModelListResponse(data=[_build_model_card()])


@router.get(
    "/models/{model_id:path}",
    response_model=ModelCard,
    summary="Retrieve model details",
    description="Returns details for a specific model by ID.",
)
def get_model(model_id: str) -> ModelCard:
    if model_id not in (settings.MODEL_ID, settings.MODEL_ID.split("/")[-1]):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available model: '{settings.MODEL_ID}'.",
        )
    return _build_model_card()

