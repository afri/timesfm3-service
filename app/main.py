import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import forecast, health, info
from app.config import settings
from app.model_manager import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("timesfm3.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle context manager."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    if not settings.LAZY_LOAD:
        try:
            logger.info("Pre-warming TimesFM model...")
            model_manager.load_model()
            logger.info("Model ready for inference.")
        except Exception as e:
            logger.warning(
                f"Model pre-warm failed or skipped (will retry on first request): {e}"
            )
    yield
    logger.info("Shutting down TimesFM service...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready HTTP REST API for time series forecasting wrapping Google Research's "
        "TimesFM 3 foundation model (https://github.com/google-research/timesfm)."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root redirect to interactive documentation
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


# Health routes (both at root and under /v1)
app.include_router(health.router)
app.include_router(health.router, prefix=settings.API_PREFIX)

# API routes
app.include_router(forecast.router, prefix=settings.API_PREFIX)
app.include_router(forecast.router)  # Also allow /forecast directly
app.include_router(info.router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
    )
