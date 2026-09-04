import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "TimesFM 3 Forecasting Service"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/v1"

    # Model settings
    MODEL_ID: str = "google/timesfm-3.0-pytorch"
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Context and horizon constraints
    MAX_CONTEXT: int = 1024
    MAX_HORIZON: int = 512
    DEFAULT_HORIZON: int = 24
    NORMALIZE_INPUTS: bool = True

    # Loading configuration
    LAZY_LOAD: bool = False
    MOCK_MODE: bool = False  # Enable for testing without HF model download
    HF_HOME: Optional[str] = os.environ.get("HF_HOME", "/app/cache")

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1


settings = Settings()
