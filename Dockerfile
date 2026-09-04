# Stage 1: Base image
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/cache \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies (git is required for installing timesfm from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU by default for portable, lightweight container builds)
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX_URL}

# Copy dependency specifications
COPY requirements.txt requirements-dev.txt ./

# Install python dependencies including TimesFM from source repo
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user and model cache directory
RUN groupadd -g 1000 timesfm && \
    useradd -u 1000 -g timesfm -m -s /bin/bash timesfm && \
    mkdir -p /app/cache && \
    chown -R timesfm:timesfm /app

# Copy application source code and scripts
COPY --chown=timesfm:timesfm app/ ./app/
COPY --chown=timesfm:timesfm tests/ ./tests/
COPY --chown=timesfm:timesfm scripts/ ./scripts/

# Switch to non-root user
USER timesfm

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command to run web service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
