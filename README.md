# TimesFM 3 Forecasting Service

Containerized, production-ready REST API wrapping Google Research's **TimesFM 3** time-series foundation model ([google-research/timesfm](https://github.com/google-research/timesfm)).

TimesFM is a decoder-only transformer model designed for zero-shot time series forecasting. Version 3.0 introduces native multivariate forecasting, flexible covariate support (past-only and past-and-future), and state-of-the-art benchmark accuracy.

---

## ⚡ Prerequisites

To build and run this service, you only need:
- **Git**
- **Docker** (Docker Engine 20.10+ / Docker Desktop)

> **No local Python installation or machine learning libraries are required on the host machine.** Everything runs isolated inside the Docker container.

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-org>/timesfm3-service.git
cd timesfm3-service
```

### 2. Build the Docker Image
```bash
make build
# or directly with docker:
docker build -t timesfm3-service .
```

### 3. Run the Container

#### Option A: Standard Model Run (Loads TimesFM 3.0 weights)
```bash
make run
# or with docker:
docker run -d --name timesfm3-service \
  -p 8000:8000 \
  -v timesfm3_cache:/app/cache \
  -e HF_HOME=/app/cache \
  timesfm3-service
```
*Note: On first startup, the container will download the model weights from Hugging Face (`google/timesfm-3.0-pytorch`) into the persistent volume.*

#### Option B: Mock Mode (Instant Startup for Testing/CI)
```bash
make run-mock
# or with docker:
docker run -d --name timesfm3-service -p 8000:8000 -e MOCK_MODE=true timesfm3-service
```

#### Option C: GPU Accelerated (NVIDIA CUDA)
```bash
make run-gpu
# or with docker:
docker run -d --name timesfm3-service \
  --gpus all \
  -p 8000:8000 \
  -v timesfm3_cache:/app/cache \
  -e DEVICE=cuda \
  timesfm3-service
```

#### Option D: Docker Compose
```bash
docker compose up -d --build
```

---

## 📖 API Documentation

Once the container is running, open your browser to view the interactive documentation:
- **Interactive Documentation (Scalar)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Spec (JSON)**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service liveness/readiness, model load status, device |
| `GET` | `/v1/info` | Service configuration limits, active backend, and runtime metadata |
| `GET` | `/v1/models` | List available models (OpenAI-compatible collection format) |
| `GET` | `/v1/models/{id}` | Retrieve specific model metadata by ID |
| `POST` | `/v1/forecast` | Generate forecasts for univariate/multivariate series |

---

## 💡 Usage Examples

### 1. Single Univariate Series
```bash
curl -X POST "http://localhost:8000/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [10.5, 11.2, 12.0, 11.8, 12.5, 13.1, 13.0, 13.8, 14.2, 14.9],
    "horizon": 5,
    "quantiles": [0.1, 0.5, 0.9]
  }'
```

**Response:**
```json
{
  "point_forecast": [15.25, 15.68, 16.02, 16.45, 16.89],
  "quantiles": {
    "q10": [14.05, 14.42, 14.71, 15.08, 15.48],
    "q50": [15.25, 15.68, 16.02, 16.45, 16.89],
    "q90": [16.45, 16.94, 17.33, 17.82, 18.30]
  },
  "horizon": 5,
  "model_id": "google/timesfm-3.0-pytorch",
  "inference_time_ms": 24.31
}
```

### 2. Batch Forecasting (Multiple Series)
```bash
curl -X POST "http://localhost:8000/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
      [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    ],
    "horizon": 3
  }'
```

### 3. Python Client
A ready-to-run client script is provided in `scripts/test_client.py`:
```bash
python scripts/test_client.py 8000
```

---

## ⚙️ Configuration Reference

Configure the service via environment variables in `docker run -e KEY=VALUE` or `.env`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `google/timesfm-3.0-pytorch` | Hugging Face repo ID or path to checkpoint |
| `DEVICE` | `auto` | Device target: `auto`, `cpu`, `cuda`, or `mps` |
| `MOCK_MODE` | `false` | When `true`, uses lightweight synthetic inference for testing |
| `LAZY_LOAD` | `false` | When `true`, postpones model loading until first request |
| `MAX_CONTEXT` | `1024` | Maximum historical context length |
| `MAX_HORIZON` | `512` | Maximum allowable forecasting horizon |
| `DEFAULT_HORIZON` | `24` | Default horizon if omitted from request |
| `NORMALIZE_INPUTS` | `true` | Apply normalization to input sequences |
| `HF_HOME` | `/app/cache` | Hugging Face model download and cache directory |
| `PORT` | `8000` | Port for the HTTP server |

---

## 🧪 Running Tests

Run the full test suite inside Docker without needing local python dependencies:
```bash
make test
```
Or directly:
```bash
docker run --rm -e MOCK_MODE=true timesfm3-service pytest -v tests/
```

---

## 🛠️ Makefile Commands

| Command | Action |
|---|---|
| `make build` | Build the container image |
| `make run` | Start container with persistent cache volume |
| `make run-mock` | Start container instantly in mock mode |
| `make run-gpu` | Start container with GPU support |
| `make test` | Run pytest suite in container |
| `make stop` | Stop container |
| `make logs` | Stream container logs |
| `make sample-request` | Execute sample curl forecast request |
| `make compose-up` | Start via Docker Compose |
| `make compose-down` | Stop Docker Compose services |

---

## 📄 License
Source code for this webservice wrapper is released under the **Apache-2.0** License.
Please review the [TimesFM License Notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) regarding pretrained weights.
