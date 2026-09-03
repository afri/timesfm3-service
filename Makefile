.PHONY: help build run stop restart logs test sample-request compose-up compose-down clean

IMAGE_NAME ?= timesfm3-service
TAG ?= latest
PORT ?= 8000
CONTAINER_NAME ?= timesfm3-service-container

help:
	@echo "TimesFM 3 Web Service - Makefile"
	@echo "================================"
	@echo "Docker commands (minimum host prerequisites: git, docker):"
	@echo "  make build          Build the Docker container image"
	@echo "  make run            Run the Docker container in the background"
	@echo "  make run-mock       Run in MOCK_MODE (instant startup, no heavy model download)"
	@echo "  make test           Run test suite inside the Docker container"
	@echo "  make stop           Stop the running Docker container"
	@echo "  make logs           View live logs from the container"
	@echo "  make sample-request Send a sample forecasting request using curl"
	@echo "  make compose-up     Start using Docker Compose with persistent cache"
	@echo "  make compose-down   Stop Docker Compose services"
	@echo "  make clean          Remove stopped containers and temporary build files"

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

run:
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v timesfm3_cache:/app/cache \
		-e HF_HOME=/app/cache \
		$(IMAGE_NAME):$(TAG)
	@echo "Service started at http://localhost:$(PORT) (docs at http://localhost:$(PORT)/docs)"

run-mock:
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-e MOCK_MODE=true \
		$(IMAGE_NAME):$(TAG)
	@echo "Service started in MOCK_MODE at http://localhost:$(PORT)"

run-gpu:
	docker run -d --name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):8000 \
		-v timesfm3_cache:/app/cache \
		-e HF_HOME=/app/cache \
		-e DEVICE=cuda \
		$(IMAGE_NAME):$(TAG)
	@echo "GPU accelerated service started at http://localhost:$(PORT)"

stop:
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true

restart: stop run

logs:
	docker logs -f $(CONTAINER_NAME)

test:
	docker run --rm \
		-e MOCK_MODE=true \
		$(IMAGE_NAME):$(TAG) \
		pytest -v tests/

sample-request:
	@chmod +x scripts/sample_request.sh
	./scripts/sample_request.sh $(PORT)

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

clean: stop
	docker container prune -f
