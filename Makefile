# Sentiment Analysis DeBERTa v3 LoRA - Docker Commands
# =====================================================

# Variables
IMAGE_NAME = sentiment-analysis-deberta3-lora
TAG = latest
AWS_REGION = eu-west-3
AWS_ACCOUNT_ID = $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "your-account-id")
ECR_REPOSITORY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(IMAGE_NAME)

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Docker commands
.PHONY: build
build: ## Build Docker image locally (without AWS credentials - will fail at DVC pull)
	@echo "Building Docker image: $(IMAGE_NAME):$(TAG)"
	@echo "WARNING: This build will fail at DVC pull step without AWS credentials"
	@echo "Use 'make build-with-secrets' or './build-with-secrets.sh' for full build"
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: build-with-secrets
build-with-secrets: ## Build Docker image with AWS credentials (requires environment variables)
	@echo "Building Docker image with AWS credentials: $(IMAGE_NAME):$(TAG)"
	@if [ -z "$(AWS_ACCESS_KEY_ID)" ]; then \
		echo "ERROR: AWS_ACCESS_KEY_ID environment variable is required"; \
		echo "Set it with: export AWS_ACCESS_KEY_ID=your-access-key"; \
		exit 1; \
	fi
	@if [ -z "$(AWS_SECRET_ACCESS_KEY)" ]; then \
		echo "ERROR: AWS_SECRET_ACCESS_KEY environment variable is required"; \
		echo "Set it with: export AWS_SECRET_ACCESS_KEY=your-secret-key"; \
		exit 1; \
	fi
	@if [ -z "$(DVC_BUCKET)" ]; then \
		echo "ERROR: DVC_BUCKET environment variable is required"; \
		echo "Set it with: export DVC_BUCKET=your-bucket-name"; \
		exit 1; \
	fi
	docker build \
		--build-arg AWS_ACCESS_KEY_ID="$(AWS_ACCESS_KEY_ID)" \
		--build-arg AWS_SECRET_ACCESS_KEY="$(AWS_SECRET_ACCESS_KEY)" \
		--build-arg AWS_DEFAULT_REGION="$(AWS_DEFAULT_REGION)" \
		--build-arg DVC_BUCKET="$(DVC_BUCKET)" \
		-t $(IMAGE_NAME):$(TAG) .

.PHONY: run
run: ## Run Docker container locally
	@echo "Running Docker container locally"
	docker run -p 8501:8501 \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		-e AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
		-e DVC_BUCKET=$(DVC_BUCKET) \
		$(IMAGE_NAME):$(TAG)

.PHONY: run-detached
run-detached: ## Run Docker container in background
	@echo "Running Docker container in background"
	docker run -d -p 8501:8501 \
		--name $(IMAGE_NAME)-container \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		-e AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
		-e DVC_BUCKET=$(DVC_BUCKET) \
		$(IMAGE_NAME):$(TAG)

.PHONY: stop
stop: ## Stop running container
	@echo "Stopping container: $(IMAGE_NAME)-container"
	docker stop $(IMAGE_NAME)-container || true
	docker rm $(IMAGE_NAME)-container || true

.PHONY: logs
logs: ## Show container logs
	@echo "Showing logs for: $(IMAGE_NAME)-container"
	docker logs -f $(IMAGE_NAME)-container

.PHONY: shell
shell: ## Open shell in running container
	@echo "Opening shell in: $(IMAGE_NAME)-container"
	docker exec -it $(IMAGE_NAME)-container /bin/bash

# AWS ECR commands
.PHONY: ecr-login
ecr-login: ## Login to AWS ECR
	@echo "Logging in to AWS ECR"
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPOSITORY)

.PHONY: ecr-create-repo
ecr-create-repo: ## Create ECR repository
	@echo "Creating ECR repository: $(IMAGE_NAME)"
	aws ecr create-repository --repository-name $(IMAGE_NAME) --region $(AWS_REGION) || echo "Repository may already exist"

.PHONY: tag-ecr
tag-ecr: ## Tag image for ECR
	@echo "Tagging image for ECR: $(ECR_REPOSITORY):$(TAG)"
	docker tag $(IMAGE_NAME):$(TAG) $(ECR_REPOSITORY):$(TAG)

.PHONY: push-ecr
push-ecr: ecr-login tag-ecr ## Push image to ECR
	@echo "Pushing image to ECR: $(ECR_REPOSITORY):$(TAG)"
	docker push $(ECR_REPOSITORY):$(TAG)

.PHONY: pull-ecr
pull-ecr: ecr-login ## Pull image from ECR
	@echo "Pulling image from ECR: $(ECR_REPOSITORY):$(TAG)"
	docker pull $(ECR_REPOSITORY):$(TAG)

# Development commands
.PHONY: clean
clean: stop ## Clean up Docker resources
	@echo "Cleaning up Docker resources"
	docker rmi $(IMAGE_NAME):$(TAG) || true
	docker system prune -f

.PHONY: test-local
test-local: build run-detached ## Build and test locally
	@echo "Waiting for container to start..."
	@sleep 10
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8501/_stcore/health || echo "Health check failed"
	@echo "Container is running. Visit http://localhost:8501 to test the app"

# Full deployment workflow
.PHONY: deploy
deploy: ecr-create-repo build push-ecr ## Full deployment to ECR
	@echo "Deployment completed successfully!"
	@echo "Image available at: $(ECR_REPOSITORY):$(TAG)"

# Testing and Development commands
.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install -r requirements.txt

.PHONY: test
test: ## Run tests with pytest
	@echo "Running tests..."
	pytest tests/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=60

.PHONY: coverage-report
coverage-report: test-coverage ## Generate and open coverage report
	@echo "Opening coverage report..."
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Coverage report generated in htmlcov/index.html"; \
	fi

.PHONY: lint
lint: ## Run linting with flake8
	@echo "Running flake8 linting..."
	flake8 src/ tests/ app_refactored.py --max-line-length=100 --extend-ignore=E203,W503,E402

.PHONY: format
format: ## Format code with black and isort
	@echo "Formatting code with black..."
	black src/ tests/ app_refactored.py --line-length=100
	@echo "Sorting imports with isort..."
	isort src/ tests/ app_refactored.py --profile black

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "Running type checking..."
	mypy src/ app_refactored.py --ignore-missing-imports

.PHONY: check-all
check-all: lint type-check test ## Run all checks (lint, type-check, test)
	@echo "All checks completed successfully!"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	pre-commit install

.PHONY: pre-commit-run
pre-commit-run: ## Run pre-commit on all files
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

# Environment setup
.PHONY: setup-env
setup-env: ## Show environment setup instructions
	@echo "Environment Variables Required:"
	@echo "  AWS_ACCESS_KEY_ID=your-access-key"
	@echo "  AWS_SECRET_ACCESS_KEY=your-secret-key"
	@echo "  AWS_DEFAULT_REGION=us-east-1"
	@echo "  DVC_BUCKET=your-s3-bucket-name"
	@echo ""
	@echo "Set these in your shell or .env file before running Docker commands."
