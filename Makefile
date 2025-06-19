.PHONY: help install test test-unit test-integration test-performance lint format type-check clean build publish docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pdm install --dev

test: ## Run all tests
	pdm run pytest tests/ -v

test-unit: ## Run unit tests only
	pdm run pytest tests/test_logger_config.py tests/test_utils.py -v

test-integration: ## Run integration tests
	pdm run pytest tests/test_milvus_pg_client.py tests/test_integration.py -v -m integration

test-performance: ## Run performance tests
	pdm run pytest tests/test_performance.py -v -m slow

test-quick: ## Run quick tests (no external services required)
	pdm run pytest tests/test_logger_config.py tests/test_utils.py -v

test-coverage: ## Run tests with coverage
	pdm run pytest tests/ --cov=src/pymilvus_pg --cov-report=html --cov-report=term

lint: ## Run linting
	pdm run ruff check src/ tests/

lint-fix: ## Run linting with auto-fix
	pdm run ruff check src/ tests/ --fix

format: ## Format code
	pdm run ruff format src/ tests/

format-check: ## Check code formatting
	pdm run ruff format --check src/ tests/

type-check: ## Run type checking
	pdm run mypy src/pymilvus_pg/ --ignore-missing-imports

clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	pdm build

publish: ## Publish to PyPI (requires authentication)
	pdm publish

publish-test: ## Publish to TestPyPI
	pdm publish --repository testpypi

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

pre-commit: ## Run all pre-commit checks
	make lint
	make format-check
	make type-check
	make test-unit

setup-dev: ## Setup development environment
	pip install pdm
	pdm install --dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test-quick' to verify everything is working"

# Docker-related targets
docker-test: ## Run tests in Docker environment
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

docker-test-down: ## Stop Docker test environment
	docker-compose -f docker-compose.test.yml down -v 