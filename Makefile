.PHONY: help install install-dev test test-cov lint format typecheck clean build publish pre-commit docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        Install package dependencies"
	@echo "  make install-dev    Install package with development dependencies"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with ruff"
	@echo "  make typecheck     Run type checking with mypy"
	@echo "  make clean         Remove build artifacts and cache files"
	@echo "  make build         Build distribution packages"
	@echo "  make publish       Publish to PyPI"
	@echo "  make pre-commit    Install pre-commit hooks"
	@echo "  make docker-up     Start development environment"
	@echo "  make docker-down   Stop development environment"

# Install dependencies
install:
	pdm install --prod

# Install with development dependencies
install-dev:
	pdm install -G test -G lint

# Run tests
test:
	cd deployment/milvus && docker compose up -d --wait
	cd deployment/pgsql && docker compose up -d --wait
	pdm run pytest

# Run tests with coverage
test-cov:
	pdm run pytest --cov=src/pymilvus_pg --cov-report=term-missing --cov-report=html

# Run linting
lint:
	pdm run ruff check src/ tests/
	pdm run ruff format --check src/ tests/

# Format code
format:
	pdm run ruff check --fix src/ tests/
	pdm run ruff format src/ tests/

# Type checking
typecheck:
	pdm run mypy src/

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name ".pdm-python" -exec rm -rf {} +
	find . -type d -name ".pdm-build" -exec rm -rf {} +

# Build distribution
build: clean
	pdm build

# Publish to PyPI
publish: build
	pdm publish


# Docker commands for development environment
docker-up:
	cd deployment/milvus && docker compose up -d --wait
	cd deployment/pgsql && docker compose up -d --wait

docker-down:
	cd deployment/milvus && docker compose down --volumes
	cd deployment/pgsql && docker compose down --volumes

# Development workflow commands
check: lint typecheck test
	@echo "All checks passed!"

# Quick development setup
setup: install-dev pre-commit
	@echo "Development environment setup complete!"