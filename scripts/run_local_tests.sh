#!/bin/bash

# Local test runner script for pymilvus-pg
# This script runs tests that don't require external services

set -e

echo "🧪 Running PyMilvus-PG Local Tests"
echo "=================================="

# Check if PDM is installed
if ! command -v pdm &> /dev/null; then
    echo "❌ PDM is not installed. Please install PDM first:"
    echo "   pip install pdm"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pdm install --dev

# Run unit tests (no external services required)
echo ""
echo "🔍 Running unit tests..."
pdm run pytest tests/test_logger_config.py -v

echo ""
echo "🛠️  Running utility tests..."
pdm run pytest tests/test_utils.py -v

# Run linting
echo ""
echo "🔧 Running code linting..."
pdm run ruff check src/ tests/ || echo "⚠️  Linting issues found (run 'make lint-fix' to fix)"

# Run formatting check
echo ""
echo "🎨 Checking code formatting..."
pdm run ruff format --check src/ tests/ || echo "⚠️  Formatting issues found (run 'make format' to fix)"

echo ""
echo "✅ Local tests completed!"
echo ""
echo "To run integration tests (requires Milvus and PostgreSQL):"
echo "  make test-integration"
echo ""
echo "To run all tests:"
echo "  make test" 