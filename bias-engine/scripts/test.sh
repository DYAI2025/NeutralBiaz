#!/bin/bash
# Test script for Bias Engine API

set -e

echo "Running Bias Engine tests..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies including dev dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# Run tests with coverage
echo "Running tests with coverage..."
pytest tests/ -v --tb=short --cov=src/bias_engine --cov-report=html --cov-report=term

echo "Tests completed!"
echo "Coverage report generated in htmlcov/index.html"