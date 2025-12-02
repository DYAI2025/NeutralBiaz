#!/bin/bash
# Start script for Bias Engine API

set -e

echo "Starting Bias Engine API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run the server
echo "Starting FastAPI server..."
uvicorn bias_engine.main:app --host 0.0.0.0 --port 8000 --reload