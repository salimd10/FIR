#!/bin/bash
# Run tests for Financial Intelligence RAG system

echo "========================================="
echo "Running Financial Intelligence RAG Tests"
echo "========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source .venv/bin/activate
fi

# Run pytest with coverage
echo ""
echo "Running unit tests..."
pytest tests/ -v --tb=short

echo ""
echo "========================================="
echo "Test run complete"
echo "========================================="
