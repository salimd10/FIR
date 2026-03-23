#!/bin/bash
# Run tests for Financial Intelligence RAG system

echo "========================================="
echo "Running Financial Intelligence RAG Tests"
echo "========================================="

# Resolve pytest from the virtualenv
if [ -f ".venv/bin/pytest" ]; then
    PYTEST=".venv/bin/pytest"
else
    PYTEST="pytest"
fi

# Run pytest
echo ""
echo "Running unit tests..."
PYTHONPATH=. $PYTEST tests/ -v --tb=short

echo ""
echo "========================================="
echo "Test run complete"
echo "========================================="
