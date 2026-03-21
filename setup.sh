#!/bin/bash

# Financial Intelligence RAG - Setup Script
# This script automates the setup process

set -e  # Exit on error

echo "========================================="
echo "Financial Intelligence RAG - Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.11"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo -e "${RED}✗ Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check Docker Compose
echo "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Setup .env file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠ Please edit .env file and add your OPENAI_API_KEY${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/processed data/embeddings
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Start Docker services
echo ""
echo "Starting Docker services (Qdrant & Redis)..."
docker-compose up -d
echo -e "${GREEN}✓ Docker services started${NC}"

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 5

# Check Qdrant
echo "Checking Qdrant..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is ready${NC}"
else
    echo -e "${RED}✗ Qdrant is not responding${NC}"
fi

# Check Redis
echo "Checking Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is ready${NC}"
else
    echo -e "${RED}✗ Redis is not responding${NC}"
fi

# Check OpenAI API key
echo ""
if grep -q "your_openai_api_key_here" .env; then
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}IMPORTANT: Setup incomplete!${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    echo ""
    echo "Please complete the following steps:"
    echo "1. Edit .env file and add your OPENAI_API_KEY"
    echo "2. Download Apple 10-K PDF:"
    echo "   wget https://s2.q4cdn.com/470004039/files/doc_financials/2025/ar/_10-K-2025-As-Filed.pdf -O data/raw/apple-10k-2025.pdf"
    echo "3. Ingest the document:"
    echo "   python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf"
    echo "4. Start the API:"
    echo "   uvicorn src.api.main:app --reload"
    echo ""
else
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Download Apple 10-K PDF (if not already done):"
    echo "   wget https://s2.q4cdn.com/470004039/files/doc_financials/2025/ar/_10-K-2025-As-Filed.pdf -O data/raw/apple-10k-2025.pdf"
    echo ""
    echo "2. Ingest the document:"
    echo "   source .venv/bin/activate"
    echo "   python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf"
    echo ""
    echo "3. Start the API server:"
    echo "   uvicorn src.api.main:app --reload"
    echo ""
    echo "4. Visit http://localhost:8000/docs for API documentation"
    echo ""
fi

echo "========================================="
echo "For more information, see README.md"
echo "========================================="
