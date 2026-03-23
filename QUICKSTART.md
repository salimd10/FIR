# Financial Intelligence RAG - Quick Start Guide

This guide will get you up and running in 10 minutes.

## Prerequisites Check

```bash
# Check Python (need 3.11+)
python3 --version

# Check Docker
docker --version
docker-compose --version

# Check if you have API keys
echo $ANTHROPIC_API_KEY  # For Claude (recommended)
echo $OPENAI_API_KEY     # Required for embeddings
```

## Configure API Keys

### Get Your API Keys

1. **Anthropic API Key** (recommended for LLM):
   - Visit: https://console.anthropic.com/
   - Create an account
   - Go to "API Keys"
   - Create a new key
   - Copy the key (starts with `sk-ant-...`)

2. **OpenAI API Key** (required for embeddings):
   - Visit: https://platform.openai.com/api-keys
   - Create an account
   - Create a new API key
   - Copy the key (starts with `sk-...`)

### Set Environment Variables

Edit `.env` file:
```bash
# LLM Provider
LLM_PROVIDER=anthropic  # or "openai"

# Anthropic (for Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI (required for embeddings)
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

**Note:** You can switch between providers by changing `LLM_PROVIDER` to `anthropic` or `openai`.

## Installation (3 minutes)

### Option 1: Automated Setup (Recommended)

```bash
cd /home/salim/Desktop/PROJECTS/personal/FIR
chmod +x setup.sh
./setup.sh
```

The script will:
- Create virtual environment
- Install dependencies
- Start Docker services (Qdrant + Redis)
- Create configuration files

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Install dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
#   - ANTHROPIC_API_KEY (for Claude LLM)
#   - OPENAI_API_KEY (for embeddings)

# 4. Start Docker services
docker compose up -d

# 5. Verify services
curl http://localhost:6333/collections              # Qdrant
docker exec redis_financial_rag redis-cli ping     # Redis
```

## Download Data (2 minutes)

```bash
# Download Apple 10-K filing
wget https://s2.q4cdn.com/470004039/files/doc_financials/2025/ar/_10-K-2025-As-Filed.pdf \
     -O data/raw/apple-10k-2025.pdf

# Verify download
ls -lh data/raw/
```

## Ingest Document (5 minutes)

```bash
# Run ingestion pipeline (PYTHONPATH=. is required so Python can find the src package)
PYTHONPATH=. .venv/bin/python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf
```

This will:
1. Parse the PDF (layout-aware)
2. Extract and convert tables to Markdown
3. Chunk content (1024 tokens, 128 overlap)
4. Generate embeddings (OpenAI text-embedding-3-large)
5. Store in Qdrant and build BM25 index

**Expected Output**:
```
[INFO] Starting ingestion of apple-10k-2025.pdf
[INFO] Step 1/5: Parsing PDF...
[INFO] Step 2/5: Processing tables...
[INFO] Step 3/5: Chunking document...
[INFO] Step 4/5: Generating embeddings for 847 chunks...
[INFO] Step 5/5: Storing in vector database...
[INFO] ✓ Successfully ingested apple-10k-2025.pdf
  - 121 pages
  - 847 chunks
  - 23 tables
```

## Start API Server (<1 minute)

```bash
# Development mode (with auto-reload)
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
# PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## Test the System (2 minutes)

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "qdrant_connected": true,
  "embeddings_available": true
}
```

### 2. Check Vector Store

```bash
curl http://localhost:8000/api/collection/info
```

**Expected Response**:
```json
{
  "name": "financial_documents",
  "vectors_count": 847,
  "points_count": 847,
  "status": "green"
}
```

### 3. Ask a Question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apples R&D spending in FY2025?",
    "return_sources": true,
    "max_sources": 3
  }'
```

**Expected Response**:
```json
{
  "query_id": "abc-123-def",
  "question": "What was Apples R&D spending in FY2025?",
  "answer": "Based on the retrieved context...",
  "citations": [
    {
      "citation_id": 1,
      "text": "Research and development expenses...",
      "page_number": 39,
      "section": "Consolidated Statement of Operations",
      "score": 0.94
    }
  ],
  "confidence": 0.89,
  "processing_time_ms": 1247,
  "status": "success"
}
```

### 4. Interactive API Documentation

Open your browser and visit:
```
http://localhost:8000/docs
```

This gives you:
- Interactive API testing (Swagger UI)
- Request/response schemas
- Try out different queries

## Example Queries to Try

### Simple Factual Question
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apples main product categories?"}'
```

### Calculation Question (if full RAG orchestrator implemented)
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the year-over-year growth rate in R&D expenses?"}'
```

### Table-Heavy Question
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare net sales across different geographic segments"}'
```

### Risk Factor Question
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main supply chain risks mentioned?"}'
```

## Troubleshooting

### Issue: "Connection refused" when accessing API

**Solution**:
```bash
# Check if API is running
ps aux | grep uvicorn

# Check logs
tail -f logs/api.log

# Restart API
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Issue: "ANTHROPIC_API_KEY not found" or "OPENAI_API_KEY not found"

**Solution**:
```bash
# Edit .env file
nano .env

# Add your keys
ANTHROPIC_API_KEY=sk-ant-...  # For Claude
OPENAI_API_KEY=sk-...         # For embeddings

# Restart API
```

### Issue: "Qdrant connection failed"

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant

# Check Qdrant logs
docker logs qdrant_financial_rag
```

### Issue: "No results found"

**Solution**:
```bash
# Check if document was ingested
curl http://localhost:8000/api/collection/info

# If vectors_count is 0, re-run ingestion:
PYTHONPATH=. .venv/bin/python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf
```

## Next Steps

### 1. Explore the Codebase

Key files to review:
- `src/api/main.py` - FastAPI application
- `src/ingestion/pdf_parser.py` - PDF parsing logic
- `src/retrieval/hybrid_retriever.py` - Search implementation
- `src/agents/calculator_tool.py` - Calculator for financial math

### 2. Read Documentation

- `README.md` - Full project documentation
- `SYSTEM_OVERVIEW.md` - Interview preparation guide
- `docker-compose.yml` - Infrastructure setup

### 3. Customize Configuration

Edit `.env` to tune:
- Chunk size (`CHUNK_SIZE=1024`)
- Retrieval parameters (`TOP_K_VECTOR=20`, `TOP_K_BM25=20`)
- LLM settings (`LLM_TEMPERATURE=0`)

### 4. Add More Documents

```bash
# Download another 10-K
wget <url> -O data/raw/another-10k.pdf

# Ingest it
PYTHONPATH=. .venv/bin/python src/ingestion/document_loader.py data/raw/another-10k.pdf
```

### 5. Run Evaluation (Future Implementation)

```bash
python src/evaluation/eval_pipeline.py --dataset golden_dataset.json
```

## Architecture Overview

```
User Query
    ↓
FastAPI (/api/query)
    ↓
Hybrid Retriever
    ├── Vector Search (Qdrant)
    └── BM25 Search (In-memory)
    ↓ (RRF Fusion)
Top 5 Chunks
    ↓
RAG Orchestrator
    ├── LLM (Claude 3.5 Sonnet or GPT-4)
    ├── Calculator Tool
    └── Citation Engine
    ↓
Response with Sources
```

## Performance Expectations

### Ingestion
- **Small doc (50 pages)**: ~2-3 minutes
- **Large doc (150 pages)**: ~5-10 minutes
- **Cost**: ~$0.50 per document (OpenAI embeddings)

### Querying
- **Simple query**: <1 second
- **Complex query**: 1-3 seconds
- **Cost**: ~$0.01 per query (if using LLM)

### Storage
- **Vector DB**: ~1MB per 1000 chunks
- **BM25 Index**: ~500KB per 1000 chunks

## Development Workflow

### Make Changes

```bash
# 1. Edit code
nano src/api/main.py

# 2. API auto-reloads (if using --reload)

# 3. Test changes
curl -X POST http://localhost:8000/api/query ...
```

### Stop Services

```bash
# Stop API
Ctrl+C

# Stop Docker services
docker compose down

# Or keep data volumes
docker compose down --volumes
```

### Start Services Again

```bash
# Start Docker
docker compose up -d

# Start API
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment Checklist

- [ ] Set `DEBUG=False` in `.env`
- [ ] Use production-grade WSGI server (Gunicorn + Uvicorn workers)
- [ ] Setup reverse proxy (Nginx)
- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Configure CORS appropriately
- [ ] Setup monitoring (Prometheus + Grafana)
- [ ] Enable logging aggregation (ELK stack)
- [ ] Setup backup strategy for Qdrant data
- [ ] Implement rate limiting
- [ ] Add authentication (API keys or OAuth)

## Cost Estimation (Monthly)

For moderate usage (1000 queries/day):

| Component | Cost |
|-----------|------|
| OpenAI Embeddings (ingestion) | $10 |
| Claude 3.5 Sonnet (queries) | $90 |
| Server (2 vCPU, 8GB RAM) | $50 |
| Qdrant Cloud (optional) | $50 |
| **Total** | **~$200/month** |

To reduce costs:
- Use Claude 3.5 Haiku instead of Sonnet ($15 vs $90)
- Use GPT-4o-mini instead of Claude ($30 vs $90)
- Self-host Qdrant (save $50)
- Cache frequent queries (reduce API calls)
- Batch process embeddings (already implemented)

## Getting Help

- **Documentation**: See `README.md` and `SYSTEM_OVERVIEW.md`
- **Issues**: Check logs in `logs/api.log`
- **API Docs**: http://localhost:8000/docs
- **Qdrant UI**: http://localhost:6333/dashboard

## Summary

You now have a working Financial Intelligence RAG system! 🎉

**What you built**:
- ✅ Layout-aware PDF ingestion pipeline
- ✅ Hybrid search (vector + BM25)
- ✅ FastAPI backend with async support
- ✅ Citation tracking for source attribution
- ✅ Production-ready infrastructure (Docker)

**What's already built**:
- ✅ Full RAG orchestrator with Claude/GPT-4
- ✅ Query expansion for vague questions
- ✅ RAGAS evaluation pipeline
- ✅ Calculator tool for financial computations

**Future enhancements**:
- [ ] Streaming responses
- [ ] Multi-document support
- [ ] User feedback collection
- [ ] Caching layer

Happy coding! 🚀
