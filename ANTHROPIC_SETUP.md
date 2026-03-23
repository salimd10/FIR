# Running with Anthropic Claude - Setup Guide

## Overview

The Financial Intelligence RAG system now supports **Anthropic Claude** as the LLM provider! Claude offers excellent reasoning capabilities and cost-effective pricing.

## Quick Start (5 Steps)

### 1. Get API Keys

**Anthropic API Key** (for Claude):
- Visit: https://console.anthropic.com/
- Create account → API Keys → Create new key
- Copy key (starts with `sk-ant-...`)

**OpenAI API Key** (for embeddings only):
- Visit: https://platform.openai.com/api-keys
- Create key (starts with `sk-...`)

### 2. Create `.env` File

```bash
cd /home/salim/Desktop/PROJECTS/personal/FIR
cp .env.example .env
nano .env
```

Add your keys:
```bash
# LLM Provider
LLM_PROVIDER=anthropic

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=financial_documents

# Retrieval Configuration
TOP_K_VECTOR=20
TOP_K_BM25=20
TOP_K_FINAL=5
RRF_K=60

# LLM Configuration
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2000
```

### 3. Start Infrastructure

```bash
# Start Qdrant vector database
docker compose up -d

# Verify Qdrant is running
curl http://localhost:6333
```

### 4. Ingest Documents

```bash
# Add your PDF files to data/raw/, then run ingestion:
PYTHONPATH=. .venv/bin/python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf
```

### 5. Start API Server

```bash
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     RAG orchestrator initialized with anthropic (claude-3-5-sonnet-20241022)
INFO:     Query expander initialized with anthropic (claude-3-5-sonnet-20241022)
INFO:     ✓ All services initialized successfully
```

## Ask Questions

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the total revenue in 2023?",
    "top_k": 5,
    "expand_query": false
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "question": "What are the main risk factors?",
        "top_k": 5,
        "expand_query": True  # Let Claude expand vague queries
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

### Interactive Docs

Open http://localhost:8000/docs and try the `/api/query` endpoint.

## Model Selection

### Recommended Models

**For Production (best quality):**
```bash
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```
- Best reasoning and analysis
- $3 per million input tokens
- $15 per million output tokens

**For Development (faster, cheaper):**
```bash
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```
- Fast responses
- $0.80 per million input tokens
- $4 per million output tokens

### Switching to OpenAI

Edit `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

Restart API server.

## Features Enabled with Claude

### 1. Answer Generation
Claude analyzes retrieved financial documents and provides detailed answers with citations.

### 2. Calculator Tool
Claude can use the Python calculator tool for accurate financial computations:
```json
{
  "question": "What was the year-over-year revenue growth rate?",
  "answer": "The revenue grew from $394.3B to $383.3B. Using the calculator: ((383.3 - 394.3) / 394.3) * 100 = -2.79%, indicating a 2.79% decline."
}
```

### 3. Query Expansion
For vague queries, Claude expands them into specific sub-questions:
```json
{
  "question": "How is Apple doing?",
  "expand_query": true
}
```
Claude expands to:
- "What was Apple's total revenue in the most recent fiscal year?"
- "What is Apple's net income and profit margin?"
- "What are Apple's main product revenue segments?"

### 4. Citation Tracking
All answers include source citations with page numbers and relevance scores.

## Cost Comparison

For 1000 queries/day (30,000/month):

| Provider | Model | Monthly Cost |
|----------|-------|--------------|
| Anthropic | Claude 3.5 Sonnet | ~$90 |
| Anthropic | Claude 3.5 Haiku | ~$15 |
| OpenAI | GPT-4o | ~$150 |
| OpenAI | GPT-4o-mini | ~$30 |

**Embeddings (required for all):** ~$10/month

## Troubleshooting

### Error: "anthropic_api_key not found"

**Solution:**
```bash
# Check .env file has the key
cat .env | grep ANTHROPIC_API_KEY

# Make sure key is valid (should start with sk-ant-)
# Restart API server
```

### Error: "No module named 'langchain_anthropic'"

**Solution:**
```bash
source .venv/bin/activate
pip install langchain-anthropic
```

### Low quality answers

**Solutions:**
1. Increase `top_k` to retrieve more context (try 10-15)
2. Enable query expansion for vague questions
3. Use Claude 3.5 Sonnet instead of Haiku
4. Check if documents were properly ingested

### API is slow

**Solutions:**
1. Switch to Claude 3.5 Haiku (much faster)
2. Reduce `top_k` to 3-5
3. Disable query expansion for specific questions
4. Add caching (future enhancement)

## Performance Tips

### For Best Quality:
- Model: `claude-3-5-sonnet-20241022`
- Top K: 10-15
- Query expansion: Enabled
- Temperature: 0 (deterministic)

### For Speed:
- Model: `claude-3-5-haiku-20241022`
- Top K: 3-5
- Query expansion: Disabled
- Temperature: 0

### For Cost:
- Model: `claude-3-5-haiku-20241022`
- Cache frequent queries (future)
- Use smaller `max_tokens` (500-1000)

## Architecture

```
User Question
    ↓
FastAPI Endpoint
    ↓
Query Expander (Claude) ──→ Expand vague queries
    ↓
Hybrid Retriever
    ├── Vector Search (Qdrant)
    └── BM25 Search
    ↓ (RRF Fusion)
Top K Chunks (5-15)
    ↓
RAG Orchestrator
    ├── Claude 3.5 Sonnet/Haiku
    ├── Calculator Tool (Python REPL)
    └── Citation Engine
    ↓
Answer + Citations + Confidence
```

## Example Response

```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What was Apple's R&D spending in 2023?",
  "answer": "Based on the consolidated statement of operations, Apple's research and development expenses for fiscal year 2023 were $29.9 billion, representing approximately 7.8% of total net sales.",
  "citations": [
    {
      "source": "apple-10k-2023.pdf",
      "page": 28,
      "relevance_score": 0.94,
      "text": "Research and development expenses increased $3.1 billion, or 12%, to $29.9 billion..."
    }
  ],
  "confidence": 0.92,
  "processing_time_ms": 1247
}
```

## Next Steps

1. **Read the full guide:** `QUICKSTART.md`
2. **Explore the API:** http://localhost:8000/docs
3. **Run evaluation:** `python run_evaluation.py`
4. **Generate learning guide:** `python generate_learning_guide.py`

## Advantages of Claude

✅ **Better reasoning:** Excels at financial analysis
✅ **Lower cost:** ~50% cheaper than GPT-4
✅ **Longer context:** 200K tokens (future use)
✅ **Safety:** Strong guardrails for financial advice
✅ **Tool use:** Excellent with calculator tool

## Support

- **Anthropic Docs:** https://docs.anthropic.com/
- **LangChain Anthropic:** https://python.langchain.com/docs/integrations/chat/anthropic
- **Project Docs:** See `README.md` and `QUICKSTART.md`
