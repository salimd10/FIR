# Financial Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for analyzing SEC 10-K filings. Built with FastAPI, Qdrant, and OpenAI, featuring hybrid search, calculator tools, and rigorous evaluation.

## 🎯 Project Overview

This system allows financial analysts to ask complex questions about SEC 10-K filings and receive accurate, cited answers with multi-step calculations. It addresses three key challenges:

1. **Layout-Aware Ingestion** - Preserves table structure and relationships
2. **Hybrid Retrieval** - Combines semantic search with keyword matching (BM25)
3. **Calculator Integration** - Ensures LLM uses tools for calculations, not mental math

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Ingestion  │  │  Retrieval   │  │    Agent     │     │
│  │   Pipeline   │  │   System     │  │ Orchestrator │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│        │                   │                  │             │
│        ├──PDF Parser       ├──Vector Search   ├──LLM       │
│        ├──Table Extract    ├──BM25 Search     ├──Calculator│
│        ├──Chunking         └──Hybrid (RRF)    └──Citations │
│        └──Embeddings                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐        ┌──────────┐        ┌──────────┐
    │ OpenAI  │        │  Qdrant  │        │  Redis   │
    │   API   │        │ Vector DB│        │  Cache   │
    └─────────┘        └──────────┘        └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### Installation

1. **Clone and setup environment:**

```bash
cd /home/salim/Desktop/PROJECTS/personal/FIR
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure environment:**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

4. **Start infrastructure:**

```bash
docker-compose up -d
```

This starts:
- Qdrant vector database (port 6333)
- Redis cache (port 6379)

5. **Verify services:**

```bash
# Check Qdrant
curl http://localhost:6333/collections

# Check Redis
redis-cli ping
```

### Download Apple 10-K

```bash
# Download the Apple 10-K PDF
wget https://s2.q4cdn.com/470004039/files/doc_financials/2025/ar/_10-K-2025-As-Filed.pdf \
     -O data/raw/apple-10k-2025.pdf
```

### Ingest Document

```bash
# Process the 10-K PDF
python src/ingestion/document_loader.py data/raw/apple-10k-2025.pdf
```

This will:
1. Parse the PDF with layout awareness
2. Extract and convert tables to Markdown
3. Chunk content semantically (1024 tokens, 128 overlap)
4. Generate embeddings (OpenAI text-embedding-3-large)
5. Store in Qdrant and build BM25 index

### Start API

```bash
# Development mode
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apples R&D spending in FY2025?",
    "return_sources": true,
    "max_sources": 5
  }'
```

## 📁 Project Structure

```
FIR/
├── src/
│   ├── ingestion/              # Document processing
│   │   ├── pdf_parser.py       # Layout-aware PDF parsing
│   │   ├── table_extractor.py  # Table → Markdown conversion
│   │   ├── chunking_strategy.py # Semantic chunking
│   │   ├── embedding_service.py # OpenAI embeddings
│   │   └── document_loader.py  # Ingestion pipeline
│   │
│   ├── retrieval/              # Search components
│   │   ├── vector_store.py     # Qdrant interface
│   │   ├── bm25_search.py      # Keyword search
│   │   └── hybrid_retriever.py # RRF fusion
│   │
│   ├── agents/                 # RAG agent
│   │   ├── calculator_tool.py  # Python REPL for calculations
│   │   ├── query_expander.py   # Handle vague queries
│   │   └── rag_orchestrator.py # Main agent logic
│   │
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # API routes
│   │   └── models.py           # Pydantic schemas
│   │
│   ├── evaluation/             # Testing & metrics
│   │   ├── golden_dataset.json # 5 test questions
│   │   └── eval_pipeline.py    # RAGAS evaluation
│   │
│   ├── utils/                  # Utilities
│   │   ├── citation_engine.py  # Source tracking
│   │   └── prompts.py          # System prompts
│   │
│   └── config.py               # Configuration
│
├── data/                       # Data storage
│   ├── raw/                    # Original PDFs
│   ├── processed/              # Intermediate outputs
│   └── embeddings/             # Cached embeddings
│
├── tests/                      # Test suite
├── documentation/              # PDF learning guide
├── docker-compose.yml          # Infrastructure
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔍 Key Features

### 1. Layout-Aware Ingestion

**Problem**: Standard PDF parsers lose table structure and relationships.

**Solution**:
- Uses `unstructured.io` with `hi_res` strategy for layout detection
- `pdfplumber` for precise table extraction
- Converts tables to Markdown while preserving headers
- Maintains context with section titles and page numbers

```python
# Example: Table preserved in Markdown
| | 2025 | 2024 | 2023 |
|---|---|---|---|
| Net Sales | $400.0B | $385.0B | $370.0B |
| R&D | $31.4B | $29.9B | $28.5B |
```

### 2. Hybrid Retrieval System

**Components**:
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **BM25 Search**: Keyword matching for exact terms
- **RRF Fusion**: Combines both using Reciprocal Rank Fusion

**Why Hybrid?**
- Vector search: Captures meaning ("R&D" ≈ "Research and Development")
- BM25: Catches exact phrases ("Amortization of intangible assets")
- RRF: Best of both worlds

**RRF Formula**:
```
RRF_score(d) = Σ weight_i × (1 / (k + rank_i(d)))
```
where k=60 (constant), rank_i is position in result list i

### 3. Calculator Tool

**Problem**: LLMs are bad at arithmetic and can hallucinate numbers.

**Solution**:
- Python REPL tool for calculations
- LLM extracts numbers from context
- Generates Python code
- Tool executes code safely
- Result returned to LLM for interpretation

**Example**:
```
Question: "What was the % change in R&D from 2024 to 2025?"

Step 1: LLM extracts: R&D_2024 = 29.9, R&D_2025 = 31.4
Step 2: LLM generates: ((31.4 - 29.9) / 29.9) * 100
Step 3: Calculator executes: 5.0167...
Step 4: LLM formats: "R&D increased by 5.02%"
```

### 4. Citation Engine

Every answer includes:
- Exact page number
- Section title
- Source text snippet
- Relevance score

**Example Citation**:
```
"Apple's R&D expenses were $31.4 billion (Page 39, Consolidated Statement of Operations)."
```

### 5. Hallucination Guardrails

**System Prompt Enforces**:
- Answer ONLY from provided context
- Use calculator for ALL math
- Cite page numbers
- If not found, say "NOT_FOUND" with search details

**Example NOT_FOUND Response**:
```
NOT_FOUND: I searched for information about "employee count in 2025" in sections:
- Business Overview
- Risk Factors
- Financial Statements

This specific metric was not found in the provided 10-K filing.
You may want to ask about headcount growth rate or total compensation instead.
```

## 📊 Evaluation

### Golden Dataset

5 curated questions testing:
1. Multi-step calculations
2. Table navigation
3. GAAP vs Non-GAAP reconciliation
4. Segment analysis
5. Risk factor extraction

### RAGAS Metrics

- **Faithfulness**: Answer grounded in context (target: >0.85)
- **Answer Relevance**: Addresses the question (target: >0.80)
- **Context Recall**: Retrieved chunks contain answer (target: >0.75)
- **Context Precision**: Retrieved chunks are relevant (target: >0.70)

### Run Evaluation

```bash
python src/evaluation/eval_pipeline.py --dataset golden_dataset.json
```

## 🔧 API Reference

### POST /api/query

Ask questions about the 10-K filing.

**Request:**
```json
{
  "question": "What was the R&D growth rate?",
  "return_sources": true,
  "max_sources": 5,
  "top_k": 5
}
```

**Response:**
```json
{
  "query_id": "uuid",
  "question": "...",
  "answer": "...",
  "calculation_steps": [
    {
      "description": "Calculate percentage change",
      "code": "((31.4 - 29.9) / 29.9) * 100",
      "result": "5.02"
    }
  ],
  "citations": [
    {
      "citation_id": 1,
      "text": "Research and development expenses...",
      "page_number": 39,
      "section": "Consolidated Statement",
      "score": 0.94
    }
  ],
  "confidence": 0.89,
  "processing_time_ms": 1247,
  "status": "success"
}
```

### POST /api/documents/upload

Upload a new 10-K PDF.

**Request:**
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@path/to/10k.pdf"
```

### GET /health

Check system health.

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_ingestion/
```

## 📈 Scaling to 10,000 Documents

### Strategy

1. **Distributed Qdrant**
   - Shard collections across multiple nodes
   - Partition by document metadata (year, company, sector)
   - Horizontal scaling for concurrent queries

2. **Async Ingestion Pipeline**
   - Celery task queue for parallel processing
   - Priority queues (new documents first)
   - Batch processing for efficiency

3. **Caching Layer**
   - Redis cache for frequent queries
   - Embedding cache (avoid re-generating)
   - Result cache with TTL

4. **Document Metadata Filtering**
   - Index by: company, year, document type, sector
   - Pre-filter before vector search
   - Reduce search space dramatically

5. **Load Balancing**
   - Multiple API instances behind load balancer
   - Separate read/write Qdrant instances
   - CDN for static assets

6. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert on latency/error spikes

### Architecture at Scale

```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      ┌────▼───┐      ┌────▼───┐      ┌────▼───┐
      │FastAPI │      │FastAPI │      │FastAPI │
      │Instance│      │Instance│      │Instance│
      └────┬───┘      └────┬───┘      └────┬───┘
           │               │               │
           └───────────────┼───────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      ┌────▼───────┐  ┌────▼───────┐  ┌────▼───┐
      │  Qdrant    │  │  Qdrant    │  │ Qdrant │
      │  Shard 1   │  │  Shard 2   │  │ Shard 3│
      └────────────┘  └────────────┘  └─────────┘
           │
      ┌────▼──────────────┐
      │  Redis Cluster    │
      │  (Cache + Queue)  │
      └───────────────────┘
           │
      ┌────▼──────────────┐
      │  Celery Workers   │
      │  (Ingestion)      │
      └───────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| Backend | FastAPI | Async, fast, automatic API docs |
| Vector DB | Qdrant | Lightweight, excellent filtering, scalable |
| Embeddings | OpenAI text-embedding-3-large | High quality (3072 dims), financial text |
| LLM | GPT-4 | Function calling, high reasoning ability |
| Keyword Search | BM25 (rank-bpf) | Industry standard, exact term matching |
| PDF Parsing | Unstructured.io + pdfplumber | Layout awareness, table extraction |
| Evaluation | RAGAS | LLM-as-judge, standard RAG metrics |
| Orchestration | LangChain | Tool integration, agent patterns |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `pytest` and `black src/`
5. Submit pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Apple Inc. for SEC filings
- Anthropic for Claude (used in development)
- OpenAI for GPT-4 and embeddings
- Qdrant team for excellent vector DB

---

**Built with ❤️ for financial analysts and AI engineers**
