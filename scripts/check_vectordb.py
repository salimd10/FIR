#!/usr/bin/env python3
"""
Print current stats for the Qdrant collection and BM25 index.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_vectordb.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config import get_settings, PROCESSED_DATA_DIR
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.bm25_search import BM25KeywordSearch


def main():
    settings = get_settings()

    # Qdrant stats
    logger.info("Checking Qdrant...")
    embedding_service = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model
    )
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=embedding_service.get_embedding_dimension()
    )
    info = vector_store.get_collection_info()
    vectors_count = info.get("vectors_count", 0)

    print("\n========== Vector DB ==========")
    print(f"Collection : {settings.qdrant_collection_name}")
    print(f"Vectors    : {vectors_count}")
    print(f"Status     : {info.get('status', 'unknown')}")
    if vectors_count == 0:
        print("WARNING    : Collection is empty — run scripts/ingest.py first")
    elif vectors_count < 100:
        print(f"WARNING    : Only {vectors_count} vectors — chunk size may be too large")
    else:
        print("Health     : OK")

    # BM25 stats
    print("\n========== BM25 Index ==========")
    bm25_files = sorted(PROCESSED_DATA_DIR.glob("*_bm25.pkl"))
    if not bm25_files:
        print("Status     : No BM25 index found")
        print("WARNING    : Run scripts/ingest.py to build the index")
    else:
        latest = bm25_files[-1]
        bm25 = BM25KeywordSearch()
        bm25.load_index(latest)
        print(f"Index file : {latest.name}")
        print(f"Documents  : {len(bm25.chunks)}")
        print("Health     : OK")

    # Config summary
    print("\n========== Current Config ==========")
    print(f"Chunk size : {settings.chunk_size} tokens")
    print(f"Overlap    : {settings.chunk_overlap} tokens")
    print(f"LLM        : {settings.llm_provider} / {settings.openai_model}")
    print(f"Embedding  : {settings.openai_embedding_model}")
    print()


if __name__ == "__main__":
    main()
