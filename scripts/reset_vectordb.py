#!/usr/bin/env python3
"""
Deletes the Qdrant collection and removes the BM25 index.
Run this before re-ingesting with new chunking settings.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/reset_vectordb.py
    PYTHONPATH=. .venv/bin/python scripts/reset_vectordb.py --confirm  # skip prompt
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config import get_settings, PROCESSED_DATA_DIR
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(description="Reset Qdrant collection and BM25 index")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    settings = get_settings()

    if not args.confirm:
        print(f"This will permanently delete:")
        print(f"  - Qdrant collection: '{settings.qdrant_collection_name}'")
        print(f"  - All BM25 index files in: {PROCESSED_DATA_DIR}")
        answer = input("\nProceed? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # Delete Qdrant collection
    logger.info("Connecting to Qdrant...")
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
    vector_store.delete_all()
    logger.info(f"Qdrant collection '{settings.qdrant_collection_name}' deleted.")

    # Delete BM25 index files
    bm25_files = list(PROCESSED_DATA_DIR.glob("*_bm25.pkl"))
    if bm25_files:
        for f in bm25_files:
            f.unlink()
            logger.info(f"Deleted BM25 index: {f.name}")
    else:
        logger.info("No BM25 index files found.")

    logger.info("Reset complete. You can now re-ingest your documents.")


if __name__ == "__main__":
    main()