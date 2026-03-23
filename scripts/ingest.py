#!/usr/bin/env python3
"""
Ingest a PDF document into Qdrant and build the BM25 index.
Wrapper around src/ingestion/document_loader.py with helpful output.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/ingest.py data/raw/apple-10k-2025.pdf
    PYTHONPATH=. .venv/bin/python scripts/ingest.py data/raw/apple-10k-2025.pdf --verify
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config import get_settings, PROCESSED_DATA_DIR
from src.ingestion.document_loader import DocumentIngestionPipeline
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG system")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After ingestion, print collection stats"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    settings = get_settings()
    logger.info(f"Ingesting: {pdf_path.name}")
    logger.info(f"Chunk size: {settings.chunk_size} tokens | Overlap: {settings.chunk_overlap} tokens")

    # Run ingestion
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest(str(pdf_path))

    if args.verify:
        logger.info("Verifying ingestion...")
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
        logger.info(f"Qdrant collection '{settings.qdrant_collection_name}': {vectors_count} vectors")

        bm25_files = sorted(PROCESSED_DATA_DIR.glob("*_bm25.pkl"))
        if bm25_files:
            logger.info(f"BM25 index: {bm25_files[-1].name}")
        else:
            logger.warning("No BM25 index found after ingestion.")

        if vectors_count < 100:
            logger.warning(
                f"Only {vectors_count} vectors — chunk size may be too large. "
                f"Consider reducing CHUNK_SIZE in .env and re-ingesting."
            )
        else:
            logger.info("Ingestion looks healthy.")


if __name__ == "__main__":
    main()
