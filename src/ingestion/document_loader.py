"""
Complete document ingestion pipeline.
Orchestrates parsing, chunking, embedding, and storage.
"""
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
from dataclasses import asdict

from src.ingestion.pdf_parser import FinancialPDFParser
from src.ingestion.table_extractor import FinancialTableProcessor
from src.ingestion.chunking_strategy import FinancialDocumentChunker
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.bm25_search import BM25KeywordSearch
from src.config import get_settings, RAW_DATA_DIR, PROCESSED_DATA_DIR


class DocumentIngestionPipeline:
    """
    End-to-end pipeline for ingesting financial documents.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        bm25_search: BM25KeywordSearch
    ):
        """
        Initialize ingestion pipeline.

        Args:
            embedding_service: Embedding service instance
            vector_store: Vector store instance
            bm25_search: BM25 search instance
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.logger = logger.bind(module="ingestion_pipeline")

        self.parser = FinancialPDFParser()
        self.table_processor = FinancialTableProcessor()
        self.chunker = FinancialDocumentChunker()

    def ingest_document(
        self,
        pdf_path: Path,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF document through the complete pipeline.

        Args:
            pdf_path: Path to PDF file
            save_intermediate: Whether to save intermediate results

        Returns:
            Ingestion results dictionary
        """
        self.logger.info(f"Starting ingestion of {pdf_path.name}")

        try:
            # Step 1: Parse PDF
            self.logger.info("Step 1/5: Parsing PDF...")
            parsed_doc = self.parser.parse_document(pdf_path)

            if save_intermediate:
                self._save_parsed_doc(pdf_path.stem, parsed_doc)

            # Step 2: Process tables
            self.logger.info("Step 2/5: Processing tables...")
            for element in parsed_doc["content"]:
                if element.get("is_table"):
                    self.table_processor.process_table_element(element)

            # Step 3: Chunk document
            self.logger.info("Step 3/5: Chunking document...")
            chunks = self.chunker.chunk_document(parsed_doc)
            chunk_dicts = [asdict(chunk) for chunk in chunks]

            if save_intermediate:
                self._save_chunks(pdf_path.stem, chunk_dicts)

            # Step 4: Generate embeddings
            self.logger.info(f"Step 4/5: Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(
                texts=chunk_texts,
                batch_size=100
            )

            # Step 5: Store in vector DB and BM25
            self.logger.info("Step 5/5: Storing in vector database and BM25 index...")
            chunk_ids = self.vector_store.add_chunks(chunk_dicts, embeddings)

            # Build BM25 index
            self.bm25_search.index_chunks(chunk_dicts)

            # Save BM25 index
            bm25_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_bm25.pkl"
            self.bm25_search.save_index(bm25_path)

            results = {
                "success": True,
                "document": pdf_path.name,
                "total_pages": parsed_doc["metadata"]["total_pages"],
                "total_elements": parsed_doc["total_elements"],
                "tables_found": parsed_doc["tables_found"],
                "total_chunks": len(chunks),
                "chunk_ids": chunk_ids,
                "embeddings_generated": len(embeddings)
            }

            self.logger.info(f"✓ Successfully ingested {pdf_path.name}")
            self.logger.info(f"  - {results['total_pages']} pages")
            self.logger.info(f"  - {results['total_chunks']} chunks")
            self.logger.info(f"  - {results['tables_found']} tables")

            return results

        except Exception as e:
            self.logger.error(f"Error ingesting {pdf_path.name}: {str(e)}")
            raise

    def _save_parsed_doc(self, doc_name: str, parsed_doc: Dict[str, Any]):
        """Save parsed document to JSON."""
        output_path = PROCESSED_DATA_DIR / f"{doc_name}_parsed.json"
        with open(output_path, 'w') as f:
            json.dump(parsed_doc, f, indent=2, default=str)
        self.logger.info(f"Saved parsed document to {output_path}")

    def _save_chunks(self, doc_name: str, chunks: list):
        """Save chunks to JSON."""
        output_path = PROCESSED_DATA_DIR / f"{doc_name}_chunks.json"
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2, default=str)
        self.logger.info(f"Saved chunks to {output_path}")


def ingest_apple_10k(
    pdf_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to ingest Apple 10-K.

    Args:
        pdf_path: Path to Apple 10-K PDF

    Returns:
        Ingestion results
    """
    settings = get_settings()

    if pdf_path is None:
        # Look for PDF in raw data directory
        pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {RAW_DATA_DIR}. "
                "Please download the Apple 10-K PDF first."
            )
        pdf_path = pdf_files[0]

    # Initialize services
    embedding_service = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        cache_dir=PROCESSED_DATA_DIR / "embeddings"
    )

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=embedding_service.get_embedding_dimension()
    )

    bm25_search = BM25KeywordSearch()

    # Create pipeline
    pipeline = DocumentIngestionPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_search=bm25_search
    )

    # Ingest document
    return pipeline.ingest_document(pdf_path)


if __name__ == "__main__":
    import sys

    # Check if PDF path is provided
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = None

    try:
        results = ingest_apple_10k(pdf_path)
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
