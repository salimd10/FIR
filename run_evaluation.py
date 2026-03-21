#!/usr/bin/env python3
"""
CLI script to run RAGAS evaluation on the RAG system.
"""
import sys
from pathlib import Path
import argparse
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import get_settings, EVALUATION_DIR
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.bm25_search import BM25KeywordSearch
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.citation_engine import CitationEngine
from src.agents.rag_orchestrator import RAGOrchestrator
from src.evaluation.eval_pipeline import RAGASEvaluationPipeline


def main():
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on Financial Intelligence RAG system"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to golden dataset (default: src/evaluation/golden_dataset.json)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save evaluation report to file"
    )

    args = parser.parse_args()

    logger.info("Initializing Financial Intelligence RAG Evaluation...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Top-k: {args.top_k}")

    try:
        # Initialize settings
        settings = get_settings()

        # Initialize embedding service
        logger.info("Loading embedding service...")
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )

        # Initialize vector store
        logger.info("Connecting to Qdrant...")
        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_size=embedding_service.get_embedding_dimension()
        )

        # Check if collection exists
        info = vector_store.get_collection_info()
        if info.get("vectors_count", 0) == 0:
            logger.error("Vector store is empty! Please run document ingestion first.")
            logger.error("Run: python src/ingestion/document_loader.py <path_to_pdf>")
            sys.exit(1)

        logger.info(f"Vector store has {info.get('vectors_count', 0)} vectors")

        # Initialize BM25 search
        logger.info("Loading BM25 index...")
        bm25_search = BM25KeywordSearch()

        # Try to load existing BM25 index
        try:
            bm25_search.load_index()
            logger.info(f"BM25 index loaded with {len(bm25_search.documents)} documents")
        except Exception as e:
            logger.error(f"Could not load BM25 index: {e}")
            logger.error("Please run document ingestion first.")
            sys.exit(1)

        # Initialize hybrid retriever
        logger.info("Initializing hybrid retriever...")
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25_search,
            embedding_service=embedding_service,
            rrf_k=settings.rrf_k
        )

        # Initialize citation engine
        citation_engine = CitationEngine()

        # Initialize RAG orchestrator
        logger.info(f"Initializing RAG orchestrator with {args.model}...")
        rag_orchestrator = RAGOrchestrator(
            hybrid_retriever=hybrid_retriever,
            citation_engine=citation_engine,
            llm_model=args.model,
            temperature=0.0
        )

        # Initialize evaluation pipeline
        dataset_path = Path(args.dataset) if args.dataset else None
        logger.info("Initializing evaluation pipeline...")
        eval_pipeline = RAGASEvaluationPipeline(
            rag_orchestrator=rag_orchestrator,
            golden_dataset_path=dataset_path
        )

        # Run full evaluation
        logger.info("Starting evaluation...")
        logger.info("=" * 60)

        report = eval_pipeline.run_full_evaluation(
            top_k=args.top_k,
            save_report=args.save_report
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)

        summary = eval_pipeline.generate_report_summary(report)
        print(summary)

        # Print detailed results
        ragas_scores = report.get("ragas_metrics", {}).get("overall_scores", {})
        if ragas_scores:
            logger.info("\nDetailed RAGAS Metrics:")
            for metric, score in ragas_scores.items():
                logger.info(f"  {metric}: {score:.4f}")

        additional = report.get("additional_metrics", {})
        if additional:
            logger.info("\nAdditional Metrics:")
            logger.info(f"  Success Rate: {additional.get('success_rate', 0):.2%}")
            logger.info(f"  Avg Processing Time: {additional.get('avg_processing_time_ms', 0):.0f}ms")
            logger.info(f"  Avg Confidence: {additional.get('avg_confidence', 0):.3f}")

            calc_usage = additional.get('calculation_tool_usage', {})
            logger.info(f"  Calculator Usage: {calc_usage.get('usage_rate', 0):.2%}")

        if args.save_report:
            logger.info(f"\nDetailed report saved to: {EVALUATION_DIR}")

        logger.info("\n✓ Evaluation complete!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
