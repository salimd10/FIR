"""
Hybrid retrieval combining vector search and BM25.
Uses Reciprocal Rank Fusion (RRF) for optimal results.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np


class HybridRetriever:
    """
    Hybrid retrieval system combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from:
    - Vector search (semantic similarity)
    - BM25 search (keyword matching)
    """

    def __init__(
        self,
        vector_store,
        bm25_search,
        embedding_service,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: QdrantVectorStore instance
            bm25_search: BM25KeywordSearch instance
            embedding_service: EmbeddingService instance
            rrf_k: RRF constant (typically 60)
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.logger = logger.bind(module="hybrid_retriever")

    def retrieve(
        self,
        query: str,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            top_k_vector: Number of results from vector search
            top_k_bm25: Number of results from BM25 search
            top_k_final: Final number of results to return
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores

        Returns:
            List of ranked chunks with RRF scores
        """
        self.logger.info(f"Hybrid retrieval for query: {query[:100]}...")

        # 1. Vector Search
        query_embedding = self.embedding_service.generate_embedding(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k_vector
        )

        # 2. BM25 Search
        bm25_results = self.bm25_search.search(
            query=query,
            top_k=top_k_bm25
        )

        # 3. Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )

        # 4. Return top-k results
        return fused_results[:top_k_final]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF Formula: RRF_score = Σ weight_i * (1 / (k + rank_i))

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results

        Returns:
            Fused and ranked results
        """
        # Create a dictionary to accumulate scores
        chunk_scores = {}
        chunk_data = {}
        chunk_sources = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.get("chunk_id", result.get("id", ""))

            if chunk_id:
                rrf_score = vector_weight / (self.rrf_k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
                chunk_data[chunk_id] = result
                chunk_sources[chunk_id] = chunk_sources.get(chunk_id, {})
                chunk_sources[chunk_id]["vector"] = {
                    "rank": rank,
                    "score": result.get("score", 0)
                }

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.get("chunk_id", result.get("id", ""))

            if chunk_id:
                rrf_score = bm25_weight / (self.rrf_k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

                # Use BM25 result if not already stored
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result

                chunk_sources[chunk_id] = chunk_sources.get(chunk_id, {})
                chunk_sources[chunk_id]["bm25"] = {
                    "rank": rank,
                    "score": result.get("score", 0)
                }

        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Prepare final results
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            chunk = chunk_data[chunk_id].copy()
            chunk["rrf_score"] = rrf_score
            chunk["sources"] = chunk_sources[chunk_id]
            results.append(chunk)

        self.logger.info(f"RRF fusion produced {len(results)} unique results")
        return results

    def retrieve_with_reranking(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_final: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and optionally re-rank results.

        Args:
            query: Search query
            top_k_retrieve: Number of initial results
            top_k_final: Final number after re-ranking

        Returns:
            Re-ranked results
        """
        # Get initial results
        results = self.retrieve(
            query=query,
            top_k_vector=top_k_retrieve,
            top_k_bm25=top_k_retrieve,
            top_k_final=top_k_retrieve
        )

        # Could add cross-encoder re-ranking here
        # For now, just return top-k from RRF
        return results[:top_k_final]

    def get_context_for_rag(
        self,
        query: str,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> str:
        """
        Get formatted context string for RAG.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        results = self.retrieve(
            query=query,
            top_k_final=top_k
        )

        context_parts = []

        for idx, result in enumerate(results, start=1):
            if include_metadata:
                metadata = (
                    f"[Source {idx} - "
                    f"Page {result.get('page_number', 'N/A')}, "
                    f"Section: {result.get('section_title', 'N/A')}]"
                )
                context_parts.append(metadata)

            context_parts.append(result.get("content", ""))
            context_parts.append("")  # Empty line between sources

        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test would require initialized components
    print("Hybrid retriever module loaded successfully")
    print("RRF formula: 1 / (k + rank) where k typically = 60")
