"""
Qdrant vector store interface for semantic search.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)
from loguru import logger
import uuid


class QdrantVectorStore:
    """
    Interface for Qdrant vector database operations.
    Handles storage and retrieval of document embeddings.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "financial_documents",
        vector_size: int = 3072
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            vector_size: Dimension of embedding vectors
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.logger = logger.bind(module="vector_store")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Collection {self.collection_name} created")
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            self.logger.error(f"Error ensuring collection: {str(e)}")
            raise

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add document chunks with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors

        Returns:
            List of generated IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        ids = []

        for chunk, embedding in zip(chunks, embeddings):
            if not embedding:  # Skip empty embeddings
                continue

            point_id = str(uuid.uuid4())
            ids.append(point_id)

            # Prepare payload
            payload = {
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk.get("content", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "page_number": chunk.get("page_number", 0),
                "section_title": chunk.get("section_title", ""),
                "is_table": chunk.get("is_table", False),
                "token_count": chunk.get("token_count", 0),
                "metadata": chunk.get("metadata", {})
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        self.logger.info(f"Added {len(points)} chunks to vector store")
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_params: Optional filters (e.g., page_number, section)

        Returns:
            List of matching chunks with scores
        """
        search_filter = None

        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                search_filter = Filter(must=conditions)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter
            )

            chunks = []
            for result in results:
                chunk = result.payload.copy()
                chunk["score"] = result.score
                chunk["id"] = result.id
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    def delete_all(self):
        """Delete all points from the collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            self.logger.info(f"Deleted all chunks from {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection metadata
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {}


if __name__ == "__main__":
    # Test the vector store
    store = QdrantVectorStore()
    info = store.get_collection_info()
    print(f"Collection info: {info}")
