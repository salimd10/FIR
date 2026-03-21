"""
Embedding generation service using OpenAI API.
Handles batch processing and caching for efficiency.
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger
import numpy as np
from pathlib import Path
import pickle
import hashlib


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI's embedding models.
    Includes caching and batch processing capabilities.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize embedding service.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            cache_dir: Directory for caching embeddings
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        self.logger = logger.bind(module="embedding_service")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []

        # Check cache
        if use_cache and self.cache_dir:
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Generate embedding
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding

            # Cache the embedding
            if use_cache and self.cache_dir:
                self._save_to_cache(cache_key, embedding)

            return embedding

        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts
            batch_size: Number of texts per batch
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = []
        texts_to_embed = []
        text_indices = []

        # Check cache first
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append([])
                continue

            if use_cache and self.cache_dir:
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue

            # Need to generate this embedding
            texts_to_embed.append(text)
            text_indices.append(idx)
            embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if texts_to_embed:
            self.logger.info(f"Generating {len(texts_to_embed)} new embeddings")

            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]

                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )

                    batch_embeddings = [item.embedding for item in response.data]

                    # Insert embeddings at correct indices
                    for j, embedding in enumerate(batch_embeddings):
                        original_idx = text_indices[i + j]
                        embeddings[original_idx] = embedding

                        # Cache the embedding
                        if use_cache and self.cache_dir:
                            text = texts_to_embed[i + j]
                            cache_key = self._get_cache_key(text)
                            self._save_to_cache(cache_key, embedding)

                except Exception as e:
                    self.logger.error(f"Error in batch embedding: {str(e)}")
                    # Fill with empty embeddings
                    for j in range(len(batch)):
                        original_idx = text_indices[i + j]
                        if embeddings[original_idx] is None:
                            embeddings[original_idx] = []

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key (hash of text and model)
        """
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """
        Load embedding from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached embedding or None
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading from cache: {str(e)}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """
        Save embedding to cache.

        Args:
            cache_key: Cache key
            embedding: Embedding vector
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embeddings for the current model.

        Returns:
            Embedding dimension
        """
        # Known dimensions for OpenAI models
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }

        return dimensions.get(self.model, 1536)

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# Convenience function
def create_embedding_service(
    api_key: str,
    model: str = "text-embedding-3-large",
    cache_dir: Optional[Path] = None
) -> EmbeddingService:
    """
    Create an embedding service instance.

    Args:
        api_key: OpenAI API key
        model: Embedding model name
        cache_dir: Cache directory

    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(
        api_key=api_key,
        model=model,
        cache_dir=cache_dir
    )


if __name__ == "__main__":
    # Test the embedding service
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        service = EmbeddingService(api_key=api_key)
        test_embedding = service.generate_embedding("Apple's revenue grew by 5% in 2025.")
        print(f"Generated embedding with dimension: {len(test_embedding)}")
    else:
        print("OPENAI_API_KEY not set")
