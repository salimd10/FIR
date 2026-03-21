"""
BM25 keyword search implementation for precise term matching.
Essential for financial terminology and specific phrases.
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from loguru import logger
import re
from pathlib import Path
import pickle


class BM25KeywordSearch:
    """
    BM25-based keyword search for financial documents.
    Complements vector search by capturing exact term matches.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize BM25 search.

        Args:
            cache_dir: Directory for caching BM25 index
        """
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        self.cache_dir = cache_dir
        self.logger = logger.bind(module="bm25_search")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Create BM25 index from document chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.get("content", ""))
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.logger.info(f"Indexed {len(chunks)} chunks for BM25 search")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Preserves financial terms and numbers.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Preserve financial terms and numbers
        # Keep numbers with $ and commas
        text = re.sub(r'\$[\d,]+\.?\d*[kmb]?', lambda m: m.group().replace(',', ''), text)

        # Tokenize while preserving meaningful punctuation
        tokens = re.findall(r'\b\w+\b|\$[\d.]+[kmb]?', text)

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching chunks with BM25 scores
        """
        if not self.bm25:
            self.logger.warning("BM25 index not initialized")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return chunks with positive scores
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                chunk["rank"] = len(results) + 1
                results.append(chunk)

        return results

    def save_index(self, filepath: Path):
        """
        Save BM25 index to disk.

        Args:
            filepath: Path to save index
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'chunks': self.chunks,
                    'tokenized_corpus': self.tokenized_corpus
                }, f)
            self.logger.info(f"BM25 index saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving BM25 index: {str(e)}")

    def load_index(self, filepath: Path):
        """
        Load BM25 index from disk.

        Args:
            filepath: Path to load index from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
                self.tokenized_corpus = data['tokenized_corpus']
            self.logger.info(f"BM25 index loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading BM25 index: {str(e)}")


if __name__ == "__main__":
    # Test BM25 search
    sample_chunks = [
        {
            "chunk_id": "1",
            "content": "Apple's Research and Development expenses increased to $31.4 billion in 2025.",
            "page_number": 39
        },
        {
            "chunk_id": "2",
            "content": "Net sales in the Americas segment reached $160 billion.",
            "page_number": 45
        }
    ]

    bm25 = BM25KeywordSearch()
    bm25.index_chunks(sample_chunks)

    results = bm25.search("R&D expenses", top_k=5)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"Score: {r['score']:.2f} - {r['content'][:100]}")
