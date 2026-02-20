"""Embedding Engine — generates dense vector representations using sentence-transformers.

Supports batched encoding, caching, and GPU acceleration.
"""

from typing import List, Union, Optional
import numpy as np
from loguru import logger


class EmbeddingEngine:
    """Generate embeddings using sentence-transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformer model.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        self._cache = {}  # Simple in-memory embedding cache

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading embedding model '{self.model_name}' on {self._device}")
            self._model = SentenceTransformer(self.model_name, device=self._device)

        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into dense vector embeddings.

        Args:
            texts: Single string or list of strings to encode.
            batch_size: Batch size for encoding.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text, normalize)
            if cache_key in self._cache:
                cached_embeddings[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Populate cache
            for idx, text_idx in enumerate(uncached_indices):
                cache_key = self._cache_key(texts[text_idx], normalize)
                self._cache[cache_key] = new_embeddings[idx]
                cached_embeddings[text_idx] = new_embeddings[idx]

        # Assemble final result in order
        all_embeddings = np.stack([cached_embeddings[i] for i in range(len(texts))])
        return all_embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text and return a 1-D vector."""
        return self.encode([text], normalize=normalize)[0]

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def _cache_key(self, text: str, normalize: bool) -> str:
        """Generate a cache key from text and settings."""
        return f"{hash(text)}_{normalize}"
