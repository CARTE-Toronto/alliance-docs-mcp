"""Sentence-transformer embedder with lazy model loading."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer model with lazy loading."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Avoid fork-safety warnings from huggingface/tokenizers
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-transformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch of texts into embeddings."""
        if not isinstance(texts, Iterable):
            raise TypeError("texts must be iterable")

        model = self._ensure_model()
        # Normalized embeddings make cosine distance comparable
        vectors = model.encode(
            list(texts),
            convert_to_numpy=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in vectors]

