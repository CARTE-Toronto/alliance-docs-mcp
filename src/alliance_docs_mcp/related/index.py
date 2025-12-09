"""Embeddings-backed related-page index."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .embedder import Embedder
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class RelatedIndexUnavailable(Exception):
    """Raised when the related-page index cannot be used."""


class RelatedIndex:
    """High-level related-page index built on embeddings and Chroma."""

    def __init__(
        self,
        index_dir: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = "chroma",
        embedder: Optional[Embedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ) -> None:
        if backend != "chroma":
            raise RelatedIndexUnavailable(f"Unsupported related index backend: {backend}")

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder(model_name=model_name)
        self.vector_store = vector_store or ChromaVectorStore(self.index_dir)

    @staticmethod
    def _content_checksum(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _text_for_embedding(title: str, content: str) -> str:
        title = title or ""
        content = content or ""
        return f"{title}\n\n{content}".strip()

    def rebuild(self) -> None:
        """Clear the collection."""
        self.vector_store.reset()

    def upsert_page(self, page: Dict, markdown_content: str) -> bool:
        """Embed and upsert a page. Returns True if written, False if skipped."""
        slug = page.get("slug")
        title = page.get("title", slug or "")
        if not slug:
            return False

        text = self._text_for_embedding(title, markdown_content or "")
        checksum = self._content_checksum(text)

        existing = self.vector_store.get_metadata(slug)
        if existing and existing.get("checksum") == checksum:
            return False  # unchanged

        embedding = self.embedder.embed([text])[0]
        metadata = {
            "slug": slug,
            "title": title,
            "url": page.get("url"),
            "category": page.get("category"),
            "last_modified": page.get("last_modified"),
            "checksum": checksum,
        }

        self.vector_store.upsert(slug=slug, embedding=embedding, metadata=metadata, document=text)
        return True

    def cleanup(self, valid_slugs: Iterable[str]) -> None:
        """Remove vectors for pages that no longer exist."""
        try:
            self.vector_store.delete_missing(valid_slugs)
        except Exception as exc:
            logger.warning("Cleanup of related index failed: %s", exc)

    def find_related(
        self, slug: str, limit: int = 5, min_score: float = 0.0
    ) -> List[Dict]:
        """Return related pages by vector similarity."""
        base_meta = self.vector_store.get_metadata(slug)
        if not base_meta:
            raise RelatedIndexUnavailable(f"Slug not found in related index: {slug}")

        query_embedding = self.vector_store.get_embedding(slug)
        try:
            if query_embedding is None or len(query_embedding) == 0:  # type: ignore[arg-type]
                raise RelatedIndexUnavailable(f"No embedding stored for slug: {slug}")
        except TypeError:
            raise RelatedIndexUnavailable(f"No embedding stored for slug: {slug}")

        if query_embedding is None:
            raise RelatedIndexUnavailable(f"No embedding stored for slug: {slug}")

        try:
            result = self.vector_store.query(embedding=query_embedding, limit=limit + 1)
        except Exception as exc:
            raise RelatedIndexUnavailable(str(exc)) from exc

        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        distances = result.get("distances") or []

        if not ids or not metadatas or not distances:
            return []

        candidate_ids = ids[0] if isinstance(ids[0], list) else ids
        candidate_metas = metadatas[0] if isinstance(metadatas[0], list) else metadatas
        candidate_dists = distances[0] if isinstance(distances[0], list) else distances

        related = []
        for candidate_id, meta, dist in zip(candidate_ids, candidate_metas, candidate_dists):
            if candidate_id == slug:
                continue
            meta = meta or {}
            score = 1 - dist if dist is not None else None
            if score is None or score < min_score:
                continue
            related.append(
                {
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "category": meta.get("category"),
                    "slug": meta.get("slug") or candidate_id,
                    "score": score,
                    "last_modified": meta.get("last_modified"),
                }
            )

        # Keep deterministic ordering by score
        related.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
        return related[:limit]

