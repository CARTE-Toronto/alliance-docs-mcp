"""Chroma-backed vector store wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import chromadb

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Lightweight wrapper around a persistent Chroma collection."""

    def __init__(self, index_dir: Path, collection_name: str = "related_pages") -> None:
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # pragma: no cover - collection may not exist
            logger.debug("Collection %s did not exist when resetting", self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        slug: str,
        embedding: Sequence[float],
        metadata: Dict,
        document: Optional[str] = None,
    ) -> None:
        """Insert or update a single embedding."""
        self._collection.upsert(
            ids=[slug],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document] if document is not None else None,
        )

    def get_metadata(self, slug: str) -> Optional[Dict]:
        """Return stored metadata for an id, if present."""
        result = self._collection.get(ids=[slug], include=["metadatas"])
        if result and result.get("ids"):
            metas = result.get("metadatas") or []
            if metas:
                return metas[0]
        return None

    def get_embedding(self, slug: str) -> Optional[List[float]]:
        """Return a stored embedding for an id, if present."""
        result = self._collection.get(ids=[slug], include=["embeddings"])
        if not result or not result.get("ids"):
            return None

        embeddings = result.get("embeddings")
        if embeddings is None:
            return None

        try:
            return embeddings[0]
        except Exception:
            return None

    def get_all_ids(self, batch_size: int = 500) -> List[str]:
        """Return all ids in the collection."""
        ids: List[str] = []
        offset = 0
        while True:
            try:
                result = self._collection.get(
                    include=[],
                    limit=batch_size,
                    offset=offset,
                )
            except TypeError:
                # Older Chroma versions may not support offset
                result = self._collection.get(include=[], limit=batch_size)

            batch_ids = result.get("ids") or []
            ids.extend(batch_ids)
            if len(batch_ids) < batch_size:
                break
            offset += batch_size
        return ids

    def delete_missing(self, valid_ids: Iterable[str]) -> None:
        """Remove any records not present in valid_ids."""
        valid_set = set(valid_ids)
        existing_ids = self.get_all_ids()
        to_delete = [id_ for id_ in existing_ids if id_ not in valid_set]
        if to_delete:
            self._collection.delete(ids=to_delete)

    def query(
        self,
        embedding: Sequence[float],
        limit: int = 5,
    ) -> Dict:
        """Run a similarity query."""
        return self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["metadatas", "distances"],
        )

