"""Full-text search index using Whoosh."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import DATETIME, ID, KEYWORD, TEXT, Schema
from whoosh.qparser import FuzzyTermPlugin, MultifieldParser, OrGroup
from whoosh.query import Term

logger = logging.getLogger(__name__)


class SearchIndexUnavailable(Exception):
    """Raised when the search index cannot be used."""


class SearchIndex:
    """Lightweight Whoosh-based full-text search."""

    def __init__(self, index_dir: Path, enabled: bool = True) -> None:
        """
        Args:
            index_dir: Directory where the Whoosh index lives.
            enabled: If False, the index is not created or used.
        """
        self.enabled = enabled
        self.index_dir = Path(index_dir)
        self.schema = self._build_schema()
        self._index = self._create_or_open_index() if enabled else None

    def _build_schema(self) -> Schema:
        return Schema(
            slug=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            category=KEYWORD(stored=True, lowercase=True, commas=True, scorable=True),
            url=ID(stored=True),
            last_modified=DATETIME(stored=True),
        )

    def _create_or_open_index(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            if index.exists_in(self.index_dir):
                return index.open_dir(self.index_dir)
            return index.create_in(self.index_dir, self.schema)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to initialize search index: %s", exc)
            raise SearchIndexUnavailable(str(exc))

    def _normalize_datetime(self, value) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def index_page(
        self,
        slug: str,
        title: str,
        content: str,
        category: str,
        url: str,
        last_modified,
    ) -> None:
        """Add or update a page in the index."""
        if not self.enabled or not self._index:
            return

        normalized_dt = self._normalize_datetime(last_modified)

        try:
            writer = self._index.writer()
            writer.update_document(
                slug=slug,
                title=title or slug,
                content=content or "",
                category=category or "General",
                url=url or "",
                last_modified=normalized_dt,
            )
            writer.commit()
        except Exception as exc:
            logger.warning("Failed to index page %s: %s", slug, exc)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
        fuzzy: bool = False,
    ) -> List[Dict]:
        """Search the index with relevance ranking and highlights."""
        if not self.enabled or not self._index:
            raise SearchIndexUnavailable("Search index disabled or unavailable")

        try:
            with self._index.searcher() as searcher:
                parser = MultifieldParser(
                    ["title", "content"],
                    schema=self._index.schema,
                    group=OrGroup.factory(0.9),
                )
                if fuzzy:
                    parser.add_plugin(FuzzyTermPlugin())

                parsed_query = parser.parse(query)
                category_filter = Term("category", category.lower()) if category else None

                results = searcher.search(parsed_query, limit=limit, filter=category_filter)
                return [
                    {
                        "title": hit.get("title"),
                        "slug": hit.get("slug"),
                        "url": hit.get("url"),
                        "category": hit.get("category"),
                        "score": hit.score,
                        "highlights": hit.highlights("content", top=3),
                        "last_modified": hit.get("last_modified"),
                    }
                    for hit in results
                ]
        except SearchIndexUnavailable:
            raise
        except Exception as exc:
            logger.warning("Search failed: %s", exc)
            raise SearchIndexUnavailable(str(exc))

    def optimize(self) -> None:
        """Optimize the index storage."""
        if not self.enabled or not self._index:
            return
        try:
            self._index.optimize()
        except Exception as exc:
            logger.debug("Index optimize failed: %s", exc)


__all__ = ["SearchIndex", "SearchIndexUnavailable"]




