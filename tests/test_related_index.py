from pathlib import Path

import pytest

from alliance_docs_mcp.related.index import RelatedIndex, RelatedIndexUnavailable


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        # Deterministic, non-zero vector per text length
        return [[float(len(text))] * 3 for text in texts]


def _page(slug: str, title: str) -> dict:
    return {
        "slug": slug,
        "title": title,
        "url": f"https://example.com/{slug}",
        "category": "General",
        "last_modified": "2025-01-01T00:00:00Z",
    }


def test_related_index_upsert_and_query(tmp_path: Path):
    embedder = FakeEmbedder()
    related_index = RelatedIndex(tmp_path / "related_index", embedder=embedder)

    page_a = _page("page_a", "Alpha Page")
    page_b = _page("page_b", "Beta Page")

    wrote_a = related_index.upsert_page(page_a, "alpha content")
    wrote_b = related_index.upsert_page(page_b, "beta content")

    assert wrote_a is True
    assert wrote_b is True
    assert embedder.calls == 2

    results = related_index.find_related("page_a", limit=2)
    assert results
    assert results[0]["slug"] == "page_b"
    assert results[0]["score"] is not None


def test_related_index_skips_unchanged(tmp_path: Path):
    embedder = FakeEmbedder()
    related_index = RelatedIndex(tmp_path / "related_index", embedder=embedder)

    page = _page("page_a", "Alpha Page")
    wrote_first = related_index.upsert_page(page, "alpha content")
    wrote_second = related_index.upsert_page(page, "alpha content")

    assert wrote_first is True
    assert wrote_second is False
    assert embedder.calls == 1


def test_related_index_missing_slug_raises(tmp_path: Path):
    related_index = RelatedIndex(tmp_path / "related_index", embedder=FakeEmbedder())
    with pytest.raises(RelatedIndexUnavailable):
        related_index.find_related("missing-slug")


def test_related_index_cleanup(tmp_path: Path):
    related_index = RelatedIndex(tmp_path / "related_index", embedder=FakeEmbedder())
    page_a = _page("page_a", "Alpha Page")
    page_b = _page("page_b", "Beta Page")
    related_index.upsert_page(page_a, "alpha content")
    related_index.upsert_page(page_b, "beta content")

    related_index.cleanup({"page_a"})
    # Only self remains, so no related results
    results = related_index.find_related("page_a", limit=2)
    assert results == []

