import tempfile
from pathlib import Path

import pytest

from alliance_docs_mcp.search_index import SearchIndex, SearchIndexUnavailable


def test_search_index_creates_and_searches():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_dir = Path(tmp_dir) / "search_index"
        search_index = SearchIndex(index_dir)

        search_index.index_page(
            slug="gpu_usage",
            title="GPU Usage Guide",
            content="Request GPUs with --gpus-per-node flags.",
            category="Technical",
            url="https://example.com/gpu",
            last_modified="2025-01-01T00:00:00Z",
        )
        search_index.index_page(
            slug="cpu_usage",
            title="CPU Usage Guide",
            content="CPU guide",
            category="Technical",
            url="https://example.com/cpu",
            last_modified="2025-01-02T00:00:00Z",
        )

        results = search_index.search("GPU", limit=5)
        assert results
        assert results[0]["slug"] == "gpu_usage"
        assert "gpus" in results[0]["highlights"].lower()
        assert results[0]["score"] >= 0


def test_search_index_disabled_returns_unavailable():
    with tempfile.TemporaryDirectory() as tmp_dir:
        search_index = SearchIndex(Path(tmp_dir), enabled=False)
        with pytest.raises(SearchIndexUnavailable):
            search_index.search("anything")




