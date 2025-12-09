"""FastMCP server for Alliance documentation."""

import gzip
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP
from fastmcp.resources import TextResource
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .related import RelatedIndex, RelatedIndexUnavailable
from .search_index import SearchIndex, SearchIndexUnavailable
from .storage import DocumentationStorage

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Alliance Docs")

def _discover_docs_directory() -> Path:
    """Determine the directory that contains mirrored documentation files."""
    configured_docs_dir = os.getenv("DOCS_DIR")
    candidates = []

    if configured_docs_dir:
        configured_path = Path(configured_docs_dir)
        candidates.append(configured_path)

    module_path = Path(__file__).resolve()
    candidates.extend(
        [
            Path.cwd() / "docs",
            module_path.parent / "docs",
            module_path.parents[1] / "docs",
            module_path.parents[2] / "docs",
        ]
    )

    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Documentation directory not found. Set DOCS_DIR environment variable to the docs path."
    )


docs_path = _discover_docs_directory()
storage = DocumentationStorage(str(docs_path))
search_index = None

try:
    search_index_disabled = os.getenv("DISABLE_SEARCH_INDEX", "").lower() in ("1", "true", "yes")
    search_index_dir = Path(os.getenv("SEARCH_INDEX_DIR", docs_path / "search_index"))
    if not search_index_disabled:
        search_index = SearchIndex(search_index_dir)
    else:
        logger.info("Search index disabled via DISABLE_SEARCH_INDEX")
except Exception as exc:  # pragma: no cover - defensive initialization
    logger.warning("Search index unavailable, falling back to title search: %s", exc)
    search_index = None

related_index = None

try:
    related_index_disabled = os.getenv("DISABLE_RELATED_INDEX", "").lower() in ("1", "true", "yes")
    related_index_dir = Path(os.getenv("RELATED_INDEX_DIR", docs_path / "related_index"))
    related_model_name = os.getenv("RELATED_MODEL_NAME", "all-MiniLM-L6-v2")
    related_backend = os.getenv("RELATED_BACKEND", "chroma")

    if not related_index_disabled:
        related_index = RelatedIndex(
            related_index_dir,
            model_name=related_model_name,
            backend=related_backend,
        )
    else:
        logger.info("Related index disabled via DISABLE_RELATED_INDEX")
except Exception as exc:  # pragma: no cover - defensive initialization
    logger.warning("Related index unavailable, using heuristic fallback: %s", exc)
    related_index = None


def _resolve_page_path(file_path: str) -> Path:
    """Resolve a page path to an absolute location on disk."""
    path_obj = Path(file_path)

    if path_obj.is_absolute():
        return path_obj

    # Allow paths that already include the docs/ prefix
    if path_obj.parts and path_obj.parts[0] == "docs":
        path_obj = Path(*path_obj.parts[1:])

    candidate = docs_path / path_obj
    return candidate.resolve()


def _register_document_resources() -> None:
    """Register each mirrored document as an MCP resource."""
    pages = storage.get_all_pages()
    total = 0

    for page in pages:
        slug = page.get("slug")
        file_path = page.get("file_path")

        if not slug or not file_path:
            continue

        uri = f"alliance-docs://page/{slug}"
        absolute_path = _resolve_page_path(file_path)

        if not absolute_path.exists():
            logger.warning("Resource file missing on disk: %s", absolute_path)
            continue

        try:
            if absolute_path.suffix == ".gz":
                with gzip.open(absolute_path, "rt", encoding="utf-8") as handle:
                    page_text = handle.read()
            else:
                page_text = absolute_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.error("Error loading resource %s: %s", absolute_path, exc)
            continue

        metadata = {
            "title": page.get("title"),
            "url": page.get("url"),
            "category": page.get("category"),
            "last_modified": page.get("last_modified"),
            "page_id": page.get("page_id"),
        }

        resource = TextResource(
            uri=uri,
            name=page.get("title") or slug,
            description=page.get("url"),
            text=page_text,
            mime_type="text/markdown",
            tags={page.get("category", "General")},
            meta=metadata,
        )

        mcp.add_resource(resource)
        total += 1

    logger.info("Registered %s documentation resources", total)


_register_document_resources()


async def _search_docs_impl(
    query: str,
    category: Optional[str] = None,
    limit: int = 20,
    search_content: bool = True,
    fuzzy: bool = False,
) -> List[dict]:
    """Core search implementation used by the MCP tool and tests."""
    try:
        if search_content and search_index:
            try:
                results = search_index.search(query, category=category, limit=limit, fuzzy=fuzzy)
                return [
                    {
                        "title": hit.get("title"),
                        "url": hit.get("url"),
                        "category": hit.get("category"),
                        "slug": hit.get("slug"),
                        "last_modified": hit.get("last_modified"),
                        "score": hit.get("score"),
                        "snippet": hit.get("highlights"),
                        "highlights": hit.get("highlights"),
                    }
                    for hit in results
                ]
            except SearchIndexUnavailable:
                logger.warning("Search index unavailable, falling back to title search")
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Full-text search failed, falling back to title search: %s", exc)

        results = storage.search_pages(query, category)
        return [
            {
                "title": page["title"],
                "url": page["url"],
                "category": page["category"],
                "slug": page["slug"],
                "last_modified": page["last_modified"],
            }
            for page in results[:limit]
        ]

    except Exception as e:
        logger.error(f"Error searching docs: {e}")
        return []


@mcp.tool()
async def search_docs(
    query: str,
    category: Optional[str] = None,
    limit: int = 20,
    search_content: bool = True,
    fuzzy: bool = False,
) -> List[dict]:
    """Search documentation with optional full-text index and relevance ranking."""
    return await _search_docs_impl(query, category, limit, search_content, fuzzy)


@mcp.tool()
async def list_categories() -> List[str]:
    """List all available documentation categories.
    
    Returns:
        List of category names
    """
    try:
        return storage.get_categories()
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        return []


def _heuristic_related(page: dict, limit: int) -> List[dict]:
    """Lightweight heuristic fallback for related pages."""
    base_tokens = set((page.get("title", "") or "").lower().split())
    candidates = []

    for candidate in storage.get_all_pages():
        if candidate.get("slug") == page.get("slug"):
            continue

        score = 0
        if candidate.get("category") == page.get("category"):
            score += 2

        candidate_tokens = set((candidate.get("title", "") or "").lower().split())
        score += len(base_tokens.intersection(candidate_tokens))

        if score > 0:
            candidates.append((score, candidate))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "title": candidate["title"],
            "url": candidate["url"],
            "category": candidate["category"],
            "slug": candidate["slug"],
            "score": score,
        }
        for score, candidate in candidates[:limit]
    ]


@mcp.tool()
async def get_page_by_title(title: str) -> Optional[dict]:
    """Find a specific page by title.
    
    Args:
        title: Page title to search for
        
    Returns:
        Page metadata or None if not found
    """
    try:
        # Search for exact title match
        results = storage.search_pages(title)
        
        for page in results:
            if page["title"].lower() == title.lower():
                return {
                    "title": page["title"],
                    "url": page["url"],
                    "category": page["category"],
                    "slug": page["slug"],
                    "last_modified": page["last_modified"]
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding page by title {title}: {e}")
        return None


@mcp.tool()
async def list_recent_updates(limit: int = 10) -> List[dict]:
    """List recently updated pages.
    
    Args:
        limit: Maximum number of pages to return
        
    Returns:
        List of recent pages with metadata
    """
    try:
        recent_pages = storage.get_recent_pages(limit)
        
        return [
            {
                "title": page["title"],
                "url": page["url"],
                "category": page["category"],
                "slug": page["slug"],
                "last_modified": page["last_modified"]
            }
            for page in recent_pages
        ]
        
    except Exception as e:
        logger.error(f"Error getting recent updates: {e}")
        return []


@mcp.tool()
async def find_related_pages(slug: str, limit: int = 5, min_score: float = 0.0) -> List[dict]:
    """Find related pages using embeddings when available, with heuristic fallback."""
    return await _find_related_pages_impl(slug, limit, min_score)


async def _find_related_pages_impl(slug: str, limit: int = 5, min_score: float = 0.0) -> List[dict]:
    """Core related-pages implementation for tool and tests."""
    try:
        page = storage.get_page_by_slug(slug)
        if not page:
            return []

        if related_index:
            try:
                results = related_index.find_related(slug, limit=limit, min_score=min_score)
                if results:
                    return results
            except RelatedIndexUnavailable as exc:
                logger.warning("Related index unavailable for %s: %s", slug, exc)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Related index error for %s: %s", slug, exc)

        return _heuristic_related(page, limit)

    except Exception as exc:
        logger.error("Error finding related pages for %s: %s", slug, exc)
        return []


@mcp.tool()
async def get_page_info(slug: str) -> Optional[dict]:
    """Get detailed information about a page.
    
    Args:
        slug: Page slug
        
    Returns:
        Detailed page information or None if not found
    """
    try:
        page_data = storage.get_page_by_slug(slug)
        if not page_data:
            return None
        
        # Load full content to get metadata
        page_content = storage.load_page(page_data["file_path"])
        if not page_content:
            return None
        
        return {
            "title": page_data["title"],
            "url": page_data["url"],
            "category": page_data["category"],
            "slug": page_data["slug"],
            "last_modified": page_data["last_modified"],
            "page_id": page_data["page_id"],
            "metadata": page_content["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error getting page info for {slug}: {e}")
        return None


@mcp.tool()
async def list_all_pages() -> List[dict]:
    """List all available documentation pages.
    
    Returns:
        List of all pages with basic metadata
    """
    try:
        all_pages = storage.get_all_pages()
        
        return [
            {
                "title": page["title"],
                "url": page["url"],
                "category": page["category"],
                "slug": page["slug"],
                "last_modified": page["last_modified"]
            }
            for page in all_pages
        ]
        
    except Exception as e:
        logger.error(f"Error listing all pages: {e}")
        return []


@mcp.tool()
async def get_page_content(slug: str) -> str:
    """Get the full content of a documentation page.
    
    Args:
        slug: Page slug (filename without extension)
        
    Returns:
        Full markdown content of the page
    """
    try:
        page_data = storage.get_page_by_slug(slug)
        if not page_data:
            return f"Page not found: {slug}"
        
        # Load the actual content
        page_content = storage.load_page(page_data["file_path"])
        if not page_content:
            return f"Error loading page content: {slug}"
        
        return page_content["content"]
        
    except Exception as e:
        logger.error(f"Error getting page content for {slug}: {e}")
        return f"Error loading page: {e}"


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Simple health endpoint for platform probes."""
    return PlainTextResponse("ok")


@mcp.custom_route("/", methods=["GET"])
async def root(request: Request) -> PlainTextResponse:
    """Return a basic status message for root requests."""
    return PlainTextResponse("Alliance Docs MCP server is running. Try /health for probe status.")


def main():
    """Run the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alliance Docs MCP Server")
    parser.add_argument("--docs-dir", default="./docs", help="Documentation directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Update storage path if provided
    global storage
    storage = DocumentationStorage(args.docs_dir)
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Alliance Docs MCP Server")
    logger.info(f"Documentation directory: {args.docs_dir}")
    
    # Run the server as stdio (for MCP protocol)
    mcp.run()


if __name__ == "__main__":
    main()
