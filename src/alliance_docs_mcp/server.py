"""FastMCP server for Alliance documentation."""

import logging
import os
from typing import List, Optional

from fastmcp import FastMCP

from .storage import DocumentationStorage

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Alliance Docs")

# Initialize storage
docs_dir = os.getenv("DOCS_DIR", "./docs")
storage = DocumentationStorage(docs_dir)


@mcp.resource("alliance-docs://page/{slug}")
async def get_page(slug: str) -> str:
    """Get a documentation page by slug.
    
    Args:
        slug: Page slug (filename without extension)
        
    Returns:
        Markdown content of the page
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
        logger.error(f"Error getting page {slug}: {e}")
        return f"Error loading page: {e}"


@mcp.tool()
async def search_docs(query: str, category: Optional[str] = None) -> List[dict]:
    """Search documentation pages.
    
    Args:
        query: Search query
        category: Optional category filter
        
    Returns:
        List of matching pages with metadata
    """
    try:
        results = storage.search_pages(query, category)
        
        # Return simplified results for MCP
        return [
            {
                "title": page["title"],
                "url": page["url"],
                "category": page["category"],
                "slug": page["slug"],
                "last_modified": page["last_modified"]
            }
            for page in results
        ]
        
    except Exception as e:
        logger.error(f"Error searching docs: {e}")
        return []


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


def main():
    """Run the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alliance Docs MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--docs-dir", default="./docs", help="Documentation directory")
    
    args = parser.parse_args()
    
    # Update storage path if provided
    global storage
    storage = DocumentationStorage(args.docs_dir)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting Alliance Docs MCP Server on {args.host}:{args.port}")
    logger.info(f"Documentation directory: {args.docs_dir}")
    
    # Run the server
    mcp.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
