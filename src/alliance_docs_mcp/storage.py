"""File storage and retrieval for mirrored documentation."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class DocumentationStorage:
    """Handles storage and retrieval of mirrored documentation."""
    
    def __init__(self, docs_dir: str = "./docs"):
        """Initialize the storage handler.
        
        Args:
            docs_dir: Base directory for storing documentation
        """
        self.docs_dir = Path(docs_dir)
        self.pages_dir = self.docs_dir / "pages"
        self.index_file = self.docs_dir / "index.json"
        
        # Ensure directories exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
    
    def save_page(self, page_data: Dict, markdown_content: str) -> str:
        """Save a page as a markdown file.
        
        Args:
            page_data: Page metadata
            markdown_content: Markdown content with frontmatter
            
        Returns:
            Path to the saved file
        """
        # Create category directory
        category = self._extract_category(page_data.get("title", ""))
        category_dir = self.pages_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = self._title_to_filename(page_data.get("title", ""))
        file_path = category_dir / f"{filename}.md"
        
        # Write the markdown file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Saved page: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving page {page_data.get('title', '')}: {e}")
            raise
    
    def load_page(self, file_path: str) -> Optional[Dict]:
        """Load a page from a markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with content and metadata, or None if not found
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse frontmatter
            frontmatter, markdown_content = self._parse_frontmatter(content)
            
            return {
                "content": markdown_content,
                "metadata": frontmatter,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error loading page {file_path}: {e}")
            return None
    
    def update_index(self, pages: List[Dict]) -> None:
        """Update the index file with page metadata.
        
        Args:
            pages: List of page metadata dictionaries
        """
        index_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_pages": len(pages),
            "pages": pages
        }
        
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated index with {len(pages)} pages")
            
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            raise
    
    def load_index(self) -> Dict:
        """Load the index file.
        
        Returns:
            Index data dictionary
        """
        if not self.index_file.exists():
            return {
                "last_updated": None,
                "total_pages": 0,
                "pages": []
            }
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {
                "last_updated": None,
                "total_pages": 0,
                "pages": []
            }
    
    def get_all_pages(self) -> List[Dict]:
        """Get all pages from the index.
        
        Returns:
            List of page metadata
        """
        index = self.load_index()
        return index.get("pages", [])
    
    def search_pages(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """Search pages by title and content.
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            List of matching pages
        """
        pages = self.get_all_pages()
        results = []
        
        query_lower = query.lower()
        
        for page in pages:
            # Filter by category if specified
            if category and page.get("category", "").lower() != category.lower():
                continue
            
            # Search in title
            title_match = query_lower in page.get("title", "").lower()
            
            # Search in content (if available)
            content_match = False
            if "content" in page:
                content_match = query_lower in page["content"].lower()
            
            if title_match or content_match:
                results.append(page)
        
        return results
    
    def get_categories(self) -> List[str]:
        """Get all available categories.
        
        Returns:
            List of category names
        """
        pages = self.get_all_pages()
        categories = set()
        
        for page in pages:
            category = page.get("category", "General")
            categories.add(category)
        
        return sorted(list(categories))
    
    def get_recent_pages(self, limit: int = 10) -> List[Dict]:
        """Get recently updated pages.
        
        Args:
            limit: Maximum number of pages to return
            
        Returns:
            List of recent pages
        """
        pages = self.get_all_pages()
        
        # Sort by last_modified
        pages.sort(
            key=lambda x: x.get("last_modified", ""),
            reverse=True
        )
        
        return pages[:limit]
    
    def get_page_by_slug(self, slug: str) -> Optional[Dict]:
        """Get a page by its slug.
        
        Args:
            slug: Page slug (filename without extension)
            
        Returns:
            Page data or None if not found
        """
        pages = self.get_all_pages()
        
        for page in pages:
            if page.get("slug") == slug:
                return page
        
        return None
    
    def _parse_frontmatter(self, content: str) -> tuple:
        """Parse YAML frontmatter from markdown content.
        
        Args:
            content: Markdown content with frontmatter
            
        Returns:
            Tuple of (frontmatter dict, markdown content)
        """
        if not content.startswith("---\n"):
            return {}, content
        
        # Find the end of frontmatter
        lines = content.split('\n')
        if len(lines) < 2:
            return {}, content
        
        end_marker = None
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end_marker = i
                break
        
        if end_marker is None:
            return {}, content
        
        # Extract frontmatter
        frontmatter_lines = lines[1:end_marker]
        frontmatter_content = '\n'.join(frontmatter_lines)
        
        try:
            frontmatter = yaml.safe_load(frontmatter_content) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing frontmatter: {e}")
            frontmatter = {}
        
        # Extract markdown content
        markdown_content = '\n'.join(lines[end_marker + 1:])
        
        return frontmatter, markdown_content
    
    def _extract_category(self, title: str) -> str:
        """Extract category from page title.
        
        Args:
            title: Page title
            
        Returns:
            Category name
        """
        # Simple category extraction based on common patterns
        if ":" in title:
            return title.split(":")[0]
        
        # Default categories based on keywords
        title_lower = title.lower()
        if any(word in title_lower for word in ["getting", "started", "setup", "install"]):
            return "Getting Started"
        elif any(word in title_lower for word in ["user", "guide", "tutorial"]):
            return "User Guide"
        elif any(word in title_lower for word in ["api", "reference", "technical"]):
            return "Technical Reference"
        elif any(word in title_lower for word in ["troubleshoot", "problem", "issue"]):
            return "Troubleshooting"
        else:
            return "General"
    
    def _title_to_filename(self, title: str) -> str:
        """Convert page title to filename.
        
        Args:
            title: Page title
            
        Returns:
            Safe filename
        """
        # Remove or replace invalid characters
        filename = title.replace(" ", "_")
        filename = filename.replace("/", "_")
        filename = filename.replace("\\", "_")
        filename = filename.replace(":", "_")
        filename = filename.replace("*", "_")
        filename = filename.replace("?", "_")
        filename = filename.replace('"', "_")
        filename = filename.replace("<", "_")
        filename = filename.replace(">", "_")
        filename = filename.replace("|", "_")
        
        # Remove multiple underscores
        filename = "_".join(filter(None, filename.split("_")))
        
        return filename
    
    def cleanup_old_files(self, keep_recent: int = 100) -> None:
        """Clean up old files that are no longer in the index.
        
        Args:
            keep_recent: Number of recent files to keep
        """
        try:
            # Get all markdown files
            markdown_files = list(self.pages_dir.rglob("*.md"))
            
            # Get current pages from index
            current_pages = self.get_all_pages()
            current_files = {page.get("file_path") for page in current_pages if page.get("file_path")}
            
            # Find files to remove
            files_to_remove = []
            for file_path in markdown_files:
                if str(file_path) not in current_files:
                    files_to_remove.append(file_path)
            
            # Remove old files (keep most recent)
            files_to_remove.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            files_to_remove = files_to_remove[keep_recent:]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error removing file {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
