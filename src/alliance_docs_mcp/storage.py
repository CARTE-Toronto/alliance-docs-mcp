"""File storage and retrieval for mirrored documentation."""

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class DocumentationStorage:
    """Handles storage and retrieval of mirrored documentation."""
    
    def __init__(self, docs_dir: str = "./docs", compress_threshold_mb: int = 10):
        """Initialize the storage handler.
        
        Args:
            docs_dir: Base directory for storing documentation
            compress_threshold_mb: Minimum size (in MB) before markdown is gzip-compressed
        """
        self.docs_dir = Path(docs_dir)
        self.pages_dir = self.docs_dir / "pages"
        self.index_file = self.docs_dir / "index.json"
        self.compress_threshold_bytes = max(compress_threshold_mb, 0) * 1024 * 1024
        
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
        base_path = category_dir / f"{filename}.md"
        file_path = base_path
        encoded_content = markdown_content.encode("utf-8")
        should_compress = (
            self.compress_threshold_bytes > 0
            and len(encoded_content) >= self.compress_threshold_bytes
        )
        
        if should_compress:
            file_path = base_path.with_suffix(base_path.suffix + ".gz")
        
        # Remove stale variants (plain/compressed) before writing
        for stale_path in (base_path, base_path.with_suffix(base_path.suffix + ".gz")):
            if stale_path != file_path and stale_path.exists():
                try:
                    stale_path.unlink()
                except Exception as exc:
                    logger.warning(f"Unable to remove stale file {stale_path}: {exc}")
        
        # Write the markdown (possibly compressed) file
        try:
            if should_compress:
                with gzip.open(file_path, "wb") as f:
                    f.write(encoded_content)
                logger.info(
                    "Saved compressed page %s (%.2f MB → %.2f MB)",
                    file_path,
                    len(encoded_content) / (1024 * 1024),
                    file_path.stat().st_size / (1024 * 1024),
                )
            else:
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
            path = Path(file_path)
            if path.suffix == ".gz":
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(path, 'r', encoding='utf-8') as f:
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
            # Get markdown files (plain and compressed)
            markdown_files = list(self.pages_dir.rglob("*.md"))
            markdown_files.extend(self.pages_dir.rglob("*.md.gz"))
            
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
    
    def build_llms_txt(self) -> str:
        """Build llms.txt file containing a directory of all page names.
        
        Returns:
            Path to the created llms.txt file
        """
        try:
            pages = self.get_all_pages()
            llms_txt_path = self.docs_dir / "llms.txt"
            
            # Sort pages by title for consistent output
            pages_sorted = sorted(pages, key=lambda x: x.get("title", ""))
            
            # Build content - simple list of page titles with URLs
            lines = []
            lines.append("# Alliance Documentation - Page Directory")
            lines.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
            lines.append(f"# Total pages: {len(pages_sorted)}")
            lines.append("")
            
            for page in pages_sorted:
                title = page.get("title", "Unknown")
                url = page.get("url", "")
                category = page.get("category", "General")
                lines.append(f"- {title} ({category}): {url}")
            
            content = "\n".join(lines)
            
            # Write the file
            with open(llms_txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Created llms.txt with {len(pages_sorted)} pages at {llms_txt_path}")
            return str(llms_txt_path)
            
        except Exception as e:
            logger.error(f"Error building llms.txt: {e}")
            raise
    
    def build_llms_full_txt(self, compress: bool = True) -> str:
        """Build llms_full.txt containing all documentation content.
        
        Args:
            compress: Whether to compress the output with gzip (default: True)
        
        Returns:
            Path to the created llms_full.txt or llms_full.txt.gz file
        """
        try:
            pages = self.get_all_pages()
            
            # Sort pages by title for consistent output
            pages_sorted = sorted(pages, key=lambda x: x.get("title", ""))
            
            # Build content
            lines = []
            lines.append("# Alliance Documentation - Full Content")
            lines.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
            lines.append(f"# Total pages: {len(pages_sorted)}")
            lines.append("=" * 80)
            lines.append("")
            
            for page in pages_sorted:
                title = page.get("title", "Unknown")
                url = page.get("url", "")
                category = page.get("category", "General")
                file_path = page.get("file_path", "")
                
                # Add page header
                lines.append("")
                lines.append("=" * 80)
                lines.append(f"PAGE: {title}")
                lines.append(f"URL: {url}")
                lines.append(f"Category: {category}")
                lines.append("=" * 80)
                lines.append("")
                
                # Load and add page content
                if file_path:
                    try:
                        page_data = self.load_page(file_path)
                        if page_data and "content" in page_data:
                            lines.append(page_data["content"])
                        else:
                            lines.append(f"[Content not available for {title}]")
                    except Exception as e:
                        logger.warning(f"Error loading content for {title}: {e}")
                        lines.append(f"[Error loading content: {e}]")
                else:
                    lines.append(f"[No file path for {title}]")
                
                lines.append("")
            
            content = "\n".join(lines)
            encoded_content = content.encode('utf-8')
            
            # Determine if compression is needed
            base_path = self.docs_dir / "llms_full.txt"
            
            if compress:
                output_path = base_path.with_suffix(base_path.suffix + ".gz")
                with gzip.open(output_path, 'wb') as f:
                    f.write(encoded_content)
                
                original_size_mb = len(encoded_content) / (1024 * 1024)
                compressed_size_mb = output_path.stat().st_size / (1024 * 1024)
                
                logger.info(
                    f"Created compressed llms_full.txt.gz ({original_size_mb:.2f} MB → "
                    f"{compressed_size_mb:.2f} MB) at {output_path}"
                )
            else:
                output_path = base_path
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                size_mb = len(encoded_content) / (1024 * 1024)
                logger.info(f"Created llms_full.txt ({size_mb:.2f} MB) at {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error building llms_full.txt: {e}")
            raise
