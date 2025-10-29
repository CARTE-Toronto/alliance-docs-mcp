"""Tests for documentation storage functionality."""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from alliance_docs_mcp.storage import DocumentationStorage


class TestDocumentationStorage:
    """Test cases for DocumentationStorage."""
    
    def test_init(self):
        """Test storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            assert storage.docs_dir == Path(temp_dir)
            assert storage.pages_dir == Path(temp_dir) / "pages"
            assert storage.index_file == Path(temp_dir) / "index.json"
    
    def test_save_and_load_page(self):
        """Test saving and loading a page."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Test data
            page_data = {
                "title": "Test Page",
                "url": "https://example.com/Test_Page",
                "pageid": 123,
                "lastmodified": "2025-01-01T00:00:00Z"
            }
            
            markdown_content = """---
title: "Test Page"
url: "https://example.com/Test_Page"
category: "General"
last_modified: "2025-01-01T00:00:00Z"
---

# Test Page

This is test content.
"""
            
            # Save page
            file_path = storage.save_page(page_data, markdown_content)
            assert Path(file_path).exists()
            
            # Load page
            loaded_page = storage.load_page(file_path)
            assert loaded_page is not None
            assert "Test Page" in loaded_page["content"]
            assert loaded_page["metadata"]["title"] == "Test Page"
    
    def test_update_and_load_index(self):
        """Test updating and loading the index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Test pages
            pages = [
                {
                    "page_id": 1,
                    "title": "Page 1",
                    "url": "https://example.com/Page_1",
                    "category": "General",
                    "last_modified": "2025-01-01T00:00:00Z",
                    "file_path": "/path/to/page1.md",
                    "slug": "page_1"
                },
                {
                    "page_id": 2,
                    "title": "Page 2",
                    "url": "https://example.com/Page_2",
                    "category": "Technical",
                    "last_modified": "2025-01-02T00:00:00Z",
                    "file_path": "/path/to/page2.md",
                    "slug": "page_2"
                }
            ]
            
            # Update index
            storage.update_index(pages)
            
            # Load index
            index = storage.load_index()
            assert index["total_pages"] == 2
            assert len(index["pages"]) == 2
            assert index["pages"][0]["title"] == "Page 1"
    
    def test_search_pages(self):
        """Test searching pages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Create test index
            pages = [
                {
                    "page_id": 1,
                    "title": "Getting Started Guide",
                    "url": "https://example.com/Getting_Started",
                    "category": "Getting Started",
                    "last_modified": "2025-01-01T00:00:00Z",
                    "file_path": "/path/to/getting_started.md",
                    "slug": "getting_started"
                },
                {
                    "page_id": 2,
                    "title": "Technical Reference",
                    "url": "https://example.com/Technical_Reference",
                    "category": "Technical",
                    "last_modified": "2025-01-02T00:00:00Z",
                    "file_path": "/path/to/technical.md",
                    "slug": "technical_reference"
                }
            ]
            
            storage.update_index(pages)
            
            # Test search
            results = storage.search_pages("getting")
            assert len(results) == 1
            assert results[0]["title"] == "Getting Started Guide"
            
            # Test category filter
            results = storage.search_pages("reference", category="Technical")
            assert len(results) == 1
            assert results[0]["title"] == "Technical Reference"
    
    def test_get_categories(self):
        """Test getting categories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Create test index with different categories
            pages = [
                {"title": "Page 1", "category": "General"},
                {"title": "Page 2", "category": "Technical"},
                {"title": "Page 3", "category": "General"},
                {"title": "Page 4", "category": "User Guide"}
            ]
            
            storage.update_index(pages)
            categories = storage.get_categories()
            
            assert "General" in categories
            assert "Technical" in categories
            assert "User Guide" in categories
            assert len(categories) == 3  # No duplicates
    
    def test_title_to_filename(self):
        """Test title to filename conversion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Test various title formats
            assert storage._title_to_filename("Simple Title") == "Simple_Title"
            assert storage._title_to_filename("Title/With/Slashes") == "Title_With_Slashes"
            assert storage._title_to_filename("Title:With:Colons") == "Title_With_Colons"
            assert storage._title_to_filename("Title*With*Stars") == "Title_With_Stars"
            assert storage._title_to_filename("Title With Spaces") == "Title_With_Spaces"
    
    def test_build_llms_txt(self):
        """Test building llms.txt directory file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Create test index with sample pages
            pages = [
                {
                    "page_id": 1,
                    "title": "Getting Started",
                    "url": "https://example.com/Getting_Started",
                    "category": "General",
                    "last_modified": "2025-01-01T00:00:00Z",
                    "file_path": f"{temp_dir}/pages/General/getting_started.md",
                    "slug": "getting_started"
                },
                {
                    "page_id": 2,
                    "title": "Advanced Topics",
                    "url": "https://example.com/Advanced_Topics",
                    "category": "Technical",
                    "last_modified": "2025-01-02T00:00:00Z",
                    "file_path": f"{temp_dir}/pages/Technical/advanced.md",
                    "slug": "advanced_topics"
                }
            ]
            
            storage.update_index(pages)
            
            # Build llms.txt
            llms_txt_path = storage.build_llms_txt()
            
            # Verify file exists
            assert Path(llms_txt_path).exists()
            assert llms_txt_path.endswith("llms.txt")
            
            # Verify content
            with open(llms_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check header
            assert "Alliance Documentation - Page Directory" in content
            assert "Total pages: 2" in content
            
            # Check page entries (should be sorted by title)
            assert "Advanced Topics (Technical)" in content
            assert "Getting Started (General)" in content
            assert "https://example.com/Getting_Started" in content
            assert "https://example.com/Advanced_Topics" in content
    
    def test_build_llms_full_txt_uncompressed(self):
        """Test building llms_full.txt without compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Create test pages with actual content
            pages = []
            for i in range(2):
                page_data = {
                    "title": f"Test Page {i+1}",
                    "url": f"https://example.com/Page_{i+1}",
                    "pageid": i+1,
                    "lastmodified": "2025-01-01T00:00:00Z"
                }
                
                markdown_content = f"""---
title: "Test Page {i+1}"
url: "https://example.com/Page_{i+1}"
category: "General"
last_modified: "2025-01-01T00:00:00Z"
---

# Test Page {i+1}

This is test content for page {i+1}.
"""
                
                file_path = storage.save_page(page_data, markdown_content)
                
                pages.append({
                    "page_id": i+1,
                    "title": f"Test Page {i+1}",
                    "url": f"https://example.com/Page_{i+1}",
                    "category": "General",
                    "last_modified": "2025-01-01T00:00:00Z",
                    "file_path": file_path,
                    "slug": f"test_page_{i+1}"
                })
            
            storage.update_index(pages)
            
            # Build llms_full.txt without compression
            llms_full_path = storage.build_llms_full_txt(compress=False)
            
            # Verify file exists
            assert Path(llms_full_path).exists()
            assert llms_full_path.endswith("llms_full.txt")
            assert not llms_full_path.endswith(".gz")
            
            # Verify content
            with open(llms_full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check header
            assert "Alliance Documentation - Full Content" in content
            assert "Total pages: 2" in content
            
            # Check page content
            assert "PAGE: Test Page 1" in content
            assert "PAGE: Test Page 2" in content
            assert "This is test content for page 1" in content
            assert "This is test content for page 2" in content
    
    def test_build_llms_full_txt_compressed(self):
        """Test building llms_full.txt with compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DocumentationStorage(temp_dir)
            
            # Create test page
            page_data = {
                "title": "Test Page",
                "url": "https://example.com/Test",
                "pageid": 1,
                "lastmodified": "2025-01-01T00:00:00Z"
            }
            
            markdown_content = """---
title: "Test Page"
url: "https://example.com/Test"
category: "General"
---

# Test Page

This is test content.
"""
            
            file_path = storage.save_page(page_data, markdown_content)
            
            pages = [{
                "page_id": 1,
                "title": "Test Page",
                "url": "https://example.com/Test",
                "category": "General",
                "last_modified": "2025-01-01T00:00:00Z",
                "file_path": file_path,
                "slug": "test_page"
            }]
            
            storage.update_index(pages)
            
            # Build llms_full.txt with compression
            llms_full_path = storage.build_llms_full_txt(compress=True)
            
            # Verify file exists and is compressed
            assert Path(llms_full_path).exists()
            assert llms_full_path.endswith(".txt.gz")
            
            # Verify content by decompressing
            with gzip.open(llms_full_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            
            # Check content
            assert "Alliance Documentation - Full Content" in content
            assert "PAGE: Test Page" in content
            assert "This is test content" in content
