"""Tests for documentation storage functionality."""

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
