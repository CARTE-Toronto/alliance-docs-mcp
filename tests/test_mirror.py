"""Tests for MediaWiki mirroring functionality."""

import pytest
from unittest.mock import Mock, patch

from alliance_docs_mcp.mirror import MediaWikiClient, filter_to_target_language


class TestMediaWikiClient:
    """Test cases for MediaWikiClient."""
    
    def test_init(self):
        """Test client initialization."""
        client = MediaWikiClient("https://example.com/api.php", "TestAgent/1.0")
        assert client.api_url == "https://example.com/api.php"
        assert client.session.headers["User-Agent"] == "TestAgent/1.0"
        client.close()
    
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.json.return_value = {"query": {"pages": []}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = MediaWikiClient("https://example.com/api.php")
        result = client._make_request({"action": "query"})
        
        assert result == {"query": {"pages": []}}
        client.close()
    
    @patch('requests.Session.get')
    def test_make_request_failure(self, mock_get):
        """Test API request failure."""
        mock_get.side_effect = Exception("Network error")
        
        client = MediaWikiClient("https://example.com/api.php")
        
        with pytest.raises(Exception):
            client._make_request({"action": "query"})
        
        client.close()
    
    @patch('requests.Session.get')
    def test_get_all_pages(self, mock_get):
        """Test getting all pages."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "allpages": [
                    {"pageid": 1, "title": "Test Page 1"},
                    {"pageid": 2, "title": "Test Page 2"}
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = MediaWikiClient("https://example.com/api.php")
        pages, next_token = client.get_all_pages()
        
        assert len(pages) == 2
        assert pages[0]["title"] == "Test Page 1"
        assert next_token is None
        client.close()
    
    @patch('requests.Session.get')
    def test_get_page_content(self, mock_get):
        """Test getting page content."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "title": "Test Page",
                        "fullurl": "https://example.com/Test_Page",
                        "revisions": [{
                            "*": "Test content",
                            "timestamp": "2025-01-01T00:00:00Z"
                        }]
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = MediaWikiClient("https://example.com/api.php")
        content = client.get_page_content(123)
        
        assert content is not None
        assert content["title"] == "Test Page"
        assert content["content"] == "Test content"
        client.close()

    def test_filter_to_target_language(self):
        """Ensure only English pages are kept."""
        pages = [
            {"pageid": 1, "title": "ABINIT"},
            {"pageid": 2, "title": "ABINIT/en"},
            {"pageid": 3, "title": "ABINIT/fr"},
            {"pageid": 4, "title": "Standalone Page"},
            {"pageid": 5, "title": "Guide/Section"},
            {"pageid": 6, "title": "Guide/Section/en"},
            {"pageid": 7, "title": "Guide/Section/fr"},
        ]
        
        filtered = filter_to_target_language(pages)
        filtered_ids = [page["pageid"] for page in filtered]
        
        assert filtered_ids == [2, 4, 6]
