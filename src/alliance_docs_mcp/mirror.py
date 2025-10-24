"""MediaWiki API client for fetching documentation pages."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

TARGET_LANGUAGE = "en"
TRANSLATION_SUFFIXES = {TARGET_LANGUAGE, "fr"}
SKIP_TITLE_SUBSTRINGS = ("wheels",)


def _get_translation_suffix(title: str) -> Optional[str]:
    """Return translation suffix if title is a known translation subpage."""
    if not title or "/" not in title:
        return None
    suffix = title.rsplit("/", 1)[1].strip().lower()
    if suffix in TRANSLATION_SUFFIXES:
        return suffix
    return None


def _should_skip_title(title: str) -> bool:
    """Return True if the page title should be skipped during syncing."""
    if not title:
        return False
    normalized = title.strip().lower()
    return any(substring in normalized for substring in SKIP_TITLE_SUBSTRINGS)


def filter_to_target_language(pages: List[Dict], target_language: str = TARGET_LANGUAGE) -> List[Dict]:
    """Filter pages to include only the target language and base pages without translations."""
    english_titles = {
        page.get("title", "").strip().lower()
        for page in pages
        if not _should_skip_title(page.get("title", ""))
        if _get_translation_suffix(page.get("title", "")) == target_language
    }
    
    filtered_pages: List[Dict] = []
    for page in pages:
        title = page.get("title", "")
        normalized_title = title.strip()

        if _should_skip_title(normalized_title):
            continue

        suffix = _get_translation_suffix(normalized_title)
        
        if suffix and suffix != target_language:
            continue
        
        if not suffix:
            candidate = f"{normalized_title}/{target_language}".lower()
            if candidate in english_titles:
                continue
        
        filtered_pages.append(page)
    
    return filtered_pages


class MediaWikiClient:
    """Client for interacting with MediaWiki API."""
    
    def __init__(self, api_url: str, user_agent: str = "AllianceDocsMCP/1.0"):
        """Initialize the MediaWiki client.
        
        Args:
            api_url: Base URL for the MediaWiki API
            user_agent: User agent string for requests
        """
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _make_request(self, params: Dict) -> Dict:
        """Make a request to the MediaWiki API.
        
        Args:
            params: Query parameters for the API request
            
        Returns:
            JSON response from the API
            
        Raises:
            requests.RequestException: If the request fails
        """
        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_all_pages(
        self,
        limit: int = 500,
        continue_token: Optional[str] = None,
        redirect_filter: str = "nonredirects",
    ) -> Tuple[List[Dict], Optional[str]]:
        """Get a list of all pages from the wiki.
        
        Args:
            limit: Maximum number of pages to fetch per request
            continue_token: Token for pagination
            redirect_filter: Filter pages by redirect status (e.g., "nonredirects")
            
        Returns:
            Tuple of (pages list, next continue token)
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "aplimit": limit,
            "apnamespace": 0,  # Main namespace only
        }
        
        if redirect_filter:
            params["apfilterredir"] = redirect_filter
        
        if continue_token:
            params["apcontinue"] = continue_token
        
        response = self._make_request(params)
        
        pages = []
        if "query" in response and "allpages" in response["query"]:
            pages = response["query"]["allpages"]
        
        next_token = None
        if "continue" in response and "apcontinue" in response["continue"]:
            next_token = response["continue"]["apcontinue"]
        
        return pages, next_token
    
    def get_page_content(self, page_id: int) -> Optional[Dict]:
        """Get the content of a specific page.
        
        Args:
            page_id: The page ID to fetch
            
        Returns:
            Dictionary with page content and metadata, or None if not found
        """
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions|info",
            "pageids": page_id,
            "rvprop": "content|timestamp",
            "rvlimit": 1,
            "inprop": "url|displaytitle"
        }
        
        response = self._make_request(params)
        
        if "query" not in response or "pages" not in response["query"]:
            return None
        
        pages = response["query"]["pages"]
        if str(page_id) not in pages:
            return None
        
        page_data = pages[str(page_id)]
        if "revisions" not in page_data or not page_data["revisions"]:
            return None
        
        revision = page_data["revisions"][0]
        
        return {
            "pageid": page_id,
            "title": page_data.get("title", ""),
            "url": page_data.get("fullurl", ""),
            "displaytitle": page_data.get("displaytitle", ""),
            "content": revision.get("*", ""),
            "timestamp": revision.get("timestamp", ""),
            "lastmodified": revision.get("timestamp", "")
        }
    
    def get_page_by_title(self, title: str) -> Optional[Dict]:
        """Get a page by its title.
        
        Args:
            title: The page title to fetch
            
        Returns:
            Dictionary with page content and metadata, or None if not found
        """
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions|info",
            "titles": title,
            "rvprop": "content|timestamp",
            "rvlimit": 1,
            "inprop": "url|displaytitle"
        }
        
        response = self._make_request(params)
        
        if "query" not in response or "pages" not in response["query"]:
            return None
        
        pages = response["query"]["pages"]
        # Find the page (could be -1 if not found)
        for page_id, page_data in pages.items():
            if page_id != "-1" and "revisions" in page_data and page_data["revisions"]:
                revision = page_data["revisions"][0]
                return {
                    "pageid": int(page_id),
                    "title": page_data.get("title", ""),
                    "url": page_data.get("fullurl", ""),
                    "displaytitle": page_data.get("displaytitle", ""),
                    "content": revision.get("*", ""),
                    "timestamp": revision.get("timestamp", ""),
                    "lastmodified": revision.get("timestamp", "")
                }
        
        return None
    
    def get_recent_changes(
        self,
        since: Optional[datetime] = None,
        limit: int = 50,
        redirect_filter: str = "!redirect",
    ) -> List[Dict]:
        """Get recently changed pages.
        
        Args:
            since: Only get changes since this datetime
            limit: Maximum number of changes to fetch
            redirect_filter: Filter changes by redirect status (MediaWiki rcshow syntax)
            
        Returns:
            List of recently changed pages
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "recentchanges",
            "rcnamespace": 0,
            "rclimit": limit,
            "rcprop": "title|timestamp|ids"
        }
        
        if since:
            params["rcstart"] = since.isoformat()
        
        if redirect_filter:
            params["rcshow"] = redirect_filter
        
        response = self._make_request(params)
        
        if "query" not in response or "recentchanges" not in response["query"]:
            return []
        
        return response["query"]["recentchanges"]
    
    def close(self):
        """Close the session."""
        self.session.close()


async def fetch_all_pages(client: MediaWikiClient) -> List[Dict]:
    """Fetch all pages from the wiki asynchronously.
    
    Args:
        client: MediaWiki client instance
        
    Returns:
        List of all pages with basic metadata
    """
    all_pages = []
    continue_token = None
    
    logger.info("Starting to fetch all pages from MediaWiki")
    
    while True:
        try:
            pages, next_token = client.get_all_pages(
                limit=500,
                continue_token=continue_token,
                redirect_filter="nonredirects",
            )
            all_pages.extend(pages)
            logger.info(f"Fetched {len(pages)} pages, total: {len(all_pages)}")
            
            if not next_token:
                break
            
            continue_token = next_token
            # Small delay to be respectful to the API
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching pages: {e}")
            break
    
    filtered_pages = filter_to_target_language(all_pages)
    logger.info(
        "Completed fetching %d pages after filtering (removed %d non-target pages)",
        len(filtered_pages),
        len(all_pages) - len(filtered_pages),
    )
    return filtered_pages


async def fetch_page_contents(client: MediaWikiClient, page_ids: List[int]) -> List[Dict]:
    """Fetch content for multiple pages.
    
    Args:
        client: MediaWiki client instance
        page_ids: List of page IDs to fetch
        
    Returns:
        List of pages with full content
    """
    pages_with_content = []
    
    # Process in batches to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(page_ids), batch_size):
        batch = page_ids[i:i + batch_size]
        
        for page_id in batch:
            try:
                content = client.get_page_content(page_id)
                if content:
                    pages_with_content.append(content)
                await asyncio.sleep(0.1)  # Small delay between requests
            except Exception as e:
                logger.error(f"Error fetching content for page {page_id}: {e}")
        
        # Longer delay between batches
        if i + batch_size < len(page_ids):
            await asyncio.sleep(1)
    
    return pages_with_content
