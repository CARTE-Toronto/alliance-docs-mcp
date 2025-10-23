"""WikiText to Markdown converter with frontmatter support."""

import re
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urljoin

import pypandoc


class WikiTextConverter:
    """Converts MediaWiki WikiText to Markdown with frontmatter."""
    
    def __init__(self, base_url: str = "https://docs.alliancecan.ca/wiki/"):
        """Initialize the converter.
        
        Args:
            base_url: Base URL for the wiki (for link generation)
        """
        self.base_url = base_url
    
    def convert_to_markdown(self, wikitext: str, metadata: Dict) -> str:
        """Convert WikiText to Markdown with frontmatter.
        
        Args:
            wikitext: The raw WikiText content
            metadata: Page metadata (title, url, etc.)
            
        Returns:
            Markdown content with frontmatter
        """
        # Clean up the wikitext first
        cleaned_wikitext = self._clean_wikitext(wikitext)
        
        # Convert using pandoc
        try:
            markdown_content = pypandoc.convert_text(
                cleaned_wikitext,
                'markdown',
                format='mediawiki',
                extra_args=['--wrap=none']
            )
        except Exception as e:
            # Fallback to basic conversion if pandoc fails
            markdown_content = self._basic_wikitext_to_markdown(cleaned_wikitext)
        
        # Process links to make them absolute
        markdown_content = self._process_links(markdown_content)
        
        # Create frontmatter
        frontmatter = self._create_frontmatter(metadata)
        
        return f"{frontmatter}\n\n{markdown_content}"
    
    def _clean_wikitext(self, wikitext: str) -> str:
        """Clean up common WikiText issues.
        
        Args:
            wikitext: Raw WikiText content
            
        Returns:
            Cleaned WikiText content
        """
        # Remove category links (they're usually at the end)
        wikitext = re.sub(r'\[\[Category:[^\]]+\]\]', '', wikitext)
        
        # Remove template calls that might cause issues
        wikitext = re.sub(r'\{\{[^}]*\}\}', '', wikitext)
        
        # Clean up multiple newlines
        wikitext = re.sub(r'\n{3,}', '\n\n', wikitext)
        
        return wikitext.strip()
    
    def _basic_wikitext_to_markdown(self, wikitext: str) -> str:
        """Basic WikiText to Markdown conversion without pandoc.
        
        Args:
            wikitext: WikiText content
            
        Returns:
            Basic Markdown content
        """
        # Convert headers
        wikitext = re.sub(r'^= (.+) =$', r'# \1', wikitext, flags=re.MULTILINE)
        wikitext = re.sub(r'^== (.+) ==$', r'## \1', wikitext, flags=re.MULTILINE)
        wikitext = re.sub(r'^=== (.+) ===$', r'### \1', wikitext, flags=re.MULTILINE)
        wikitext = re.sub(r'^==== (.+) ====$', r'#### \1', wikitext, flags=re.MULTILINE)
        
        # Convert bold and italic
        wikitext = re.sub(r"'''(.*?)'''", r'**\1**', wikitext)
        wikitext = re.sub(r"''(.*?)''", r'*\1*', wikitext)
        
        # Convert links
        wikitext = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'[\2](\1)', wikitext)
        wikitext = re.sub(r'\[\[([^\]]+)\]\]', r'[\1](\1)', wikitext)
        
        # Convert external links
        wikitext = re.sub(r'\[([^\s\]]+) ([^\]]+)\]', r'[\2](\1)', wikitext)
        
        # Convert lists
        wikitext = re.sub(r'^\* ', r'- ', wikitext, flags=re.MULTILINE)
        wikitext = re.sub(r'^# ', r'1. ', wikitext, flags=re.MULTILINE)
        
        # Convert code blocks
        wikitext = re.sub(r'<pre>(.*?)</pre>', r'```\n\1\n```', wikitext, flags=re.DOTALL)
        wikitext = re.sub(r'<code>(.*?)</code>', r'`\1`', wikitext)
        
        return wikitext
    
    def _process_links(self, markdown: str) -> str:
        """Process links to make them absolute where appropriate.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Markdown with processed links
        """
        # Convert relative wiki links to absolute URLs
        def replace_wiki_link(match):
            link_text = match.group(1)
            link_url = match.group(2)
            
            # If it's already an absolute URL, keep it
            if link_url.startswith(('http://', 'https://')):
                return match.group(0)
            
            # If it's a wiki page, make it absolute
            if not link_url.startswith('/'):
                link_url = f"/{link_url}"
            
            absolute_url = urljoin(self.base_url, link_url)
            return f'[{link_text}]({absolute_url})'
        
        # Match markdown links
        markdown = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_wiki_link, markdown)
        
        return markdown
    
    def _create_frontmatter(self, metadata: Dict) -> str:
        """Create YAML frontmatter for the page.
        
        Args:
            metadata: Page metadata
            
        Returns:
            YAML frontmatter string
        """
        frontmatter_data = {
            "title": metadata.get("title", ""),
            "url": metadata.get("url", ""),
            "category": self._extract_category(metadata.get("title", "")),
            "last_modified": metadata.get("lastmodified", ""),
            "page_id": metadata.get("pageid", ""),
            "display_title": metadata.get("displaytitle", ""),
        }
        
        # Remove empty values
        frontmatter_data = {k: v for k, v in frontmatter_data.items() if v}
        
        # Convert to YAML
        yaml_lines = ["---"]
        for key, value in frontmatter_data.items():
            if isinstance(value, str):
                # Escape quotes in strings
                value = value.replace('"', '\\"')
                yaml_lines.append(f'{key}: "{value}"')
            else:
                yaml_lines.append(f"{key}: {value}")
        yaml_lines.append("---")
        
        return "\n".join(yaml_lines)
    
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
    
    def extract_links(self, wikitext: str) -> list:
        """Extract all links from WikiText.
        
        Args:
            wikitext: WikiText content
            
        Returns:
            List of link dictionaries
        """
        links = []
        
        # Extract internal links [[page|text]]
        internal_links = re.findall(r'\[\[([^|\]]+)(?:\|([^\]]+))?\]\]', wikitext)
        for link in internal_links:
            page, text = link
            links.append({
                "type": "internal",
                "page": page,
                "text": text or page,
                "url": urljoin(self.base_url, page.replace(" ", "_"))
            })
        
        # Extract external links [url text]
        external_links = re.findall(r'\[([^\s\]]+) ([^\]]+)\]', wikitext)
        for link in external_links:
            url, text = link
            links.append({
                "type": "external",
                "url": url,
                "text": text
            })
        
        return links
