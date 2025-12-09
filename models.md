# Alliance Documentation MCP Server

## Architecture Overview

This project creates an MCP (Model Context Protocol) server that provides programmatic access to the Digital Research Alliance of Canada's technical documentation. The system has two main components:

1. **Documentation Mirroring Service**: Fetches and converts MediaWiki pages to markdown files
2. **MCP Server**: Exposes documentation via MCP resources and tools using fastmcp

## Component Details

### 1. Repository Structure

```
alliance-docs-mcp/
├── pyproject.toml           # uv project configuration
├── .python-version          # Python 3.11+
├── README.md                # Setup and usage instructions
├── models.md                # This architecture document
├── src/
│   └── alliance_docs_mcp/
│       ├── __init__.py
│       ├── server.py        # FastMCP server implementation
│       ├── mirror.py        # MediaWiki mirroring logic
│       ├── converter.py     # WikiText to Markdown conversion
│       └── storage.py       # File storage and retrieval
├── docs/                    # Mirrored markdown files
│   └── pages/               # Organized by category
├── scripts/
│   └── sync_docs.py         # Weekly sync script
└── tests/
```

### 2. Documentation Mirroring (mirror.py)

**Approach**: Use MediaWiki API directly via requests library

**Key API Endpoints**:
- `api.php?action=query&list=allpages` - List all pages
- `api.php?action=query&prop=revisions&rvprop=content` - Get page content
- `api.php?action=parse` - Parse wikitext to HTML if needed

**Process**:
1. Fetch list of all pages from `docs.alliancecan.ca/wiki/api.php`
2. For each page, retrieve raw wikitext content
3. Convert wikitext to markdown using a pure-Python parser (wikitextparser) with HTML stripping
4. Save to `docs/pages/{category}/{page_name}.md` with metadata
5. Maintain an index file (`docs/index.json`) with page metadata:
   - Title, URL, category, last modified, file path

**Storage Format** (Markdown with frontmatter):
```markdown
---
title: "Technical documentation"
url: "https://docs.alliancecan.ca/wiki/Technical_documentation"
category: "General"
last_modified: "2025-09-05T13:10:00Z"
---

# Page Content
...
```

### 3. MCP Server Implementation (server.py)

**Using fastmcp framework**:

**MCP Resources** (individual pages):
- Each markdown file exposed as `alliance-docs://page/{page_slug}`
- Dynamically generated list from docs/index.json
- Returns markdown content when accessed

**MCP Tools** (search/query capabilities):
- `search_docs(query: str, category: Optional[str])`: Full-text search across all docs
- `list_categories()`: Returns available documentation categories
- `get_page_by_title(title: str)`: Find specific page by title
- `list_recent_updates(limit: int)`: Show recently updated pages

**Example fastmcp server structure**:
```python
from fastmcp import FastMCP

mcp = FastMCP("Alliance Docs")

@mcp.resource("alliance-docs://page/{slug}")
async def get_page(slug: str) -> str:
    # Load and return markdown content
    pass

@mcp.tool()
async def search_docs(query: str, category: str = None) -> list:
    # Search across markdown files
    pass
```

### 4. Scheduled Updates

**Frequency**: Weekly (via cron job or GitHub Actions)

**Script** (`scripts/sync_docs.py`):
- Check for new or modified pages via MediaWiki API timestamps
- Download only changed content to minimize API load
- Update docs/index.json
- Log sync results

**Cron setup** (example):
```bash
0 2 * * 0 cd /path/to/alliance-docs-mcp && uv run python scripts/sync_docs.py
```

**Alternative**: GitHub Actions workflow for automated weekly sync and commit

### 5. Dependencies

**Core**:
- `fastmcp` - MCP server framework
- `requests` - HTTP client for MediaWiki API
- `wikitextparser` + `beautifulsoup4` - WikiText to Markdown conversion and HTML cleanup
- `pyyaml` - Frontmatter parsing
- `python-dotenv` - Configuration management

**Development**:
- `pytest` - Testing framework
- `black` - Code formatting
- `ruff` - Linting

### 6. Configuration

**Environment variables** (`.env`):
```
MEDIAWIKI_API_URL=https://docs.alliancecan.ca/mediawiki/api.php
DOCS_DIR=./docs
SYNC_SCHEDULE=weekly
USER_AGENT=AllianceDocsMCP/1.0
```

### 7. Testing Strategy

- Unit tests for MediaWiki API interactions
- Integration tests for mirroring process
- MCP server endpoint tests using fastmcp testing utilities
- Validation that markdown conversion preserves content integrity

## Implementation Phases

This will be implemented in the following order to ensure each component works before building the next layer.

### To-dos

- [x] Initialize repository with uv, create project structure, setup pyproject.toml with dependencies
- [x] Implement MediaWiki API client in mirror.py to fetch page lists and content
- [x] Create WikiText to Markdown converter with frontmatter in converter.py
- [x] Build storage.py for saving markdown files and maintaining index.json
- [x] Develop sync_docs.py script that orchestrates mirroring process
- [x] Implement MCP resources in server.py to expose individual documentation pages
- [x] Add MCP tools for search, listing categories, and querying documentation
- [x] Create README.md with setup instructions and finalize models.md
