# Alliance Documentation MCP Server

A Model Context Protocol (MCP) server that provides programmatic access to the Digital Research Alliance of Canada's technical documentation. This server mirrors the documentation from the MediaWiki site and exposes it through MCP resources and tools for use with MCP-compatible clients.

## Features

- **Documentation Mirroring**: Syncs documentation from the Alliance MediaWiki site
- **MCP Resources**: Exposes individual documentation pages as MCP resources
- **Search & Query Tools**: Provides search, categorization, and querying capabilities
- **Startup Refresh**: Container entrypoint triggers an incremental sync on boot; schedule additional runs as needed
- **Markdown Storage**: Stores documentation as markdown files with metadata

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management

### Installation

1. **Clone and setup the repository:**
   ```bash
   git clone <repository-url>
   cd alliance-docs-mcp
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment (optional):**
   Create a `.env` file (or export the variables directly) if you want to override defaults. For example:
   ```env
   MEDIAWIKI_API_URL=https://docs.alliancecan.ca/mediawiki/api.php
   DOCS_DIR=./docs
   USER_AGENT=AllianceDocsMCP/1.0
   ```

4. **Initial documentation sync:**
   ```bash
   uv run python scripts/sync_docs.py
   ```
   _Note: Docker images built from this repository automatically run this full sync during the image build so containers start with a warm cache._

5. **Start the MCP server:**
   ```bash
   uv run python -m alliance_docs_mcp.server
   ```

## Usage

### MCP Resources

The server exposes documentation pages as MCP resources:

- **Resource URI**: `alliance-docs://page/{slug}`
- **Content**: Markdown content of the documentation page

Example:
```
alliance-docs://page/technical_documentation
```

### MCP Tools

The server provides several tools for querying documentation:

#### `search_docs(query: str, category: Optional[str] = None)`
Search documentation pages by content and title.

**Parameters:**
- `query`: Search query string
- `category`: Optional category filter

**Returns:** List of matching pages with metadata

#### `list_categories()`
List all available documentation categories.

**Returns:** List of category names

#### `get_page_by_title(title: str)`
Find a specific page by its title.

**Parameters:**
- `title`: Page title to search for

**Returns:** Page metadata or None if not found

#### `list_recent_updates(limit: int = 10)`
List recently updated pages.

**Parameters:**
- `limit`: Maximum number of pages to return

**Returns:** List of recent pages with metadata

#### `get_page_info(slug: str)`
Get detailed information about a specific page.

**Parameters:**
- `slug`: Page slug

**Returns:** Detailed page information including metadata

#### `list_all_pages()`
List all available documentation pages.

**Returns:** List of all pages with basic metadata

### Synchronization

#### Manual Sync

Run a full synchronization (with rich progress bars and visual feedback):
```bash
uv run python scripts/sync_docs.py
```

Run an incremental sync (only changed pages):
```bash
uv run python scripts/sync_docs.py --incremental
```

For FastMCP Cloud deployments, run one of the sync commands above locally and commit the updated `docs/` directory before pushing so the hosted server always mirrors the latest content.

The sync script provides:
- **Colored output** with rich formatting
- **Progress bars** for download and processing phases
- **Real-time statistics** including pages/second
- **Summary table** with detailed metrics
- **Error tracking** with warnings for failed pages

> **Note:** Markdown pages larger than 10 MB are stored as `.md.gz` files. The server automatically decompresses them at runtime, so no additional configuration is required.

#### LLM-Optimized Documentation Files

The sync process automatically generates two files for LLM consumption:

- **`docs/llms.txt`**: A simple directory listing all page names, categories, and URLs (~35 KB)
- **`docs/llms_full.txt.gz`**: Complete documentation content in a single compressed file (~2.6 MB compressed, ~393 MB uncompressed)

These files are regenerated on every sync (both full and incremental) and committed to the repository, making it easy for LLMs to access the entire documentation corpus.

#### Automated Sync

Set up a cron job for weekly updates:
```bash
# Add to crontab (runs every Sunday at 2 AM)
0 2 * * 0 cd /path/to/alliance-docs-mcp && uv run python scripts/sync_docs.py --incremental
```

This repository also ships with `.github/workflows/weekly-sync.yml`, which performs the same incremental sync on Sundays using GitHub Actions and pushes any changes back to `main`.

## Configuration

### Environment Variables

Set the following environment variables (via `.env`, shell exports, or your hosting platform's secret manager) to customize behavior:

- `MEDIAWIKI_API_URL` (default `https://docs.alliancecan.ca/mediawiki/api.php`)
- `DOCS_DIR` (default `./docs`, or `/data/docs` in the container)
- `USER_AGENT` (default `AllianceDocsMCP/1.0`)

### Server Configuration

The MCP server can be configured with command-line arguments:

```bash
uv run python -m alliance_docs_mcp.server --help
```

Options:
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to bind to (default: 8000)
- `--docs-dir`: Documentation directory (default: ./docs)

### Docker Deployment

The provided Docker image ships with a pre-synced documentation cache baked into `/app/docs_seed`. When the container starts, the entrypoint primes the configured `DOCS_DIR` from this seed (if empty) and then launches the MediaWiki sync in the background so the MCP server begins accepting connections immediately. You can configure startup behavior with:

- `RUN_SYNC_ON_START=0` to skip the background sync (useful when running in read-only environments)
- `SYNC_MODE=full` to force a full resync instead of the default incremental sync
- The container starts the server via `fastmcp run server_entrypoint.py:mcp --transport http --path /mcp/ --port 8080`, so any additional FastMCP CLI flags can be injected by overriding `CMD` in your own image if needed.
- A lightweight `/health` endpoint is exposed for platform probes; point load balancer checks there instead of MCP protocol paths.

## Project Structure

```
alliance-docs-mcp/
├── src/
│   └── alliance_docs_mcp/
│       ├── __init__.py
│       ├── server.py        # FastMCP server implementation
│       ├── mirror.py        # MediaWiki API client
│       ├── converter.py     # WikiText to Markdown converter
│       └── storage.py       # File storage and retrieval
├── docs/                    # Mirrored markdown files
│   ├── pages/               # Organized by category
│   └── index.json           # Page metadata index
├── scripts/
│   └── sync_docs.py         # Synchronization script
├── tests/                   # Test files
├── pyproject.toml           # Project configuration
└── README.md
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run ruff check src/
```

### Deployment Options

**FastMCP Cloud (managed)**
- Sign in at [fastmcp.cloud](https://fastmcp.cloud) with your GitHub account and create a project that points at this repository.
- Use `server_entrypoint.py:mcp` as the entrypoint so the platform runs the exported FastMCP server instance.
- Configure environment variables (e.g., `MEDIAWIKI_API_URL`, `DOCS_DIR`, `USER_AGENT`) via the project settings; the service installs dependencies directly from `pyproject.toml`.
- Push to `main` to trigger deployments; each pull request automatically gets its own preview environment for testing changes.

**Self-managed container/VM**
- Build the Docker image in this repo and run it anywhere that can expose HTTP on port `8080`.
- Provide the same environment variables via your scheduler or container runtime.
- Point load balancer health checks at `/health` and connect MCP clients to the `/mcp/` path served by `fastmcp run`.

### Adding New Features

1. **New MCP Tools**: Add new tool functions to `server.py`
2. **Storage Enhancements**: Extend `storage.py` for new functionality
3. **API Improvements**: Modify `mirror.py` for different API interactions

## Troubleshooting

### Common Issues

1. **Sync Failures**: Check API access and network connectivity
2. **Missing Pages**: Verify MediaWiki API responses
3. **Conversion Errors**: Check pandoc installation and WikiText format

### Logs

Check the `sync.log` file for synchronization issues:
```bash
tail -f sync.log
```

### Debug Mode

Run with verbose logging:
```bash
uv run python scripts/sync_docs.py --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Digital Research Alliance of Canada](https://alliancecan.ca/) for providing the documentation
- [FastMCP](https://github.com/jlowin/fastmcp) for the MCP server framework
- [uv](https://github.com/astral-sh/uv) for Python package management
