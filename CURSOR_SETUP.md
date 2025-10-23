# Testing Alliance Docs MCP Server with Cursor

This guide shows how to run the MCP server locally and test it with Cursor.

## 🚀 Quick Start

### 1. **Sync Documentation First**
```bash
# Set the correct API URL and sync documentation
export MEDIAWIKI_API_URL=https://docs.alliancecan.ca/mediawiki/api.php
uv run python scripts/sync_docs.py
```

### 2. **Configure Cursor with MCP**

#### Option A: Copy the provided mcp.json
The project includes a ready-to-use `mcp.json` file. Copy it to your Cursor settings directory:

```bash
# Copy the MCP configuration to Cursor settings
cp mcp.json ~/Library/Application\ Support/Cursor/User/globalStorage/mcp.json
```

#### Option B: Manual Configuration
Create an `mcp.json` file in your Cursor settings directory:

```json
{
  "mcpServers": {
    "alliance-docs": {
      "command": "uv",
      "args": ["run", "python", "-m", "alliance_docs_mcp.server"],
      "cwd": "/Users/alex/repos/alliance-docs-mcp",
      "env": {
        "MEDIAWIKI_API_URL": "https://docs.alliancecan.ca/mediawiki/api.php",
        "DOCS_DIR": "./docs"
      }
    }
  }
}
```

### 3. **Restart Cursor**
After adding the MCP configuration, restart Cursor to load the new server.

#### Option B: Test MCP Client
```bash
# Test the MCP server functionality
uv run python test_mcp_client.py
```

## 🧪 Testing the Server

### Available MCP Resources:
- `alliance-docs://page/{slug}` - Individual documentation pages

### Available MCP Tools:
- `search_docs(query, category)` - Search documentation
- `list_categories()` - List all categories
- `get_page_by_title(title)` - Find page by title
- `list_recent_updates(limit)` - Recent updates
- `get_page_info(slug)` - Page details
- `list_all_pages()` - All pages

### Test Commands in Cursor:
```
# Search for documentation
@alliance-docs search_docs "getting started"

# List categories
@alliance-docs list_categories

# Get a specific page
@alliance-docs get_page_by_title "Technical documentation"

# List recent updates
@alliance-docs list_recent_updates 5
```

## 🔧 Development Testing

### Run Tests:
```bash
uv run python -m pytest tests/ -v
```

### Test MCP Client:
```bash
uv run python test_mcp_client.py
```

### Check Server Status:
```bash
# Check if server is running
curl http://localhost:8000/health  # If using HTTP mode
```

## 📊 Monitoring

### View Sync Logs:
```bash
tail -f sync.log
```

### Check Documentation:
```bash
# View synced documentation
ls -la docs/pages/
cat docs/index.json | jq '.total_pages'
```

## 🐛 Troubleshooting

### Common Issues:

1. **No documentation found**:
   ```bash
   # Re-run sync with verbose output
   uv run python scripts/sync_docs.py --verbose
   ```

2. **MCP connection failed**:
   - Check server is running: `ps aux | grep alliance_docs_mcp`
   - Check logs: `tail -f sync.log`

3. **API errors**:
   - Verify API URL: `curl https://docs.alliancecan.ca/mediawiki/api.php?action=query&format=json&meta=siteinfo`

### Debug Mode:
```bash
# Run with debug logging
export PYTHONPATH=src
uv run python -m alliance_docs_mcp.server --verbose
```

## 📝 Example Usage

Once connected to Cursor, you can ask questions like:

- "What documentation is available about getting started?"
- "Show me the technical documentation page"
- "What are the recent updates to the documentation?"
- "Search for information about machine learning"

The MCP server will provide relevant documentation content directly in your Cursor chat!
