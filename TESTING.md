# Testing the Alliance Docs MCP Server

This guide explains how to test that the MCP server is working correctly.

## 🧪 **Testing Methods**

### **1. Documentation Sync Test**

First, verify that the documentation mirroring is working:

```bash
# Run the sync with verbose output
uv run python scripts/sync_docs.py --verbose
```

**Expected Results:**
- ✅ Should show progress bars and status updates
- ✅ Should download and process ~419 pages
- ✅ Should create `docs/index.json` and `docs/pages/` directory
- ✅ Should show a summary table with statistics

### **2. MCP Server Startup Test**

Test that the MCP server starts correctly:

```bash
# Test server startup (will show FastMCP banner)
uv run python -m alliance_docs_mcp.server --verbose
```

**Expected Results:**
- ✅ Should show "Starting Alliance Docs MCP Server" message
- ✅ Should display FastMCP banner with server info
- ✅ Should show "Starting MCP server 'Alliance Docs' with transport 'stdio'"

### **3. MCP Protocol Test**

Test the MCP protocol communication:

```bash
# Send an initialize request to the server
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | uv run python -m alliance_docs_mcp.server
```

**Expected Results:**
- ✅ Should return a JSON response with server capabilities
- ✅ Should include `resources` and `tools` in capabilities
- ✅ Should show server info: `"name":"Alliance Docs"`

### **4. Search Functionality Test**

Test the search capabilities:

```bash
# Test search functionality
uv run python -c "
from src.alliance_docs_mcp.storage import DocumentationStorage
storage = DocumentationStorage('docs')
results = storage.search_pages('python')
print(f'Found {len(results)} results for \"python\"')
for result in results[:3]:
    print(f'- {result.get(\"title\", \"Unknown\")}')
"
```

**Expected Results:**
- ✅ Should find pages containing "python"
- ✅ Should return structured results with titles and metadata

### **5. MCP Protocol Integration Test**

The MCP server is designed to work with MCP clients (like Cursor, Claude Desktop, etc.). The server:

- ✅ **Starts correctly** and shows the FastMCP banner
- ✅ **Responds to initialize requests** with proper capabilities
- ✅ **Exposes resources** for individual documentation pages
- ✅ **Provides tools** for search and category listing
- ✅ **Handles stdio communication** as expected by MCP clients

**Note**: The server is designed to work with MCP clients, not direct command-line testing. The initialization response shows the server is working correctly.

## 🔍 **Verification Checklist**

- [ ] **Documentation Sync**: 419 pages downloaded and processed
- [ ] **File Structure**: `docs/index.json` and `docs/pages/` created
- [ ] **MCP Server**: Starts without errors
- [ ] **Protocol**: Responds to initialize requests
- [ ] **Resources**: Lists individual documentation pages
- [ ] **Tools**: Exposes search and category tools
- [ ] **Search**: Returns relevant results for queries

## 🚨 **Troubleshooting**

### Common Issues:

1. **"No module named 'fastmcp'"**
   - Solution: Make sure to use `uv run python` instead of just `python`

2. **"Documentation directory not found"**
   - Solution: Run the sync script first: `uv run python scripts/sync_docs.py`

3. **"No pages found"**
   - Solution: Check that `docs/index.json` exists and contains page data

4. **Server won't start**
   - Solution: Check that all dependencies are installed: `uv sync`

### Debug Mode:

For detailed debugging, use verbose mode:

```bash
uv run python -m alliance_docs_mcp.server --verbose
```

This will show detailed logging information about server startup and operations.

## 📊 **Performance Expectations**

- **Sync Time**: ~3-5 minutes for 419 pages
- **Server Startup**: < 2 seconds
- **Search Response**: < 1 second for typical queries
- **Resource Listing**: < 1 second

## ✅ **Success Indicators**

The MCP server is working correctly when:

1. ✅ Documentation sync completes successfully
2. ✅ Server starts and shows FastMCP banner
3. ✅ MCP protocol responds to initialize requests
4. ✅ Resources and tools are properly listed
5. ✅ Search returns relevant results
6. ✅ No error messages in logs

If all these tests pass, your Alliance Docs MCP server is ready for use! 🎉
