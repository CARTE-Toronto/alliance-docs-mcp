---
name: Next Version Features
overview: Analyze current Alliance Docs MCP server implementation and FastMCP capabilities to recommend valuable features for the next version, including prompts, structured schemas, middleware, and enhanced user interactions.
todos:
  - id: structured-schemas
    content: Create Pydantic models for tool responses (PageMetadata, SearchResult, RelatedPageResult, CategoryInfo) and apply to tool return types
    status: pending
  - id: caching-middleware
    content: Implement caching middleware with configurable TTLs for different tool types, add environment variable to control caching
    status: pending
  - id: mcp-prompts
    content: Add MCP prompts for common documentation query patterns (search, related content, category exploration, technical questions)
    status: pending
  - id: progress-reporting
    content: Add progress reporting to long-running operations (list_all_pages, complex searches)
    status: pending
  - id: user-elicitation
    content: Implement user elicitation for vague queries and result filtering in search_docs and find_related_pages
    status: pending
  - id: batch-operations
    content: Add batch operation tools (get_multiple_pages, search_multiple_categories) to reduce client round trips
    status: pending
  - id: enhanced-health
    content: Enhance /health endpoint with index status, document counts, last sync timestamp, and server metadata
    status: pending
---

# Next Version Feature Recommendations for Alliance Docs MCP Server

## Current State Analysis

The server currently implements:

- **8 MCP Tools**: search, list_categories, get_page_by_title, list_recent_updates, get_page_info, list_all_pages, find_related_pages, get_page_content
- **MCP Resources**: TextResource for each documentation page
- **Custom Routes**: `/health` and `/` endpoints
- **Search Infrastructure**: Whoosh full-text search with highlights
- **Related Pages**: Embeddings-based (Chroma + sentence-transformers) with heuristic fallback

**Missing FastMCP Features:**

- Prompts (reusable message templates)
- Structured output schemas (Pydantic models)
- User elicitation (interactive workflows)
- Middleware (caching, resource tools)
- Progress reporting for long operations
- Server composition/proxying capabilities

## Recommended Features for Next Version

### 1. MCP Prompts (High Value)

**Why**: Prompts provide reusable templates that guide LLMs in generating structured responses, improving consistency and reducing prompt engineering in clients.

**Implementation**:

- Add prompts in [`server.py`](src/alliance_docs_mcp/server.py) using `@mcp.prompt()`
- Create templates for common documentation query patterns:
  - `documentation_search_template`: Guide for searching documentation effectively
  - `related_content_template`: Template for discovering related documentation
  - `category_exploration_template`: Template for exploring documentation by category
  - `technical_question_template`: Template for answering technical questions using docs

**Example Use Cases**:

- Clients can use prompts to structure queries like "How do I set up X?" or "What are the requirements for Y?"
- Prompts can incorporate system context about available tools and resources

### 2. Structured Output Schemas (High Value)

**Why**: Pydantic models ensure type-safe, validated responses that are predictable for clients and enable better IDE support.

**Implementation**:

- Define Pydantic models in a new `schemas.py` module or directly in [`server.py`](src/alliance_docs_mcp/server.py)
- Create models for tool responses:
  - `PageMetadata`: Standardized page metadata structure
  - `SearchResult`: Unified search result structure with score, highlights
  - `RelatedPageResult`: Related page result with similarity scores
  - `CategoryInfo`: Category information structure
- Apply schemas to tool return types using FastMCP's schema support

**Benefits**:

- Better validation and error messages
- Consistent response structures
- Improved documentation through schema definitions

### 3. Caching Middleware (Medium-High Value)

**Why**: Many documentation queries are repeated. Caching reduces server load and improves response times, especially for expensive operations like search and related page calculations.

**Implementation**:

- Import `cache_middleware` from `fastmcp.middleware`
- Apply to the FastMCP server instance in [`server.py`](src/alliance_docs_mcp/server.py)
- Configure TTL for different tool types:
  - Short TTL for `list_recent_updates` (e.g., 5 minutes)
  - Medium TTL for search results (e.g., 30 minutes)
  - Long TTL for static queries like `list_categories` (e.g., 1 hour)

**Considerations**:

- Cache invalidation strategy when docs are synced
- Memory usage monitoring
- Environment variable to disable caching if needed

### 4. User Elicitation for Complex Queries (Medium Value)

**Why**: Some queries benefit from clarification or additional context. Elicitation allows tools to ask users for missing parameters or preferences.

**Implementation**:

- Use FastMCP's elicitation features in tools like `search_docs`:
  - If query is too vague (e.g., < 3 characters), ask for clarification
  - If search returns many results, ask if user wants to filter by category
  - For `find_related_pages`, ask if user wants to adjust similarity threshold
- Implement in [`server.py`](src/alliance_docs_mcp/server.py) using `request_user_input()`

**Example Flow**:

```
User: "search for docker"
Tool: "I found 47 results. Would you like to filter by a specific category? Available categories: Getting Started, Technical Reference, User Guide..."
```

### 5. Progress Reporting (Medium Value)

**Why**: Some operations (like large searches or related page calculations) can take time. Progress reporting improves UX.

**Implementation**:

- Use FastMCP's progress context in tools that process multiple items
- Apply to operations in [`server.py`](src/alliance_docs_mcp/server.py):
  - `list_all_pages()` when returning large result sets
  - `search_docs()` when processing complex queries
  - Resource registration during startup (if exposed as tool)

### 6. Enhanced Tool: Batch Operations (Medium Value)

**Why**: Clients often need to fetch multiple pages or perform multiple searches. Batch operations reduce round trips.

**Implementation**:

- Add `get_multiple_pages(slugs: List[str])` tool
- Add `search_multiple_categories(queries: List[str], categories: List[str])` tool
- Implement in [`server.py`](src/alliance_docs_mcp/server.py)

### 7. Resource Tool Middleware (Low-Medium Value)

**Why**: Some clients cannot list or read resources directly. This middleware provides tools as fallback.

**Implementation**:

- Import `resource_tool_middleware` from `fastmcp.middleware`
- Apply to server in [`server.py`](src/alliance_docs_mcp/server.py)
- Automatically adds `list_resources` and `read_resource` tools

**Note**: Less critical since current resources are well-documented, but provides compatibility for clients that need it.

### 8. Server Metadata and Health Enhancements (Low Value)

**Why**: Better observability and debugging.

**Implementation**:

- Enhance `/health` endpoint in [`server.py`](src/alliance_docs_mcp/server.py) to include:
  - Index status (search index, related index)
  - Document count
  - Last sync timestamp
  - Index sizes
- Add server metadata endpoint with version, capabilities, etc.

## Priority Recommendations

**Phase 1 (High Impact, Low Effort)**:

1. Structured Output Schemas
2. Caching Middleware
3. Enhanced Health Endpoint

**Phase 2 (High Impact, Medium Effort)**:

4. MCP Prompts
5. Progress Reporting

**Phase 3 (Medium Impact, Variable Effort)**:

6. User Elicitation
7. Batch Operations
8. Resource Tool Middleware

## Implementation Considerations

- **Backward Compatibility**: All changes should maintain backward compatibility with existing tool signatures
- **Configuration**: Add environment variables for new features (e.g., `ENABLE_CACHING`, `CACHE_TTL`)
- **Testing**: Add tests for new features in [`tests/`](tests/)
- **Documentation**: Update [`README.md`](README.md) with new features and usage examples
- **Performance**: Monitor impact of middleware and new features on server performance