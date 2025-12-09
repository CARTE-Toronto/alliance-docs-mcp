# Alliance Docs MCP Server - Version 2 Proposal

## Executive Summary

This proposal outlines enhancements to transform the Alliance Docs MCP server from a basic metadata search system into a powerful, intelligent documentation search and discovery platform. The primary focus is on implementing full-text search capabilities while maintaining backward compatibility and adding smart features that leverage modern search technologies.

## Current Limitations (V1)

### Identified Issues

1. **Limited Search Capability**
   - Only searches page titles (substring matching)
   - Does not search actual markdown content
   - No relevance ranking
   - Cannot find information like "GPU usage on Nibi" when the phrase doesn't appear in titles

2. **No Content Indexing**
   - Full markdown content is stored but not indexed
   - Content search requires loading entire files (slow, inefficient)
   - No way to search across page bodies

3. **Basic Query Matching**
   - Simple case-insensitive substring search
   - No phrase matching, fuzzy search, or semantic understanding
   - No support for complex queries

4. **Limited Discovery**
   - No related pages suggestions
   - No content-based recommendations
   - No search result snippets/highlights

## Proposed Enhancements (V2)

### 1. Full-Text Search Engine Integration

#### 1.1 Whoosh-Based Search (Recommended for V2.0)

**Why Whoosh:**
- Pure Python, no external dependencies
- Lightweight and fast for this use case
- Easy to integrate and maintain
- Good enough for 400+ pages
- No additional infrastructure required

**Implementation:**
```python
# New module: src/alliance_docs_mcp/search_index.py
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, DATETIME
from whoosh.qparser import MultifieldParser

class SearchIndex:
    """Full-text search index using Whoosh."""
    
    def __init__(self, index_dir: Path):
        self.schema = Schema(
            slug=ID(stored=True, unique=True),
            title=TEXT(stored=True),
            content=TEXT(stored=True),
            category=KEYWORD(stored=True),
            url=ID(stored=True),
            last_modified=DATETIME(stored=True),
        )
        self.index_dir = index_dir
        self.index = self._create_or_open_index()
    
    def index_page(self, slug: str, title: str, content: str, 
                   category: str, url: str, last_modified: datetime):
        """Add or update a page in the search index."""
        writer = self.index.writer()
        writer.update_document(
            slug=slug,
            title=title,
            content=content,
            category=category,
            url=url,
            last_modified=last_modified,
        )
        writer.commit()
    
    def search(self, query: str, category: Optional[str] = None, 
               limit: int = 20) -> List[Dict]:
        """Search with relevance ranking."""
        with self.index.searcher() as searcher:
            # Multi-field search: title (boosted) + content
            parser = MultifieldParser(
                ["title", "content"],
                schema=self.index.schema,
                fieldboosts={"title": 3.0, "content": 1.0}
            )
            parsed_query = parser.parse(query)
            
            results = searcher.search(parsed_query, limit=limit)
            return [
                {
                    "title": hit["title"],
                    "slug": hit["slug"],
                    "url": hit["url"],
                    "category": hit["category"],
                    "score": hit.score,
                    "highlights": hit.highlights("content", top=3),
                }
                for hit in results
            ]
```

**Benefits:**
- Full-text search across all content
- Relevance scoring (title matches ranked higher)
- Search result highlighting
- Fast queries (< 100ms for typical searches)
- No external service dependencies

**Migration Path:**
- Build index during sync process
- Store index alongside markdown files
- Backward compatible: fallback to title search if index missing

#### 1.2 Alternative: SQLite FTS5 (Lighter Option)

For an even lighter solution, SQLite's built-in FTS5 could be used:
- Zero additional dependencies
- Excellent performance for this scale
- Built-in ranking and snippet generation
- Simpler implementation

### 2. Enhanced Search API

#### 2.1 Improved `search_docs` Tool

```python
@mcp.tool()
async def search_docs(
    query: str,
    category: Optional[str] = None,
    limit: int = 20,
    search_content: bool = True,  # NEW: enable full-text
    fuzzy: bool = False,  # NEW: fuzzy matching
) -> List[dict]:
    """Search documentation with full-text capabilities.
    
    Args:
        query: Search query (supports phrases, boolean operators)
        category: Optional category filter
        limit: Maximum results to return
        search_content: If True, search full content; if False, title-only
        fuzzy: Enable fuzzy matching for typos
        
    Returns:
        List of matching pages with relevance scores and highlights
    """
```

**New Features:**
- Full-text content search (toggleable)
- Relevance scoring
- Search result snippets with highlighted matches
- Support for phrase queries (quoted strings)
- Boolean operators (AND, OR, NOT)
- Fuzzy matching for typo tolerance

#### 2.2 New `semantic_search` Tool (Future Enhancement)

For V2.1, consider adding semantic search using embeddings:
- Use sentence transformers (e.g., `all-MiniLM-L6-v2`)
- Generate embeddings during sync
- Store in vector database (SQLite with vector extension or Chroma)
- Enable "find similar pages" functionality

### 3. Content Indexing During Sync

#### 3.1 Enhanced Sync Process

Modify `scripts/sync_docs.py` to:
1. Extract and index markdown content during sync
2. Build search index incrementally
3. Update index when pages change
4. Maintain index metadata (last indexed timestamp)

**Implementation:**
```python
# In sync_docs.py
from alliance_docs_mcp.search_index import SearchIndex

async def sync_with_indexing():
    storage = DocumentationStorage(docs_dir)
    search_index = SearchIndex(index_dir=docs_dir / "search_index")
    
    # ... existing sync logic ...
    
    # After saving each page:
    search_index.index_page(
        slug=page_data["slug"],
        title=page_data["title"],
        content=markdown_content,  # Full markdown text
        category=page_data["category"],
        url=page_data["url"],
        last_modified=page_data["last_modified"],
    )
```

### 4. Advanced Search Features

#### 4.1 Search Result Snippets

Return highlighted snippets showing where matches occurred:
```python
{
    "title": "Using GPUs with Slurm",
    "slug": "Using_GPUs_with_Slurm_en",
    "snippet": "...request one or more GPUs for a Slurm job, use this form: `--gpus-per-node=[type:]number`. The square-bracket notation means...",
    "highlights": ["GPUs", "Slurm", "job"],
    "score": 0.85
}
```

#### 4.2 Multi-Query Search

Support complex queries:
- `"GPU usage" AND Nibi` - Both terms must appear
- `GPU OR CUDA` - Either term
- `Nibi -cloud` - Nibi but not cloud
- `"exact phrase"` - Phrase matching

#### 4.3 Search Filters

Add filtering capabilities:
- By date range (recently updated)
- By content length
- By category (existing, but enhanced)
- By system/cluster name (extract from content)

### 5. New Tools and Capabilities

#### 5.1 `find_related_pages(slug: str, limit: int = 5)`

Find pages related to a given page (implemented with embeddings + Chroma in V2.1):
- Sentence-transformer embeddings (`all-MiniLM-L6-v2` by default)
- Persistent Chroma vector store built during sync (`--rebuild-related-index` flag available)
- Graceful fallback to lightweight heuristics when embeddings are unavailable

#### 5.2 `extract_code_examples(slug: str)`

Extract code blocks from a page:
- Return all code examples with language tags
- Useful for finding usage examples

#### 5.3 `search_by_system(system_name: str)`

Find all pages mentioning a specific system (Nibi, Narval, etc.):
- Extract system names from content
- Build system-to-pages mapping
- Fast lookup

#### 5.4 `get_page_sections(slug: str)`

Return structured sections of a page:
- Parse markdown headings
- Return table of contents
- Enable section-level navigation

### 6. Performance Optimizations

#### 6.1 Incremental Index Updates

- Only re-index changed pages
- Track page checksums/hashes
- Fast sync for daily updates

#### 6.2 Index Caching

- Keep index in memory for frequently accessed pages
- Lazy loading for large indices
- Background index optimization

#### 6.3 Query Optimization

- Cache common queries
- Pre-compute category filters
- Optimize index structure for common patterns

### 7. Enhanced Metadata Extraction

#### 7.1 Content Analysis

Extract additional metadata during sync:
- System names mentioned (Nibi, Narval, Trillium, etc.)
- Software packages referenced
- Code examples count
- External links
- Image count
- Section count

#### 7.2 Auto-Tagging

Automatically tag pages based on content:
- GPU-related pages
- Cluster-specific pages
- Tutorial vs reference
- Software-specific

### 8. Developer Experience Improvements

#### 8.1 Better Error Messages

- Clear errors when index is missing
- Suggestions for failed searches
- Query syntax help

#### 8.2 Search Analytics (Optional)

Track search patterns:
- Most common queries
- Failed searches
- Popular pages
- Help improve documentation

#### 8.3 Debug Mode

Enhanced logging for:
- Search query parsing
- Index operations
- Performance metrics

## Implementation Plan

### Phase 1: Core Full-Text Search (V2.0.0)

**Timeline: 2-3 weeks**

1. **Week 1: Index Infrastructure**
   - Add Whoosh dependency
   - Implement `SearchIndex` class
   - Create index schema
   - Basic indexing functionality

2. **Week 2: Integration**
   - Modify sync script to build index
   - Update `search_docs` tool
   - Add search result highlighting
   - Testing and validation

3. **Week 3: Polish**
   - Error handling
   - Backward compatibility
   - Documentation
   - Performance tuning

**Deliverables:**
- Full-text search working
- Index built during sync
- Enhanced search results with snippets
- Backward compatible with V1

### Phase 2: Advanced Features (V2.1.0)

**Timeline: 2-3 weeks**

1. Advanced query parsing
2. Related pages functionality
3. System-specific search
4. Code example extraction
5. Enhanced metadata extraction

### Phase 3: Semantic Search (V2.2.0) - Optional

**Timeline: 3-4 weeks**

1. Embedding generation
2. Vector database integration
3. Semantic similarity search
4. "Find similar" functionality

## Technical Considerations

### Dependencies

**New Dependencies:**
```toml
dependencies = [
    # ... existing ...
    "whoosh>=2.7.4",  # Full-text search
    "sentence-transformers>=2.2.0",  # Embeddings for related pages
    "chromadb>=0.6.3",  # Persistent vector store (Chroma)
    "numpy>=1.26.0",
]
```

### Storage Requirements

- Current: ~400 pages × ~50KB avg = ~20MB markdown
- With index: +5-10MB for Whoosh index
- Total: ~30MB (still very manageable)

### Performance Targets

- Index build time: < 5 minutes for full sync
- Search query time: < 100ms (p95)
- Index update time: < 1 second per page

### Backward Compatibility

- V1 tools continue to work
- If index missing, fallback to title search
- No breaking changes to existing API
- Gradual migration path

## Migration Strategy

### For Existing Deployments

1. **Automatic Migration:**
   - On first run after upgrade, detect missing index
   - Offer to build index (can be background process)
   - Continue serving requests with title search during build

2. **Manual Migration:**
   ```bash
   # Re-sync with indexing
   uv run python scripts/sync_docs.py --rebuild-index
   ```

3. **Rollback:**
   - Remove index directory to revert to V1 behavior
   - No data loss, fully reversible

## Testing Strategy

### Unit Tests
- Index creation and updates
- Search query parsing
- Result ranking
- Error handling

### Integration Tests
- Full sync with indexing
- Search across all pages
- Performance benchmarks
- Backward compatibility

### Example Test Cases

```python
def test_search_gpu_usage_nibi():
    """Test that 'GPU usage on Nibi' finds relevant pages."""
    results = await search_docs("GPU usage on Nibi")
    assert len(results) > 0
    assert any("Nibi" in r["title"] for r in results)
    assert any("GPU" in r["title"] or "GPU" in r.get("snippet", "") 
               for r in results)

def test_phrase_search():
    """Test phrase matching."""
    results = await search_docs('"GPU usage"')
    # Should find pages with exact phrase
```

## Success Metrics

### Quantitative
- Search success rate: > 90% for common queries
- Query response time: < 100ms (p95)
- Index build time: < 5 minutes
- Zero breaking changes

### Qualitative
- Users can find "GPU usage on Nibi" type queries
- Search results are relevant and ranked well
- Snippets provide useful context
- Overall improved user experience

## Future Enhancements (Post-V2)

1. **Multi-language Support**
   - Index French pages separately
   - Language-aware search

2. **Search Suggestions**
   - Autocomplete/typeahead
   - "Did you mean?" corrections

3. **Advanced Analytics**
   - Search trend analysis
   - Documentation gap identification

4. **API Enhancements**
   - GraphQL interface
   - REST API for non-MCP clients

5. **Real-time Updates**
   - WebSocket for live index updates
   - Push notifications for new content

## Risk Assessment

### Low Risk
- Whoosh integration (mature library)
- Backward compatibility (fallback exists)
- Performance (small dataset)

### Medium Risk
- Index corruption (mitigated by backups)
- Sync performance (mitigated by incremental updates)

### Mitigation Strategies
- Comprehensive testing
- Gradual rollout
- Monitoring and alerting
- Easy rollback mechanism

## Conclusion

Version 2 transforms the Alliance Docs MCP from a basic title search system into a powerful, full-featured documentation search platform. The proposed enhancements address the core limitation (no content search) while adding intelligent features that significantly improve discoverability.

The phased approach allows for incremental delivery, starting with the critical full-text search capability, then adding advanced features based on user feedback and needs.

**Recommended Next Steps:**
1. Review and approve proposal
2. Set up development branch
3. Begin Phase 1 implementation
4. Create detailed technical specifications for each component

---

**Document Version:** 1.0  
**Date:** 2025-01-XX  
**Author:** AI Assistant (proposal)  
**Status:** Draft for Review
