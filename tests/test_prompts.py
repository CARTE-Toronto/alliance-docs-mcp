"""Tests for MCP prompts."""

import pytest

from alliance_docs_mcp.server import mcp


@pytest.mark.asyncio
async def test_prompts_are_registered():
    """Test that all prompts are registered with the MCP server."""
    prompts = await mcp._list_prompts()
    prompt_names = {p.name for p in prompts}
    
    expected_prompts = {
        "documentation_search_guide",
        "technical_question_template",
        "category_exploration_guide",
        "related_content_discovery",
        "getting_started_helper",
    }
    
    assert expected_prompts.issubset(prompt_names), f"Missing prompts. Found: {prompt_names}"


@pytest.mark.asyncio
async def test_documentation_search_guide_metadata():
    """Test documentation_search_guide prompt metadata."""
    prompts = await mcp._list_prompts()
    prompt = next((p for p in prompts if p.name == "documentation_search_guide"), None)
    
    assert prompt is not None
    assert "search" in prompt.tags or "documentation" in prompt.tags
    assert prompt.description is not None
    assert len(prompt.description) > 0


@pytest.mark.asyncio
async def test_technical_question_template_metadata():
    """Test technical_question_template prompt metadata."""
    prompts = await mcp._list_prompts()
    prompt = next((p for p in prompts if p.name == "technical_question_template"), None)
    
    assert prompt is not None
    assert "technical" in prompt.tags or "question" in prompt.tags
    assert prompt.description is not None
    assert len(prompt.description) > 0


@pytest.mark.asyncio
async def test_category_exploration_guide_metadata():
    """Test category_exploration_guide prompt metadata."""
    prompts = await mcp._list_prompts()
    prompt = next((p for p in prompts if p.name == "category_exploration_guide"), None)
    
    assert prompt is not None
    assert "category" in prompt.tags or "exploration" in prompt.tags
    assert prompt.description is not None
    assert len(prompt.description) > 0


@pytest.mark.asyncio
async def test_related_content_discovery_metadata():
    """Test related_content_discovery prompt metadata."""
    prompts = await mcp._list_prompts()
    prompt = next((p for p in prompts if p.name == "related_content_discovery"), None)
    
    assert prompt is not None
    assert "related" in prompt.tags or "discovery" in prompt.tags
    assert prompt.description is not None
    assert len(prompt.description) > 0


@pytest.mark.asyncio
async def test_getting_started_helper_metadata():
    """Test getting_started_helper prompt metadata."""
    prompts = await mcp._list_prompts()
    prompt = next((p for p in prompts if p.name == "getting_started_helper"), None)
    
    assert prompt is not None
    assert "getting-started" in prompt.tags or "onboarding" in prompt.tags
    assert prompt.description is not None
    assert len(prompt.description) > 0


@pytest.mark.asyncio
async def test_all_prompts_have_metadata():
    """Test that all prompts have proper metadata."""
    prompts = await mcp._list_prompts()
    expected_prompts = {
        "documentation_search_guide",
        "technical_question_template",
        "category_exploration_guide",
        "related_content_discovery",
        "getting_started_helper",
    }
    
    for prompt_name in expected_prompts:
        prompt = next((p for p in prompts if p.name == prompt_name), None)
        assert prompt is not None, f"Prompt {prompt_name} not found"
        assert prompt.description is not None and len(prompt.description) > 0, f"Prompt {prompt_name} missing description"
        assert prompt.tags is not None and len(prompt.tags) > 0, f"Prompt {prompt_name} missing tags"
