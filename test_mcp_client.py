#!/usr/bin/env python3
"""Test script to demonstrate MCP client usage."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Test the MCP server functionality."""
    print("🧪 Testing Alliance Documentation MCP Server")
    print("=" * 50)
    
    # Server parameters
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "alliance_docs_mcp.server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            print("✅ Connected to MCP server")
            
            # List available resources
            print("\n📚 Available Resources:")
            resources = await session.list_resources()
            for resource in resources:
                print(f"  - {resource.uri}")
            
            # List available tools
            print("\n🔧 Available Tools:")
            tools = await session.list_tools()
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test search functionality
            print("\n🔍 Testing search functionality:")
            try:
                search_result = await session.call_tool(
                    "search_docs",
                    arguments={"query": "getting started", "category": None}
                )
                print(f"Search results: {len(search_result.content)} items")
                if search_result.content:
                    print(f"First result: {search_result.content[0]}")
            except Exception as e:
                print(f"Search error: {e}")
            
            # Test categories
            print("\n📂 Testing categories:")
            try:
                categories = await session.call_tool("list_categories", arguments={})
                print(f"Categories: {categories.content}")
            except Exception as e:
                print(f"Categories error: {e}")
            
            print("\n✅ MCP server test completed!")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
