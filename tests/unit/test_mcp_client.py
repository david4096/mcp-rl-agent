"""Tests for MCP client functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from mcp_rl_agent.mcp.mock_client import MockMCPClient, MockMCPClientFactory
from mcp_rl_agent.mcp.client import MCPClientManager
from mcp_rl_agent.interfaces import MCPTool, MCPAction


class TestMockMCPClient:
    """Test mock MCP client functionality."""

    @pytest.mark.asyncio
    async def test_connection(self):
        """Test mock client connection."""
        client = MockMCPClient("test_server")

        assert not client.is_connected
        assert client.server_id == "test_server"

        success = await client.connect({})
        assert success
        assert client.is_connected

        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """Test tool discovery."""
        client = MockMCPClient("test_server")
        await client.connect({})

        tools = await client.discover_tools()

        assert len(tools) > 0
        assert all(isinstance(tool, MCPTool) for tool in tools)
        assert all(tool.server_id == "test_server" for tool in tools)

        # Check specific tools exist
        tool_names = [tool.name for tool in tools]
        assert "echo" in tool_names
        assert "calculate" in tool_names

    @pytest.mark.asyncio
    async def test_action_execution(self):
        """Test action execution."""
        client = MockMCPClient("test_server")
        await client.connect({})

        tools = await client.discover_tools()
        echo_tool = next(tool for tool in tools if tool.name == "echo")

        action = MCPAction(
            tool=echo_tool,
            arguments={"message": "Hello, World!"},
            action_id="test_action_1"
        )

        result = await client.execute_action(action)

        assert result.action_id == "test_action_1"
        assert result.success
        assert result.result is not None

    @pytest.mark.asyncio
    async def test_factory_clients(self):
        """Test factory-created clients."""
        # Test basic tools client
        basic_client = MockMCPClientFactory.create_basic_tools_client()
        await basic_client.connect({})
        basic_tools = await basic_client.discover_tools()
        assert len(basic_tools) > 0

        # Test file tools client
        file_client = MockMCPClientFactory.create_file_tools_client()
        await file_client.connect({})
        file_tools = await file_client.discover_tools()
        file_tool_names = [tool.name for tool in file_tools]
        assert "read_file" in file_tool_names
        assert "write_file" in file_tool_names

        # Test unreliable client
        unreliable_client = MockMCPClientFactory.create_unreliable_client()
        await unreliable_client.connect({})
        unreliable_tools = await unreliable_client.discover_tools()
        assert len(unreliable_tools) > 0


class TestMCPClientManager:
    """Test MCP client manager functionality."""

    @pytest.mark.asyncio
    async def test_client_management(self):
        """Test adding and removing clients."""
        manager = MCPClientManager()

        # Initially no clients
        assert len(manager.get_all_tools()) == 0

        # Mock server config
        from mcp_rl_agent.config.models import MCPServerConfig, MCPTransport

        # Since we can't easily test real stdio connections in unit tests,
        # we'll mock the client creation
        manager._clients["test_server"] = MockMCPClient("test_server")
        await manager._clients["test_server"].connect({})
        tools = await manager._clients["test_server"].discover_tools()
        manager._tools_cache["test_server"] = tools

        # Verify tools are available
        all_tools = manager.get_all_tools()
        assert len(all_tools) > 0

        server_tools = manager.get_server_tools("test_server")
        assert len(server_tools) == len(all_tools)

        # Test action execution
        if all_tools:
            action = MCPAction(
                tool=all_tools[0],
                arguments={},
                action_id="test_action"
            )
            result = await manager.execute_action(action)
            assert result.action_id == "test_action"

        # Remove server
        await manager.remove_server("test_server")
        assert len(manager.get_all_tools()) == 0