"""Mock MCP client for testing without actual MCP servers."""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
import structlog

from ..interfaces import MCPClientInterface, MCPTool, MCPAction, MCPResult
from .mock_server import MockMCPServer

logger = structlog.get_logger(__name__)


class MockMCPClient(MCPClientInterface):
    """Mock MCP client that simulates server interactions without actual network calls."""

    def __init__(self, server_id: str, tools: Optional[List[Dict[str, Any]]] = None):
        self._server_id = server_id
        self._connected = False
        self._mock_server = MockMCPServer()
        self._tools: List[MCPTool] = []

        # Override default tools if provided
        if tools:
            from .mock_server import MockTool
            mock_tools = []
            for tool_data in tools:
                mock_tools.append(MockTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("input_schema", {}),
                    response_template=tool_data.get("response_template", {"content": [{"type": "text", "text": "Mock response"}]}),
                    delay=tool_data.get("delay", 0.0),
                    success_rate=tool_data.get("success_rate", 1.0)
                ))
            self._mock_server.tools = mock_tools

    @property
    def server_id(self) -> str:
        return self._server_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, server_config: Dict[str, Any]) -> bool:
        """Mock connection - always succeeds."""
        logger.info("Connecting to mock MCP server", server_id=self._server_id)

        # Simulate connection delay
        await asyncio.sleep(0.1)

        self._connected = True
        logger.info("Successfully connected to mock MCP server", server_id=self._server_id)
        return True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        logger.info("Disconnecting from mock MCP server", server_id=self._server_id)
        self._connected = False

    async def discover_tools(self) -> List[MCPTool]:
        """Discover tools from mock server."""
        if not self.is_connected:
            raise ConnectionError("Not connected to mock server")

        logger.info("Discovering tools from mock server", server_id=self._server_id)

        # Get tools from mock server
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }

        response = await self._mock_server.handle_request(request)

        tools = []
        for tool_data in response.get("result", {}).get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                parameters=tool_data.get("inputSchema", {}),
                server_id=self._server_id
            )
            tools.append(tool)

        self._tools = tools
        logger.info("Discovered tools from mock server", server_id=self._server_id, tool_count=len(tools))
        return tools

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute action on mock server."""
        if not self.is_connected:
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error="Not connected to mock server"
            )

        logger.debug("Executing action on mock server", action_id=action.action_id, tool=action.tool.name)

        try:
            # Create request for mock server
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": action.tool.name,
                    "arguments": action.arguments
                }
            }

            # Execute on mock server
            response = await self._mock_server.handle_request(request)

            if "error" in response:
                return MCPResult(
                    action_id=action.action_id,
                    success=False,
                    result=None,
                    error=str(response["error"])
                )

            result_data = response.get("result", {})
            return MCPResult(
                action_id=action.action_id,
                success=True,
                result=result_data.get("content", []),
                metadata={"mock_response": response}
            )

        except Exception as e:
            logger.error("Error executing action on mock server", action_id=action.action_id, error=str(e))
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error=str(e)
            )


class MockMCPClientFactory:
    """Factory for creating mock MCP clients with predefined configurations."""

    @staticmethod
    def create_basic_tools_client(server_id: str = "mock_basic") -> MockMCPClient:
        """Create a mock client with basic tools."""
        return MockMCPClient(server_id)

    @staticmethod
    def create_file_tools_client(server_id: str = "mock_files") -> MockMCPClient:
        """Create a mock client focused on file operations."""
        tools = [
            {
                "name": "read_file",
                "description": "Read content from a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"}
                    },
                    "required": ["filename"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["filename", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List files in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }
        ]
        return MockMCPClient(server_id, tools)

    @staticmethod
    def create_web_tools_client(server_id: str = "mock_web") -> MockMCPClient:
        """Create a mock client focused on web operations."""
        tools = [
            {
                "name": "fetch_url",
                "description": "Fetch content from a URL",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]
        return MockMCPClient(server_id, tools)

    @staticmethod
    def create_unreliable_client(server_id: str = "mock_unreliable") -> MockMCPClient:
        """Create a mock client that simulates unreliable connections and failures."""
        tools = [
            {
                "name": "flaky_tool",
                "description": "A tool that sometimes fails",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                },
                "success_rate": 0.7,  # Fails 30% of the time
                "delay": 2.0  # Takes 2 seconds to execute
            }
        ]
        return MockMCPClient(server_id, tools)