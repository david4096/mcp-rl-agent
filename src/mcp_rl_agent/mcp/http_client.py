"""HTTP MCP client implementation."""

import asyncio
import aiohttp
import json
import uuid
from typing import Dict, List, Any, Optional
import structlog

from ..interfaces import MCPClientInterface, MCPTool, MCPAction, MCPResult
from ..config.models import MCPServerConfig, MCPTransport

logger = structlog.get_logger(__name__)


class HTTPMCPClient(MCPClientInterface):
    """MCP client that communicates over HTTP with JSON-RPC 2.0."""

    def __init__(self, config: MCPServerConfig):
        """Initialize HTTP MCP client.

        Args:
            config: MCP server configuration with HTTP transport
        """
        if config.transport != MCPTransport.HTTP:
            raise ValueError(f"HTTPMCPClient requires HTTP transport, got {config.transport}")

        self.config = config
        self.server_url = config.url
        self.session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._request_id = 0
        self._session_id: Optional[str] = None

        # HTTP-specific settings
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "MCP-RL-Agent/1.0"
        }

        logger.info("Initialized HTTP MCP client", server_url=self.server_url)

    @property
    def server_id(self) -> str:
        """Get the server identifier."""
        return self.config.id

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected and self.session is not None

    def _next_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response to extract JSON data."""
        try:
            lines = sse_text.strip().split('\n')
            data_lines = [line[5:].strip() for line in lines if line.startswith('data:')]

            if data_lines:
                # Join all data lines and parse as JSON
                json_str = '\n'.join(data_lines)
                return json.loads(json_str)
            else:
                logger.warning("No data lines found in SSE response", sse_text=sse_text)
                return {}

        except json.JSONDecodeError as e:
            logger.error("Failed to parse SSE JSON data", error=str(e), sse_text=sse_text)
            return {"error": {"message": f"Failed to parse response: {str(e)}"}}
        except Exception as e:
            logger.error("Unexpected error parsing SSE response", error=str(e), sse_text=sse_text)
            return {"error": {"message": f"Unexpected error: {str(e)}"}}

    async def connect(self, server_config: Dict[str, Any]) -> bool:
        """Connect to the MCP server."""
        try:
            logger.info("Connecting to HTTP MCP server", url=self.server_url)

            # Create HTTP session (headers will be added per-request after getting session ID)
            self.session = aiohttp.ClientSession(
                timeout=self.timeout
            )

            # Test connection with a simple health check or initialization call
            await self._test_connection()

            self._connected = True
            logger.info("Successfully connected to HTTP MCP server")
            return True

        except Exception as e:
            logger.error("Failed to connect to HTTP MCP server", error=str(e))
            if self.session:
                await self.session.close()
                self.session = None
            self._connected = False
            return False

    async def _test_connection(self) -> None:
        """Test the connection, get session ID, and initialize MCP protocol."""
        try:
            # Step 1: Send initialize request to establish session and get session ID
            await self._initialize_mcp_protocol()

        except Exception as e:
            logger.warning("Connection test failed, but might still work", error=str(e))
            # Don't fail the connection for this - some servers might not respond to unknown methods

    async def _initialize_mcp_protocol(self) -> None:
        """Initialize MCP protocol with proper handshake sequence."""
        # Step 1: Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "MCP-RL-Agent",
                    "version": "1.0.0"
                }
            }
        }

        async with self.session.post(self.server_url, json=initialize_request, headers=self.headers) as response:
            logger.debug("Initialize response status", status=response.status)
            logger.debug("Initialize response headers", headers=dict(response.headers))

            # Get session ID from headers
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                self._session_id = session_id
                logger.info("Received session ID from server", session_id=session_id)
                self.headers['mcp-session-id'] = self._session_id

            # Handle Server-Sent Events format
            if response.content_type == 'text/event-stream':
                response_text = await response.text()
                response_data = self._parse_sse_response(response_text)
            else:
                response_data = await response.json()

            logger.debug("Initialize response data", data=response_data)

            # Check for errors
            if "error" in response_data:
                error = response_data["error"]
                logger.error("MCP initialization failed", error=error)
                raise RuntimeError(f"MCP initialization failed: {error.get('message', error)}")

        # Step 2: Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }

        async with self.session.post(self.server_url, json=initialized_notification, headers=self.headers) as response:
            logger.debug("Initialized notification response", status=response.status)
            # Notification should not return content, just acknowledge

        logger.info("MCP protocol initialization complete", session_id=self._session_id)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        logger.info("Disconnecting from HTTP MCP server")

        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

        self._connected = False
        logger.info("Disconnected from HTTP MCP server")

    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the MCP server."""
        if not self.is_connected:
            raise RuntimeError("Not connected to MCP server")

        try:
            logger.info("Discovering tools from HTTP MCP server")

            request = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/list",
                "params": {}
            }

            async with self.session.post(self.server_url, json=request, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"Server returned {response.status}: {error_text}")

                # Handle Server-Sent Events format
                if response.content_type == 'text/event-stream':
                    # Parse SSE response
                    response_text = await response.text()
                    response_data = self._parse_sse_response(response_text)
                else:
                    response_data = await response.json()

                # Check for JSON-RPC error
                if "error" in response_data:
                    error = response_data["error"]
                    raise RuntimeError(f"MCP server error: {error.get('message', error)}")

                if "result" not in response_data:
                    raise RuntimeError("Invalid JSON-RPC response: missing result")

                result = response_data["result"]
                tools_data = result.get("tools", [])

                # Convert to MCPTool objects
                tools = []
                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data.get("name", "unknown"),
                        description=tool_data.get("description", ""),
                        parameters=tool_data.get("inputSchema", {}),
                        server_id=self.server_id
                    )
                    tools.append(tool)

                logger.info(f"Discovered {len(tools)} tools", tools=[t.name for t in tools])
                return tools

        except Exception as e:
            logger.error("Failed to discover tools", error=str(e))
            raise

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute an action using the MCP server."""
        if not self.is_connected:
            raise RuntimeError("Not connected to MCP server")

        try:
            logger.debug(
                "Executing action via HTTP",
                tool=action.tool.name,
                arguments=action.arguments,
                action_id=action.action_id
            )

            request = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/call",
                "params": {
                    "name": action.tool.name,
                    "arguments": action.arguments
                }
            }

            async with self.session.post(self.server_url, json=request, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return MCPResult(
                        action_id=action.action_id,
                        success=False,
                        result=None,
                        error=f"HTTP {response.status}: {error_text}"
                    )

                # Handle Server-Sent Events format
                if response.content_type == 'text/event-stream':
                    # Parse SSE response
                    response_text = await response.text()
                    response_data = self._parse_sse_response(response_text)
                else:
                    response_data = await response.json()

                # Check for JSON-RPC error
                if "error" in response_data:
                    error = response_data["error"]
                    return MCPResult(
                        action_id=action.action_id,
                        success=False,
                        result=None,
                        error=f"MCP error: {error.get('message', str(error))}"
                    )

                if "result" not in response_data:
                    return MCPResult(
                        action_id=action.action_id,
                        success=False,
                        result=None,
                        error="Invalid JSON-RPC response: missing result"
                    )

                result = response_data["result"]

                logger.debug("Action executed successfully", result=result)

                return MCPResult(
                    action_id=action.action_id,
                    success=True,
                    result=result,
                    error=None
                )

        except Exception as e:
            logger.error("Failed to execute action", error=str(e))
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error=str(e)
            )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class HTTPMCPClientFactory:
    """Factory for creating HTTP MCP clients."""

    @staticmethod
    def create_client(config: MCPServerConfig) -> HTTPMCPClient:
        """Create an HTTP MCP client from configuration."""
        return HTTPMCPClient(config)

    @staticmethod
    def supports_transport(transport: MCPTransport) -> bool:
        """Check if this factory supports the given transport."""
        return transport == MCPTransport.HTTP