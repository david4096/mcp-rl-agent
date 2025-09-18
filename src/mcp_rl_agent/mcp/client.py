"""MCP client implementation for connecting to MCP servers."""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import structlog
import websockets
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..interfaces import MCPClientInterface, MCPTool, MCPAction, MCPResult
from ..config.models import MCPServerConfig, MCPTransport
from .http_client import HTTPMCPClient

logger = structlog.get_logger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class ConnectionError(MCPError):
    """Error connecting to MCP server."""
    pass


class ProtocolError(MCPError):
    """Error in MCP protocol communication."""
    pass


class StdioMCPClient(MCPClientInterface):
    """MCP client using stdio transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._connected = False
        self._request_id = 0

    @property
    def server_id(self) -> str:
        return self.config.id

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def connect(self, server_config: Dict[str, Any]) -> bool:
        """Connect to MCP server via stdio."""
        try:
            logger.info("Connecting to MCP server", server_id=self.server_id)

            command = self.config.command + (self.config.args or [])
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Initialize connection with MCP handshake
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {
                        "roots": {"listRoots": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "mcp-rl-agent",
                        "version": "0.1.0"
                    }
                }
            }

            await self._send_request(init_request)
            response = await self._receive_response()

            if "error" in response:
                raise ConnectionError(f"Server initialization failed: {response['error']}")

            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send_request(initialized_notification)

            self._connected = True
            logger.info("Successfully connected to MCP server", server_id=self.server_id)
            return True

        except Exception as e:
            logger.error("Failed to connect to MCP server", server_id=self.server_id, error=str(e))
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to {self.server_id}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        logger.info("Disconnecting from MCP server", server_id=self.server_id)

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning("Error during disconnect", server_id=self.server_id, error=str(e))

        self._process = None
        self._connected = False

    async def discover_tools(self) -> List[MCPTool]:
        """Discover available tools from the MCP server."""
        if not self.is_connected:
            raise ConnectionError("Not connected to server")

        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "tools/list"
            }

            await self._send_request(request)
            response = await self._receive_response()

            if "error" in response:
                raise ProtocolError(f"Tools discovery failed: {response['error']}")

            tools = []
            for tool_data in response.get("result", {}).get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    parameters=tool_data.get("inputSchema", {}),
                    server_id=self.server_id
                )
                tools.append(tool)

            logger.info("Discovered tools", server_id=self.server_id, tool_count=len(tools))
            return tools

        except Exception as e:
            logger.error("Failed to discover tools", server_id=self.server_id, error=str(e))
            raise ProtocolError(f"Tool discovery failed: {e}")

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute an action using the MCP server."""
        if not self.is_connected:
            raise ConnectionError("Not connected to server")

        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "tools/call",
                "params": {
                    "name": action.tool.name,
                    "arguments": action.arguments
                }
            }

            await self._send_request(request)
            response = await self._receive_response()

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
                metadata={"response": response}
            )

        except Exception as e:
            logger.error("Failed to execute action", action_id=action.action_id, error=str(e))
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error=str(e)
            )

    async def _send_request(self, request: Dict[str, Any]) -> None:
        """Send a request to the MCP server."""
        if not self._process or not self._process.stdin:
            raise ConnectionError("No active connection")

        request_json = json.dumps(request) + "\n"
        self._process.stdin.write(request_json.encode())
        await self._process.stdin.drain()

    async def _receive_response(self) -> Dict[str, Any]:
        """Receive a response from the MCP server."""
        if not self._process or not self._process.stdout:
            raise ConnectionError("No active connection")

        line = await self._process.stdout.readline()
        if not line:
            raise ConnectionError("Connection closed by server")

        try:
            return json.loads(line.decode().strip())
        except json.JSONDecodeError as e:
            raise ProtocolError(f"Invalid JSON response: {e}")

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id


class WebSocketMCPClient(MCPClientInterface):
    """MCP client using WebSocket transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._request_id = 0

    @property
    def server_id(self) -> str:
        return self.config.id

    @property
    def is_connected(self) -> bool:
        return self._connected and self._websocket is not None

    async def connect(self, server_config: Dict[str, Any]) -> bool:
        """Connect to MCP server via WebSocket."""
        try:
            logger.info("Connecting to WebSocket MCP server", server_id=self.server_id)

            self._websocket = await websockets.connect(
                self.config.url,
                timeout=self.config.timeout
            )

            # Perform MCP handshake
            await self._perform_handshake()

            self._connected = True
            logger.info("Successfully connected to WebSocket MCP server", server_id=self.server_id)
            return True

        except Exception as e:
            logger.error("Failed to connect to WebSocket MCP server", server_id=self.server_id, error=str(e))
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to {self.server_id}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket MCP server."""
        logger.info("Disconnecting from WebSocket MCP server", server_id=self.server_id)

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error during WebSocket disconnect", server_id=self.server_id, error=str(e))

        self._websocket = None
        self._connected = False

    async def discover_tools(self) -> List[MCPTool]:
        """Discover tools via WebSocket."""
        # Implementation similar to StdioMCPClient but using WebSocket
        # Placeholder for now
        return []

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute action via WebSocket."""
        # Implementation similar to StdioMCPClient but using WebSocket
        # Placeholder for now
        return MCPResult(
            action_id=action.action_id,
            success=False,
            result=None,
            error="WebSocket implementation not complete"
        )

    async def _perform_handshake(self) -> None:
        """Perform MCP handshake over WebSocket."""
        # Implementation placeholder
        pass


class MCPClientManager:
    """Manages multiple MCP client connections."""

    def __init__(self):
        self._clients: Dict[str, MCPClientInterface] = {}
        self._tools_cache: Dict[str, List[MCPTool]] = {}

    async def add_server(self, config: MCPServerConfig) -> None:
        """Add and connect to a new MCP server."""
        if config.id in self._clients:
            logger.warning("Server already exists", server_id=config.id)
            return

        # Create appropriate client based on transport
        if config.transport == MCPTransport.STDIO:
            client = StdioMCPClient(config)
        elif config.transport == MCPTransport.WEBSOCKET:
            client = WebSocketMCPClient(config)
        elif config.transport == MCPTransport.HTTP:
            client = HTTPMCPClient(config)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

        # Connect and discover tools
        await client.connect({})
        tools = await client.discover_tools()

        self._clients[config.id] = client
        self._tools_cache[config.id] = tools

        logger.info("Added MCP server", server_id=config.id, tool_count=len(tools))

    async def remove_server(self, server_id: str) -> None:
        """Remove and disconnect from an MCP server."""
        if server_id not in self._clients:
            logger.warning("Server not found", server_id=server_id)
            return

        client = self._clients[server_id]
        await client.disconnect()

        del self._clients[server_id]
        del self._tools_cache[server_id]

        logger.info("Removed MCP server", server_id=server_id)

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from all connected servers."""
        all_tools = []
        for tools in self._tools_cache.values():
            all_tools.extend(tools)
        return all_tools

    def get_server_tools(self, server_id: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        return self._tools_cache.get(server_id, [])

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute an action on the appropriate server."""
        server_id = action.tool.server_id
        if server_id not in self._clients:
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error=f"Server {server_id} not available"
            )

        client = self._clients[server_id]
        return await client.execute_action(action)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for server_id in list(self._clients.keys()):
            await self.remove_server(server_id)