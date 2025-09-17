"""Mock MCP server implementations for development and testing."""

import asyncio
import json
import sys
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MockTool:
    """Mock tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    response_template: Dict[str, Any]
    delay: float = 0.0
    success_rate: float = 1.0


class MockMCPServer:
    """Mock MCP server for testing and development."""

    def __init__(self, tools: Optional[List[MockTool]] = None):
        self.tools = tools or self._create_default_tools()
        self.request_id = 0

    def _create_default_tools(self) -> List[MockTool]:
        """Create default mock tools."""
        return [
            MockTool(
                name="echo",
                description="Echo back the provided message",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back"
                        }
                    },
                    "required": ["message"]
                },
                response_template={
                    "content": [
                        {
                            "type": "text",
                            "text": "Echo: {message}"
                        }
                    ]
                }
            ),
            MockTool(
                name="calculate",
                description="Perform basic arithmetic calculations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                },
                response_template={
                    "content": [
                        {
                            "type": "text",
                            "text": "Result: {result}"
                        }
                    ]
                }
            ),
            MockTool(
                name="get_weather",
                description="Get weather information for a location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get weather for"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                },
                response_template={
                    "content": [
                        {
                            "type": "text",
                            "text": "Weather in {location}: 22°C, Partly cloudy"
                        }
                    ]
                }
            ),
            MockTool(
                name="write_file",
                description="Write content to a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["filename", "content"]
                },
                response_template={
                    "content": [
                        {
                            "type": "text",
                            "text": "Successfully wrote {byte_count} bytes to {filename}"
                        }
                    ]
                }
            ),
            MockTool(
                name="search_web",
                description="Search the web for information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results"
                        }
                    },
                    "required": ["query"]
                },
                response_template={
                    "content": [
                        {
                            "type": "text",
                            "text": "Found {result_count} results for '{query}'"
                        }
                    ]
                }
            )
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})

        logger.debug("Received request", method=method, request_id=request_id)

        if method == "initialize":
            return await self._handle_initialize(request_id, params)
        elif method == "tools/list":
            return await self._handle_tools_list(request_id)
        elif method == "tools/call":
            return await self._handle_tool_call(request_id, params)
        elif method == "notifications/initialized":
            # No response needed for notifications
            return None
        else:
            return self._create_error_response(request_id, -32601, f"Method not found: {method}")

    async def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "tools": {"listTools": True},
                    "resources": {"listResources": False},
                    "prompts": {"listPrompts": False}
                },
                "serverInfo": {
                    "name": "mock-mcp-server",
                    "version": "0.1.0"
                }
            }
        }

    async def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools list request."""
        tools_data = []
        for tool in self.tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools_data
            }
        }

    async def _handle_tool_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            return self._create_error_response(request_id, -32602, f"Tool not found: {tool_name}")

        # Simulate delay
        if tool.delay > 0:
            await asyncio.sleep(tool.delay)

        # Simulate success/failure based on success rate
        import random
        if random.random() > tool.success_rate:
            return self._create_error_response(
                request_id,
                -32603,
                f"Tool execution failed (simulated failure)"
            )

        # Execute mock tool
        try:
            result = await self._execute_mock_tool(tool, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return self._create_error_response(request_id, -32603, str(e))

    async def _execute_mock_tool(self, tool: MockTool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mock tool with the given arguments."""
        if tool.name == "echo":
            message = arguments.get("message", "")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Echo: {message}"
                    }
                ]
            }

        elif tool.name == "calculate":
            expression = arguments.get("expression", "")
            try:
                # Simple arithmetic evaluation (unsafe in real use!)
                result = eval(expression)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Result: {result}"
                        }
                    ]
                }
            except Exception:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: Invalid expression '{expression}'"
                        }
                    ]
                }

        elif tool.name == "get_weather":
            location = arguments.get("location", "Unknown")
            units = arguments.get("units", "celsius")
            temp_symbol = "°C" if units == "celsius" else "°F"
            temp = 22 if units == "celsius" else 72
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Weather in {location}: {temp}{temp_symbol}, Partly cloudy"
                    }
                ]
            }

        elif tool.name == "write_file":
            filename = arguments.get("filename", "untitled.txt")
            content = arguments.get("content", "")
            byte_count = len(content.encode('utf-8'))
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Successfully wrote {byte_count} bytes to {filename}"
                    }
                ]
            }

        elif tool.name == "search_web":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            import random
            result_count = random.randint(1, max_results)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Found {result_count} results for '{query}'"
                    }
                ]
            }

        else:
            # Generic response using template
            response = tool.response_template.copy()
            # Simple template substitution
            for content_item in response.get("content", []):
                if "text" in content_item:
                    text = content_item["text"]
                    for key, value in arguments.items():
                        text = text.replace(f"{{{key}}}", str(value))
                    content_item["text"] = text
            return response

    def _create_error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


async def run_stdio_server():
    """Run the mock MCP server with stdio transport."""
    server = MockMCPServer()
    logger.info("Starting mock MCP server with stdio transport")

    try:
        while True:
            # Read request from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = await server.handle_request(request)

                if response is not None:
                    response_json = json.dumps(response)
                    print(response_json, flush=True)

            except json.JSONDecodeError as e:
                logger.error("Invalid JSON request", error=str(e))
                error_response = server._create_error_response(None, -32700, "Parse error")
                print(json.dumps(error_response), flush=True)

            except Exception as e:
                logger.error("Error handling request", error=str(e))
                error_response = server._create_error_response(None, -32603, "Internal error")
                print(json.dumps(error_response), flush=True)

    except KeyboardInterrupt:
        logger.info("Mock MCP server stopped")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stdio":
        asyncio.run(run_stdio_server())
    else:
        print("Usage: python mock_server.py stdio")
        sys.exit(1)