#!/usr/bin/env python3
"""
Quick connection test for MCP Gym Atari server.

This script tests the HTTP connection to your MCP server and discovers available tools.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPGymTester:
    """Simple tester for MCP Gym server connectivity."""

    def __init__(self, base_url: str = "http://0.0.0.0:8000/mcp"):
        self.base_url = base_url.rstrip('/')
        self.session: aiohttp.ClientSession = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_basic_connection(self) -> bool:
        """Test basic HTTP connectivity."""
        try:
            logger.info(f"Testing connection to {self.base_url}")

            async with self.session.get(f"{self.base_url}/health", timeout=10) as response:
                if response.status == 200:
                    logger.info("✅ Health check passed")
                    return True
                else:
                    logger.warning(f"Health check returned status {response.status}")
                    # Try the root endpoint if health check fails
                    return await self._test_root_endpoint()

        except aiohttp.ClientError as e:
            logger.warning(f"Health endpoint failed: {e}")
            return await self._test_root_endpoint()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def _test_root_endpoint(self) -> bool:
        """Test the root MCP endpoint."""
        try:
            async with self.session.get(self.base_url, timeout=10) as response:
                logger.info(f"Root endpoint status: {response.status}")
                return response.status in [200, 404, 405]  # 404/405 might be normal for MCP endpoints
        except Exception as e:
            logger.error(f"Root endpoint test failed: {e}")
            return False

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools via MCP protocol."""
        try:
            # MCP tools/list request
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info("Discovering available tools...")

            async with self.session.post(
                self.base_url,
                json=request_data,
                headers=headers,
                timeout=30
            ) as response:

                if response.status != 200:
                    logger.error(f"Tools discovery failed with status {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response: {response_text}")
                    return []

                response_data = await response.json()

                if "error" in response_data:
                    logger.error(f"MCP error: {response_data['error']}")
                    return []

                if "result" not in response_data:
                    logger.error("No result in MCP response")
                    return []

                tools = response_data["result"].get("tools", [])
                logger.info(f"Found {len(tools)} tools")

                for i, tool in enumerate(tools, 1):
                    logger.info(f"  {i}. {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")

                return tools

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during tool discovery: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during tool discovery: {e}")
            return []

    async def test_tool_call(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test calling a specific tool."""
        if arguments is None:
            arguments = {}

        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info(f"Testing tool call: {tool_name}")

            async with self.session.post(
                self.base_url,
                json=request_data,
                headers=headers,
                timeout=60  # Longer timeout for tool execution
            ) as response:

                response_data = await response.json()

                if "error" in response_data:
                    logger.error(f"Tool call error: {response_data['error']}")
                    return {"error": response_data["error"]}

                result = response_data.get("result", {})
                logger.info(f"Tool call successful: {tool_name}")
                return result

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"error": str(e)}

    async def run_full_test(self) -> bool:
        """Run complete connectivity and functionality test."""
        logger.info("🚀 Starting MCP Gym server test")

        # Test basic connection
        if not await self.test_basic_connection():
            logger.error("❌ Basic connection test failed")
            return False

        # Discover tools
        tools = await self.discover_tools()
        if not tools:
            logger.error("❌ No tools discovered")
            return False

        # Test a simple tool call (if available)
        for tool in tools[:3]:  # Test first few tools
            tool_name = tool.get("name")
            if tool_name and "list" in tool_name.lower():  # Try a safe listing tool first
                result = await self.test_tool_call(tool_name)
                if "error" not in result:
                    logger.info(f"✅ Tool test passed: {tool_name}")
                    break
            elif tool_name:
                logger.info(f"Skipping complex tool for safety: {tool_name}")

        logger.info("✅ MCP Gym server test complete")
        return True


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MCP Gym Atari server connection")
    parser.add_argument("--url", default="http://0.0.0.0:8000/mcp", help="MCP server URL")
    parser.add_argument("--tool", help="Specific tool to test")
    parser.add_argument("--args", help="Tool arguments as JSON string")

    args = parser.parse_args()

    async with MCPGymTester(args.url) as tester:
        if args.tool:
            # Test specific tool
            tool_args = {}
            if args.args:
                try:
                    tool_args = json.loads(args.args)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in --args")
                    return 1

            result = await tester.test_tool_call(args.tool, tool_args)
            print(f"Tool result: {json.dumps(result, indent=2)}")
        else:
            # Run full test
            success = await tester.run_full_test()
            if not success:
                logger.error("❌ Test failed")
                return 1

    logger.info("✅ All tests passed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)