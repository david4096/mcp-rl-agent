#!/usr/bin/env python3
"""Quick test showing the MCP RL Agent system components working."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_components():
    """Test individual system components."""
    print("🧪 Testing MCP RL Agent Components\n")

    # Test 1: Mock MCP Client
    print("1️⃣  Testing Mock MCP Client...")
    from mcp_rl_agent.mcp.mock_client import MockMCPClient

    client = MockMCPClient("test_server")
    await client.connect({})
    tools = await client.discover_tools()
    print(f"   ✅ Connected to mock server with {len(tools)} tools")
    print(f"   🔧 Available tools: {[tool.name for tool in tools]}")

    # Test 2: LLM Provider
    print("\n2️⃣  Testing Mock LLM Provider...")
    from mcp_rl_agent.llm.base import MockLLMProvider
    from mcp_rl_agent.interfaces import Message, MessageType

    llm = MockLLMProvider("test_model", responses=["Hello! I'll help you with that task."])
    messages = [Message(MessageType.HUMAN, "Can you help me?", 0.0)]
    response = await llm.generate_response(messages)
    print(f"   ✅ LLM Response: {response}")

    # Test 3: Message Parser
    print("\n3️⃣  Testing Natural Language Parser...")
    from mcp_rl_agent.interface.message_parser import MessageParser

    parser = MessageParser()
    intent = parser.parse_message("Can you echo hello world?", tools)
    print(f"   ✅ Parsed intent: {intent.intent.value}")
    print(f"   🎯 Tool suggestion: {intent.tool_suggestion}")

    # Test 4: PPO Agent
    print("\n4️⃣  Testing PPO Agent...")
    from mcp_rl_agent.rl.ppo_agent import PPOAgent
    from mcp_rl_agent.interfaces import AgentState, Conversation

    config = {
        "learning_rate": 0.001,
        "state_dim": 64,
        "max_actions": len(tools),
        "hidden_size": 32,
        "n_steps": 10,
        "batch_size": 4
    }

    agent = PPOAgent(config)

    # Create a test state
    conversation = Conversation(id="test", messages=[], active=True)
    state = AgentState(
        conversation=conversation,
        available_actions=tools,
        context_embedding=None,
        step=0
    )

    action = agent.select_action(state)
    selected_tool = tools[action] if action < len(tools) else None
    print(f"   ✅ Agent selected action {action}: {selected_tool.name if selected_tool else 'Invalid'}")

    # Test 5: End-to-End Action Execution
    print("\n5️⃣  Testing End-to-End Action...")
    from mcp_rl_agent.interfaces import MCPAction

    if selected_tool:
        action_obj = MCPAction(
            tool=selected_tool,
            arguments={"message": "Hello World!"},
            action_id="test_action"
        )

        result = await client.execute_action(action_obj)
        print(f"   ✅ Action executed: {result.success}")
        if result.result:
            print(f"   📄 Result: {result.result}")

    print("\n🎉 All components working correctly!")
    print("\n💡 To run the full system:")
    print("   uv run python -m mcp_rl_agent.main --mode interactive")
    print("   uv run python examples/simple_demo.py")

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_components())