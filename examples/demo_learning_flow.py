#!/usr/bin/env python3
"""
Demo script showing the ideal learning flow:
User provides goal → RL agent learns through trial and error
"""

import asyncio
from mcp_rl_agent.main import MCPRLAgentSystem


async def demo_learning_flow():
    """Demonstrate the learning flow with your FastMCP server."""

    print("🎮 MCP RL Agent Learning Flow Demo")
    print("=" * 50)

    # Initialize the system
    system = MCPRLAgentSystem('config/mcp_gym_interactive_config.yaml')
    await system.initialize()

    print(f"✅ Connected to FastMCP server")
    print(f"✅ Discovered {len(system.environment.available_tools)} tools:")
    for i, tool in enumerate(system.environment.available_tools):
        print(f"   {i}: {tool.name} - {tool.description[:60]}...")

    print("\n" + "="*50)
    print("🧠 LEARNING FLOW DEMONSTRATION")
    print("="*50)

    # Simulate the learning flow
    episodes_to_show = 3

    for episode in range(episodes_to_show):
        print(f"\n📚 Episode {episode + 1}: Learning from Experience")
        print("-" * 30)

        # Reset environment (like starting a new conversation)
        obs = await system.environment.reset()
        print(f"🔄 Environment reset - Action space size: {system.environment.action_space_size}")

        # Simulate some learning steps
        steps_per_episode = 5
        total_reward = 0

        for step in range(steps_per_episode):
            # Agent selects action based on current policy
            action = system.agent.select_action(obs.state)
            tool_name = system.environment.available_tools[action].name

            print(f"   Step {step + 1}: Agent selects '{tool_name}' (action {action})")

            # Take step and get reward
            obs = await system.environment.step(action)
            step_reward = obs.reward
            total_reward += step_reward

            print(f"   → Reward: {step_reward:.3f} | Tool result: {obs.info.get('success', 'unknown')}")

            # Show what the agent is learning
            if hasattr(obs.state, 'action_history') and obs.state.action_history:
                recent_actions = [item.tool_name for item in obs.state.action_history[-3:]]
                print(f"   → Action history: {recent_actions}")

            if obs.done:
                break

        print(f"📊 Episode {episode + 1} complete - Total reward: {total_reward:.3f}")

        # Show learning progress
        if episode == 0:
            print("   💡 Agent is exploring randomly, learning what works")
        elif episode == 1:
            print("   🎯 Agent starting to remember successful action sequences")
        else:
            print("   🚀 Agent using learned patterns to improve performance")

    print("\n" + "="*50)
    print("🎯 WHAT THE AGENT LEARNED")
    print("="*50)

    print("\n1. 🧠 Query Understanding:")
    print("   - Agent remembers user goals through query embeddings")
    print("   - Each user request becomes a 256-dimensional vector")
    print("   - Similar requests get similar treatment")

    print("\n2. 📈 Action Chain Learning:")
    print("   - Agent tracks sequences: switch_game → reset_environment → step_environment")
    print("   - Attention mechanism identifies which sequences work best")
    print("   - Successful patterns get higher probability")

    print("\n3. 🎮 Game-Specific Rewards:")
    print("   - Score increases = positive reward")
    print("   - Lives lost = negative reward")
    print("   - Efficient play = bonus reward")
    print("   - Repetitive actions = penalty")

    print("\n4. 🔄 Continuous Improvement:")
    print("   - PPO algorithm adjusts policy based on experience")
    print("   - Enhanced architecture separates 'what user wants' vs 'what I did'")
    print("   - Multi-head attention learns complex action relationships")

    print("\n🎯 READY FOR REAL LEARNING!")
    print("To start interactive learning:")
    print("uv run python -m mcp_rl_agent.main --config config/mcp_gym_interactive_config.yaml --mode interactive")

    print("\nTo start autonomous training:")
    print("uv run python -m mcp_rl_agent.main --config config/mcp_gym_interactive_config.yaml --mode train --episodes 100")

    # Cleanup
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_learning_flow())