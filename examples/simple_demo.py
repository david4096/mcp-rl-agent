#!/usr/bin/env python3
"""Simple demonstration of the MCP RL Agent system."""

import asyncio
import sys
from pathlib import Path

# Add src to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_rl_agent.main import MCPRLAgentSystem


async def run_demo():
    """Run a simple demonstration."""
    print("=== MCP RL Agent Demo ===\n")

    # Use default configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    # Create and initialize system
    system = MCPRLAgentSystem(str(config_path))

    try:
        print("Initializing system...")
        await system.initialize()

        print("System initialized! Starting demo interaction...\n")

        # Reset environment
        obs = await system.environment.reset()
        print(f"Environment reset. Available actions: {len(obs.state.available_actions)}")

        # Demonstrate a few interactions
        for step in range(5):
            print(f"\n--- Step {step + 1} ---")

            # Agent selects action
            action = system.agent.select_action(obs.state)
            available_tools = obs.state.available_actions

            if action < len(available_tools):
                tool = available_tools[action]
                print(f"Agent selected action: {tool.name} ({tool.description})")
            else:
                print(f"Agent selected invalid action: {action}")

            # Take step
            obs = await system.environment.step(action)

            print(f"Reward: {obs.reward:.3f}")
            print(f"Done: {obs.done}")

            if obs.info:
                print(f"Info: {obs.info}")

            if obs.done:
                print("Episode completed!")
                break

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        print("\nShutting down...")
        await system.shutdown()
        print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(run_demo())