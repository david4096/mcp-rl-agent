#!/usr/bin/env python3
"""
Example script for connecting to and training with MCP Gym Atari server.

This example demonstrates:
1. Connecting to an HTTP-based MCP server
2. Discovering available Atari games
3. Training with the enhanced PPO architecture
4. Evaluating performance on different games
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from mcp_rl_agent.config.loader import ConfigLoader
from mcp_rl_agent.mcp.client_manager import MCPClientManager
from mcp_rl_agent.llm.factory import LLMProviderFactory
from mcp_rl_agent.interface.operator import MockOperatorInterface
from mcp_rl_agent.env.mcp_env import MCPRLEnvironment
from mcp_rl_agent.rl.enhanced_ppo import EnhancedPPOAgent

logger = logging.getLogger(__name__)


class MCPGymTrainer:
    """Trainer specifically for MCP Gym Atari environments."""

    def __init__(self, config_path: str = "config/mcp_gym_atari_config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.mcp_manager: MCPClientManager = None
        self.environment: MCPRLEnvironment = None
        self.agent: EnhancedPPOAgent = None

    async def setup(self):
        """Initialize all components."""
        logger.info("Setting up MCP Gym trainer")

        # Load configuration
        config_loader = ConfigLoader()
        self.config = config_loader.load_config(str(self.config_path))

        # Setup logging
        self._setup_logging()

        # Initialize MCP client manager
        logger.info("Connecting to MCP Gym server at http://0.0.0.0:8000/mcp")
        self.mcp_manager = MCPClientManager(self.config["mcp_servers"])

        try:
            await self.mcp_manager.connect_all()
            logger.info("Successfully connected to MCP Gym server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            logger.error("Make sure the MCP Gym server is running on http://0.0.0.0:8000")
            raise

        # Discover available games
        await self._discover_games()

        # Initialize LLM provider
        llm_provider = LLMProviderFactory.create_provider(
            self.config["llm_provider"]["type"],
            self.config["llm_provider"]["config"]
        )

        # Initialize operator interface
        operator = MockOperatorInterface(self.config.get("operator", {}))

        # Initialize environment
        logger.info("Initializing RL environment with Atari support")
        self.environment = MCPRLEnvironment(
            mcp_client_manager=self.mcp_manager,
            llm_provider=llm_provider,
            operator_interface=operator,
            config=self.config["environment"]
        )

        # Initialize enhanced PPO agent
        logger.info("Initializing Enhanced PPO agent")
        self.agent = EnhancedPPOAgent(
            action_space_size=self.environment.action_space_size,
            config=self.config["rl"]
        )

        logger.info("Setup complete!")

    async def _discover_games(self):
        """Discover available Atari games from the MCP server."""
        logger.info("Discovering available games...")

        tools = self.mcp_manager.get_all_tools()
        game_tools = []

        for tool in tools:
            if any(keyword in tool.name.lower() for keyword in ['game', 'atari', 'gym', 'env']):
                game_tools.append(tool)

        logger.info(f"Found {len(game_tools)} game-related tools:")
        for tool in game_tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        if not game_tools:
            logger.warning("No game-related tools found. Available tools:")
            for tool in tools:
                logger.info(f"  - {tool.name}: {tool.description}")

    def _setup_logging(self):
        """Configure logging."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        logging.basicConfig(
            level=level,
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/mcp_gym_trainer.log")
            ]
        )

    async def test_connection(self):
        """Test the connection to the MCP server."""
        logger.info("Testing MCP server connection...")

        try:
            # Test basic connectivity
            status = self.mcp_manager.get_connection_status()
            logger.info(f"Connection status: {status}")

            # Test tool discovery
            tools = self.mcp_manager.get_all_tools()
            logger.info(f"Available tools: {len(tools)}")

            # Try a simple tool call if available
            if tools:
                test_tool = tools[0]
                logger.info(f"Testing tool: {test_tool.name}")

                # This would need to be implemented in your environment
                # result = await self.mcp_manager.execute_tool(test_tool, {})
                # logger.info(f"Tool test result: {result}")

            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def train(self, episodes: int = 1000):
        """Train the agent on Atari games."""
        logger.info(f"Starting training for {episodes} episodes")

        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_reward = 0
        episode_rewards = []

        for episode in range(episodes):
            logger.info(f"Episode {episode + 1}/{episodes}")

            try:
                # Reset environment
                obs = await self.environment.reset()
                episode_reward = 0
                step = 0

                while not obs.done and step < self.config["environment"]["max_episode_length"]:
                    # Select action
                    action = self.agent.select_action(obs.state)

                    # Take step
                    obs = await self.environment.step(action)
                    episode_reward += obs.reward
                    step += 1

                    if step % 100 == 0:
                        logger.debug(f"Step {step}, Reward: {obs.reward:.3f}")

                episode_rewards.append(episode_reward)
                total_reward += episode_reward

                # Log episode results
                avg_reward = total_reward / (episode + 1)
                logger.info(f"Episode {episode + 1} complete: Reward={episode_reward:.2f}, Avg={avg_reward:.2f}, Steps={step}")

                # Save checkpoint
                if (episode + 1) % self.config["training"]["save_interval"] == 0:
                    checkpoint_path = checkpoint_dir / f"episode_{episode + 1}.pt"
                    self.agent.save_checkpoint(str(checkpoint_path))
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Evaluation
                if (episode + 1) % self.config["training"]["eval_interval"] == 0:
                    eval_reward = await self.evaluate(num_episodes=5)
                    logger.info(f"Evaluation after episode {episode + 1}: {eval_reward:.2f}")

            except Exception as e:
                logger.error(f"Episode {episode + 1} failed: {e}")
                continue

        logger.info("Training complete!")
        return episode_rewards

    async def evaluate(self, num_episodes: int = 10):
        """Evaluate the trained agent."""
        logger.info(f"Evaluating for {num_episodes} episodes")

        total_reward = 0
        for episode in range(num_episodes):
            obs = await self.environment.reset()
            episode_reward = 0
            step = 0

            while not obs.done and step < self.config["environment"]["max_episode_length"]:
                # Use deterministic action selection for evaluation
                action = self.agent.select_action(obs.state)
                obs = await self.environment.step(action)
                episode_reward += obs.reward
                step += 1

            total_reward += episode_reward
            logger.debug(f"Eval episode {episode + 1}: {episode_reward:.2f}")

        avg_reward = total_reward / num_episodes
        logger.info(f"Evaluation complete: Average reward = {avg_reward:.2f}")
        return avg_reward

    async def cleanup(self):
        """Clean up resources."""
        if self.environment:
            self.environment.close()
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()
        logger.info("Cleanup complete")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train MCP RL Agent with MCP Gym Atari server")
    parser.add_argument("--config", default="config/mcp_gym_atari_config.yaml", help="Config file path")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--test-only", action="store_true", help="Only test connection")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (requires trained model)")

    args = parser.parse_args()

    trainer = MCPGymTrainer(args.config)

    try:
        await trainer.setup()

        if args.test_only:
            success = await trainer.test_connection()
            print(f"Connection test: {'PASSED' if success else 'FAILED'}")
        elif args.eval_only:
            await trainer.evaluate(20)
        else:
            await trainer.train(args.episodes)
            await trainer.evaluate(10)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    finally:
        await trainer.cleanup()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)