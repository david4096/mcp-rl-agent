#!/usr/bin/env python3
"""
Example script demonstrating how to train the MCP RL Agent with Atari games.

This example shows:
1. How to load game-specific configurations
2. How to set up the environment with Atari reward functions
3. How to train the enhanced PPO agent
4. How to evaluate performance across different games
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from mcp_rl_agent.config.loader import ConfigLoader
from mcp_rl_agent.mcp.client_manager import MCPClientManager
from mcp_rl_agent.llm.factory import LLMProviderFactory
from mcp_rl_agent.interface.operator import MockOperatorInterface
from mcp_rl_agent.env.mcp_env import MCPRLEnvironment
from mcp_rl_agent.rl.enhanced_ppo import EnhancedPPOAgent
from mcp_rl_agent.training.trainer import MCPRLTrainer

logger = logging.getLogger(__name__)


class AtariTrainingSession:
    """Manages training sessions for Atari games using MCP RL Agent."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.environment: MCPRLEnvironment = None
        self.agent: EnhancedPPOAgent = None
        self.trainer: MCPRLTrainer = None

    async def setup(self):
        """Set up all components for training."""
        logger.info(f"Loading configuration from {self.config_path}")

        # Load configuration
        config_loader = ConfigLoader()
        self.config = config_loader.load_config(str(self.config_path))

        # Setup logging
        self._setup_logging()

        # Initialize MCP client manager
        logger.info("Initializing MCP client manager")
        mcp_manager = MCPClientManager(self.config["mcp_servers"])
        await mcp_manager.connect_all()

        # Initialize LLM provider
        logger.info("Initializing LLM provider")
        llm_provider = LLMProviderFactory.create_provider(
            self.config["llm_provider"]["type"],
            self.config["llm_provider"]["config"]
        )

        # Initialize operator interface (mock for training)
        operator = MockOperatorInterface(self.config.get("operator", {}))

        # Initialize environment with Atari support
        logger.info("Initializing MCP RL Environment with Atari reward system")
        self.environment = MCPRLEnvironment(
            mcp_client_manager=mcp_manager,
            llm_provider=llm_provider,
            operator_interface=operator,
            config=self.config["environment"]
        )

        # Initialize enhanced PPO agent
        logger.info("Initializing Enhanced PPO Agent")
        self.agent = EnhancedPPOAgent(
            action_space_size=self.environment.action_space_size,
            config=self.config["rl"]
        )

        # Initialize trainer
        logger.info("Initializing trainer")
        self.trainer = MCPRLTrainer(
            agent=self.agent,
            environment=self.environment,
            config=self.config["training"]
        )

        logger.info("Setup completed successfully")

    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Set up file logging if specified
        for handler_config in log_config.get("handlers", []):
            if handler_config["type"] == "file":
                file_handler = logging.FileHandler(handler_config["filename"])
                file_handler.setLevel(level)
                formatter = logging.Formatter(log_config.get("format"))
                file_handler.setFormatter(formatter)
                logging.getLogger().addHandler(file_handler)

    async def train(self):
        """Run the training process."""
        if not self.trainer:
            raise RuntimeError("Must call setup() before training")

        logger.info("Starting training process")

        try:
            # Run training
            training_stats = await self.trainer.train()

            logger.info("Training completed successfully")
            logger.info(f"Final training stats: {training_stats}")

            return training_stats

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    async def evaluate(self, num_episodes: int = 10):
        """Evaluate the trained agent."""
        if not self.trainer:
            raise RuntimeError("Must call setup() before evaluation")

        logger.info(f"Starting evaluation with {num_episodes} episodes")

        evaluation_results = await self.trainer.evaluate(num_episodes)

        logger.info("Evaluation completed")
        logger.info(f"Evaluation results: {evaluation_results}")

        return evaluation_results

    async def cleanup(self):
        """Clean up resources."""
        if self.environment:
            self.environment.close()
        logger.info("Cleanup completed")


async def main():
    """Main entry point for Atari training example."""
    parser = argparse.ArgumentParser(description="Train MCP RL Agent on Atari games")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires trained model)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )

    args = parser.parse_args()

    # Initialize training session
    session = AtariTrainingSession(args.config)

    try:
        # Setup
        await session.setup()

        if args.eval_only:
            # Only evaluate
            await session.evaluate(args.eval_episodes)
        else:
            # Train and then evaluate
            await session.train()
            await session.evaluate(args.eval_episodes)

    except Exception as e:
        logger.error(f"Training session failed: {e}")
        return 1

    finally:
        # Cleanup
        await session.cleanup()

    return 0


if __name__ == "__main__":
    # Run the example
    exit_code = asyncio.run(main())
    exit(exit_code)