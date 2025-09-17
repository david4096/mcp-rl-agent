"""Main entry point for the MCP RL Agent system."""

import asyncio
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional
import structlog

from .config.manager import ConfigurationManager
from .mcp.client import MCPClientManager
from .mcp.mock_client import MockMCPClientFactory
from .llm.base import LLMProviderFactory
from .interface.operator import OperatorInterfaceFactory
from .env.mcp_env import MCPRLEnvironment
from .rl.ppo_agent import PPOAgent

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MCPRLAgentSystem:
    """Main system orchestrator for the MCP RL Agent."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigurationManager(config_path)
        self.config = None

        # System components
        self.mcp_client_manager: Optional[MCPClientManager] = None
        self.llm_provider = None
        self.operator_interface = None
        self.environment: Optional[MCPRLEnvironment] = None
        self.agent: Optional[PPOAgent] = None

        # State
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all system components."""
        logger.info("Initializing MCP RL Agent system")

        # Load configuration
        self.config = self.config_manager.get_config()

        # Set random seed if specified
        if self.config.seed is not None:
            import random
            import numpy as np
            import torch
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            logger.info("Set random seed", seed=self.config.seed)

        # Initialize MCP client manager
        self.mcp_client_manager = MCPClientManager()

        # Connect to MCP servers
        for server_config in self.config.mcp_servers:
            try:
                await self.mcp_client_manager.add_server(server_config)
                logger.info("Connected to MCP server", server_id=server_config.id)
            except Exception as e:
                logger.error("Failed to connect to MCP server", server_id=server_config.id, error=str(e))
                # For mock servers, create mock client instead
                if "mock" in server_config.id:
                    from .mcp.mock_client import MockMCPClient
                    mock_client = MockMCPClient(server_config.id)
                    await mock_client.connect({})
                    tools = await mock_client.discover_tools()
                    # Add to client manager (simplified for demo)
                    logger.info("Using mock client", server_id=server_config.id, tool_count=len(tools))

        # Initialize LLM provider
        llm_config = self.config.llm.dict()
        self.llm_provider = LLMProviderFactory.create_provider(llm_config)
        logger.info("Initialized LLM provider", provider=self.llm_provider.model_name)

        # Initialize operator interface
        operator_config = self.config.operator.dict()
        self.operator_interface = OperatorInterfaceFactory.create_interface(operator_config)
        logger.info("Initialized operator interface", type=self.config.operator.type)

        # Initialize environment
        env_config = self.config.environment.dict()
        self.environment = MCPRLEnvironment(
            mcp_client_manager=self.mcp_client_manager,
            llm_provider=self.llm_provider,
            operator_interface=self.operator_interface,
            config=env_config
        )
        logger.info("Initialized RL environment")

        # Initialize PPO agent
        rl_config = self.config.rl.dict()
        rl_config.update({
            "state_dim": env_config.get("embedding_dim", 512),
            "max_actions": 50  # Will be updated based on available tools
        })
        self.agent = PPOAgent(rl_config)
        logger.info("Initialized PPO agent")

        logger.info("System initialization complete")

    async def run_training(self, num_episodes: int = 100) -> None:
        """Run training loop."""
        logger.info("Starting training", num_episodes=num_episodes)
        self.running = True

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            if not self.running:
                break

            logger.info("Starting episode", episode=episode + 1)

            try:
                # Reset environment
                obs = await self.environment.reset()

                episode_reward = 0.0
                episode_length = 0
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

                while not obs.done and self.running:
                    # Agent selects action
                    action = self.agent.select_action(obs.state)

                    # Store current state
                    states.append(obs.state)
                    actions.append(action)

                    # Take step in environment
                    next_obs = await self.environment.step(action)

                    # Store experience
                    rewards.append(next_obs.reward)
                    next_states.append(next_obs.state)
                    dones.append(next_obs.done)

                    # Update metrics
                    episode_reward += next_obs.reward
                    episode_length += 1

                    # Move to next observation
                    obs = next_obs

                # Update agent with episode data
                if states:
                    metrics = self.agent.update(states, actions, rewards, next_states, dones)

                    logger.info("Episode completed",
                               episode=episode + 1,
                               reward=episode_reward,
                               length=episode_length,
                               **metrics)

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Log average performance every 10 episodes
                if (episode + 1) % 10 == 0:
                    avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                    avg_length = sum(episode_lengths[-10:]) / min(10, len(episode_lengths))
                    logger.info("Training progress",
                               episodes_completed=episode + 1,
                               avg_reward_10=avg_reward,
                               avg_length_10=avg_length)

            except Exception as e:
                logger.error("Error in training episode", episode=episode + 1, error=str(e))
                continue

        logger.info("Training completed", total_episodes=len(episode_rewards))

    async def run_interactive(self) -> None:
        """Run interactive mode for testing and demonstration."""
        logger.info("Starting interactive mode")
        self.running = True

        print("\n=== MCP RL Agent Interactive Mode ===")
        print("The agent will interact with you and learn from your feedback.")
        print("Type 'quit' or 'exit' to stop.\n")

        try:
            while self.running:
                # Reset environment for new conversation
                obs = await self.environment.reset()

                print("Starting new conversation...")

                while not obs.done and self.running:
                    # Agent selects action
                    action = self.agent.select_action(obs.state)

                    # Take step in environment
                    obs = await self.environment.step(action)

                    # Check for user quit command
                    if obs.state.conversation and obs.state.conversation.messages:
                        last_msg = obs.state.conversation.messages[-1]
                        if last_msg.type.value == "human" and last_msg.content.lower() in ["quit", "exit"]:
                            self.running = False
                            break

        except KeyboardInterrupt:
            logger.info("Interactive mode interrupted by user")
        except Exception as e:
            logger.error("Error in interactive mode", error=str(e))

        print("\nGoodbye!")

    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        logger.info("Shutting down MCP RL Agent system")
        self.running = False

        # Close environment
        if self.environment:
            self.environment.close()

        # Disconnect from MCP servers
        if self.mcp_client_manager:
            await self.mcp_client_manager.disconnect_all()

        # Close operator interface
        if hasattr(self.operator_interface, 'disconnect'):
            await self.operator_interface.disconnect()

        logger.info("System shutdown complete")


def setup_signal_handlers(system: MCPRLAgentSystem):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Received signal, initiating shutdown", signal=signum)
        asyncio.create_task(system.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP RL Agent")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "interactive"],
        default="interactive",
        help="Run mode: train or interactive"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=100,
        help="Number of training episodes (train mode only)"
    )

    args = parser.parse_args()

    # Create system
    system = MCPRLAgentSystem(args.config)

    # Setup signal handlers
    setup_signal_handlers(system)

    try:
        # Initialize system
        await system.initialize()

        # Run in selected mode
        if args.mode == "train":
            await system.run_training(args.episodes)
        else:
            await system.run_interactive()

    except Exception as e:
        logger.error("System error", error=str(e))
        sys.exit(1)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())