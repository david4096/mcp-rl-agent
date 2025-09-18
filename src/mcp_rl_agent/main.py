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
        print(f"\n🔗 Connecting to {len(self.config.mcp_servers)} MCP server(s)...")

        for server_config in self.config.mcp_servers:
            print(f"\n📡 Connecting to: {server_config.name}")
            print(f"   Transport: {server_config.transport}")
            print(f"   URL: {getattr(server_config, 'url', 'N/A')}")

            try:
                await self.mcp_client_manager.add_server(server_config)

                # Get discovered tools and display them
                tools = []
                if server_config.id in self.mcp_client_manager._tools_cache:
                    tools = self.mcp_client_manager._tools_cache[server_config.id]

                print(f"✅ Connected to {server_config.name}")
                print(f"🔧 Discovered {len(tools)} tools:")

                for i, tool in enumerate(tools):
                    description = tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
                    print(f"   [{i}] {tool.name}: {description}")

                logger.info("Connected to MCP server", server_id=server_config.id, tool_count=len(tools))

            except Exception as e:
                print(f"❌ Failed to connect to {server_config.name}: {e}")
                logger.error("Failed to connect to MCP server", server_id=server_config.id, error=str(e))

                # For mock servers, create mock client instead
                if "mock" in server_config.id:
                    print(f"🔄 Using mock client for {server_config.id}")
                    from .mcp.mock_client import MockMCPClient
                    mock_client = MockMCPClient(server_config.id)
                    await mock_client.connect({})
                    tools = await mock_client.discover_tools()
                    print(f"🔧 Mock tools: {[tool.name for tool in tools]}")
                    # Add to client manager (simplified for demo)
                    logger.info("Using mock client", server_id=server_config.id, tool_count=len(tools))

        # Initialize LLM provider
        print(f"🤖 Loading LLM Provider: {self.config.llm.provider}")
        print(f"📝 Model: {self.config.llm.model_name}")

        llm_config = self.config.llm.model_dump()
        self.llm_provider = LLMProviderFactory.create_provider(llm_config)

        print(f"✅ LLM Provider initialized successfully")
        logger.info("Initialized LLM provider", provider=self.config.llm.provider, model=self.config.llm.model_name)

        # Initialize operator interface
        operator_config = self.config.operator.model_dump()
        self.operator_interface = OperatorInterfaceFactory.create_interface(operator_config)
        logger.info("Initialized operator interface", type=self.config.operator.type)

        # Initialize environment
        env_config = self.config.environment.model_dump()
        self.environment = MCPRLEnvironment(
            mcp_client_manager=self.mcp_client_manager,
            llm_provider=self.llm_provider,
            operator_interface=self.operator_interface,
            config=env_config
        )
        logger.info("Initialized RL environment")

        # Initialize PPO agent
        rl_config = self.config.rl.model_dump()
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
                try:
                    # Reset environment for new query-response cycle
                    print("\n" + "="*60)
                    print("🆕 Ready for your next query...")
                    print("="*60)

                    obs = await self.environment.reset()

                    print(f"🔄 Environment reset complete")
                    if hasattr(obs.state, 'available_actions') and obs.state.available_actions:
                        print(f"🛠️  {len(obs.state.available_actions)} tools available:")
                        for i, action in enumerate(obs.state.available_actions[:5]):
                            print(f"   [{i}] {action.name}")
                        if len(obs.state.available_actions) > 5:
                            print(f"   ... and {len(obs.state.available_actions) - 5} more")

                    # PHASE 1: Get human query
                    print(f"\n💭 Please enter your request (or 'quit' to exit):")

                    # Wait for human to enter their query
                    user_query = await self.operator_interface.receive_message()
                    if not user_query or user_query.lower() in ["quit", "exit"]:
                        print("👋 Goodbye!")
                        self.running = False
                        break

                    print(f"\n👤 Human Query: '{user_query}'")
                    print(f"🤖 Agent will now work to fulfill this request...\n")

                    # Add the human query to conversation
                    from .interfaces import MessageType
                    self.environment.conversation_manager.add_message(
                        self.environment.current_conversation_id,
                        MessageType.HUMAN,
                        user_query
                    )

                    # PHASE 2: Agent executes a series of actions to fulfill the query
                    episode_steps = 0
                    max_episode_steps = 20  # Reasonable limit for a single query
                    episode_reward = 0
                    action_chain = []

                    # For RL learning - collect experience
                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    dones = []

                    query_fulfilled = False
                    while not query_fulfilled and self.running and episode_steps < max_episode_steps:
                        print(f"--- Action {episode_steps + 1} ---")

                        # Store current state for RL learning
                        states.append(obs.state)

                        # Agent selects action (with anti-overfitting bias)
                        action = self._select_action_with_diversity_bias(obs.state, action_chain, user_query)

                        # Store action for RL learning
                        actions.append(action)

                        # Show what action was selected
                        if hasattr(obs.state, 'available_actions') and obs.state.available_actions and action < len(obs.state.available_actions):
                            selected_tool = obs.state.available_actions[action]
                            print(f"🤖 Executing: {selected_tool.name}")
                            action_chain.append(selected_tool.name)
                        else:
                            print(f"🤖 Action {action} (out of bounds)")
                            action_chain.append(f"invalid_action_{action}")

                        # Take step in environment
                        next_obs = await self.environment.step(action)
                        episode_steps += 1
                        step_reward = next_obs.reward
                        episode_reward += step_reward

                        # Store experience for RL learning
                        rewards.append(step_reward)
                        next_states.append(next_obs.state)
                        dones.append(next_obs.done)

                        # Check if query seems fulfilled (basic heuristics)
                        query_fulfilled = self._is_query_fulfilled(next_obs, episode_steps, user_query)

                        # Show progress
                        if next_obs.info:
                            success_indicator = "✅" if next_obs.info.get("success", False) else "❌"
                            print(f"   {success_indicator} Result: {next_obs.info.get('action', 'unknown')}")

                        # Update obs for next iteration
                        obs = next_obs

                        # Small pause between actions
                        await asyncio.sleep(0.2)

                    # PHASE 3: Show results and get feedback
                    print(f"\n🏁 Query execution completed!")
                    print(f"📊 Summary:")
                    print(f"   • Actions taken: {episode_steps}")
                    print(f"   • Action chain: {' → '.join(action_chain)}")
                    print(f"   • Total reward: {episode_reward:.3f}")

                    # Show final conversation state
                    if obs.state.conversation and obs.state.conversation.messages:
                        print(f"💬 Final conversation:")
                        recent_messages = obs.state.conversation.messages[-3:]
                        for msg in recent_messages:
                            prefix = "👤" if msg.type.value == "human" else "🤖" if msg.type.value == "agent" else "⚙️"
                            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                            print(f"   {prefix} {msg.type.value}: {content}")

                    # PHASE 4: Get human feedback on the overall performance
                    print(f"\n📋 How well did the agent fulfill your request?")
                    print("   Enter a score from -1 (terrible) to +1 (excellent), or 'good'/'bad':")

                    feedback_response = await self.operator_interface.receive_message()
                    if feedback_response:
                        try:
                            if feedback_response.lower() in ['good', 'great', 'excellent', 'perfect']:
                                human_feedback = 1.0
                            elif feedback_response.lower() in ['bad', 'terrible', 'awful', 'wrong']:
                                human_feedback = -1.0
                            elif feedback_response.lower() in ['ok', 'okay', 'fine']:
                                human_feedback = 0.5
                            else:
                                human_feedback = max(-1.0, min(1.0, float(feedback_response)))

                            print(f"👤 Human feedback: {human_feedback:.1f}")

                            # Apply human feedback to the episode reward
                            final_episode_reward = episode_reward + (human_feedback * 2.0)  # Scale human feedback
                            print(f"🎯 Final episode reward: {final_episode_reward:.3f}")

                        except ValueError:
                            print("⚠️  Invalid feedback format, using neutral (0.0)")
                            human_feedback = 0.0
                            final_episode_reward = episode_reward
                    else:
                        print("⚠️  No feedback received, using automatic reward only")
                        final_episode_reward = episode_reward

                    # PHASE 5: Update RL agent with the complete episode experience
                    if states and len(states) == len(actions) == len(rewards) == len(next_states) == len(dones):
                        print(f"🧠 Training agent on {len(states)} action steps...")

                        # Apply final human feedback to last reward
                        if 'human_feedback' in locals() and rewards:
                            rewards[-1] += human_feedback * 1.0  # Boost last reward with human feedback

                        # Update the agent with the complete episode
                        metrics = self.agent.update(states, actions, rewards, next_states, dones)

                        print(f"📈 Training metrics: {metrics}")
                        logger.info("RL agent updated", metrics=metrics)
                    else:
                        print(f"⚠️  Skipping RL update due to mismatched experience lengths")

                    # Log the learning episode
                    logger.info("Interactive episode completed",
                               query=user_query,
                               steps=episode_steps,
                               actions=action_chain,
                               reward=final_episode_reward,
                               human_feedback=human_feedback if 'human_feedback' in locals() else 0.0)

                    print(f"\n✅ Episode complete! Agent learned from {episode_steps} actions. Ready for your next request...\n")

                except Exception as e:
                    logger.error("Error in conversation episode", error=str(e))
                    print(f"Error in conversation: {e}")
                    # Wait before retrying to prevent rapid error loops
                    await asyncio.sleep(2.0)

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

    def _is_query_fulfilled(self, obs, episode_steps: int, user_query: str) -> bool:
        """Determine if the user's query has been reasonably fulfilled."""

        # Never stop immediately - allow at least 3 actions
        if episode_steps < 3:
            return False

        # Basic stopping conditions
        if obs.done:
            return True

        query_lower = user_query.lower()

        # Game-related queries need proper setup sequence
        if any(word in query_lower for word in ['play', 'game', 'start', 'run']):
            # Check if we have a complete game setup sequence
            if obs.state.conversation and obs.state.conversation.messages:
                # Look for evidence of successful game setup
                agent_messages = [msg.content.lower() for msg in obs.state.conversation.messages
                                if msg.type.value == "agent"]

                # Need evidence of game switching AND environment interaction
                has_game_switch = any("switch" in msg or "breakout" in msg for msg in agent_messages)
                has_env_action = any("reset" in msg or "step" in msg or "observation" in msg for msg in agent_messages)

                # Only consider fulfilled if we have both game setup AND some interaction
                if has_game_switch and has_env_action and episode_steps >= 5:
                    print(f"🎮 Game setup sequence detected, considering query fulfilled")
                    return True

        # For non-game queries, need meaningful conversation
        else:
            if obs.state.conversation and obs.state.conversation.messages:
                # Count substantive agent responses (not just acknowledgments)
                agent_responses = [msg for msg in obs.state.conversation.messages
                                 if msg.type.value == "agent" and len(msg.content.strip()) > 50]

                # Need multiple meaningful exchanges
                if len(agent_responses) >= 2 and episode_steps >= 4:
                    return True

        # Safety valve - don't run too long
        if episode_steps >= 15:
            print(f"🛑 Max episode length reached, ending query")
            return True

        return False

    def _select_action_with_diversity_bias(self, state, action_chain: list, user_query: str) -> int:
        """Select action with bias against repetitive patterns and towards query-relevant actions."""

        # Get base action from agent
        base_action = self.agent.select_action(state)

        if not hasattr(state, 'available_actions') or not state.available_actions:
            return base_action

        available_tools = state.available_actions
        if base_action >= len(available_tools):
            return base_action

        selected_tool_name = available_tools[base_action].name

        # Anti-overfitting penalties
        repetition_penalty = 0.0

        # Heavy penalty for immediate repetition
        if action_chain and action_chain[-1] == selected_tool_name:
            repetition_penalty += 0.8

        # Moderate penalty for frequent use in this chain
        if action_chain.count(selected_tool_name) >= 2:
            repetition_penalty += 0.5

        # Specific penalty for overused tools
        if selected_tool_name == "get_available_games":
            if action_chain.count("get_available_games") >= 1:
                repetition_penalty += 1.0  # Strong penalty for multiple game lists

        # Query relevance bonus
        query_bonus = 0.0
        query_lower = user_query.lower()
        tool_name_lower = selected_tool_name.lower()

        # Direct action matching
        if "play" in query_lower or "start" in query_lower:
            if "switch" in tool_name_lower:
                query_bonus += 0.5
            elif "reset" in tool_name_lower:
                query_bonus += 0.3
            elif "step" in tool_name_lower:
                query_bonus += 0.2

        # Game name matching
        if "breakout" in query_lower and "switch" in tool_name_lower:
            query_bonus += 0.7

        # Calculate bias adjustment
        bias_adjustment = query_bonus - repetition_penalty

        # If the bias is strongly negative, try to find a better action
        if bias_adjustment < -0.5:
            print(f"🔄 Exploring alternatives to {selected_tool_name} (bias: {bias_adjustment:.2f})")

            # Find query-relevant alternatives
            best_alternative = None
            best_score = -999

            for i, tool in enumerate(available_tools):
                if i == base_action:
                    continue  # Skip the originally selected action

                tool_score = 0.0

                # Boost direct action tools for game queries
                if "play" in query_lower or "game" in query_lower:
                    if "switch" in tool.name.lower():
                        tool_score += 1.0
                    elif "reset" in tool.name.lower() and "switch_game" in action_chain:
                        tool_score += 0.8
                    elif "step" in tool.name.lower() and any(x in action_chain for x in ["reset_environment", "switch_game"]):
                        tool_score += 0.6

                # Penalize already used tools
                tool_score -= action_chain.count(tool.name) * 0.3

                if tool_score > best_score:
                    best_score = tool_score
                    best_alternative = i

            # Use alternative if it's significantly better
            if best_alternative is not None and best_score > 0.3:
                print(f"🎯 Switching to {available_tools[best_alternative].name} (score: {best_score:.2f})")
                return best_alternative

        return base_action


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