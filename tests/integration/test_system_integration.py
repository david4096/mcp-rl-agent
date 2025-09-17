"""Integration tests for the complete system."""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path

from mcp_rl_agent.main import MCPRLAgentSystem
from mcp_rl_agent.config.manager import ConfigurationManager


class TestSystemIntegration:
    """Test complete system integration."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal mock configuration for testing."""
        config_data = {
            "mcp_servers": [
                {
                    "id": "mock_basic",
                    "name": "Mock Basic Tools Server",
                    "transport": "stdio",
                    "command": ["python", "-m", "mcp_rl_agent.mcp.mock_server", "stdio"]
                }
            ],
            "llm": {
                "provider": "mock",
                "model_name": "mock_model",
                "responses": [
                    "I'll help you with that.",
                    "Let me use the appropriate tool.",
                    "Task completed successfully."
                ],
                "delay": 0.0
            },
            "rl": {
                "learning_rate": 0.01,
                "n_steps": 5,
                "batch_size": 2,
                "n_epochs": 1,
                "hidden_size": 32,
                "state_dim": 64,
                "max_actions": 10
            },
            "environment": {
                "max_episode_length": 10,
                "embedding_dim": 64
            },
            "operator": {
                "type": "mock",
                "responses": [
                    "Hello, can you help me?",
                    "That's good, thank you."
                ],
                "feedback_scores": [0.8, 0.9],
                "delay": 0.0
            },
            "logging": {
                "level": "ERROR"  # Reduce noise in tests
            },
            "seed": 42
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_system_initialization(self, mock_config):
        """Test that the system initializes correctly."""
        system = MCPRLAgentSystem(mock_config)

        try:
            await system.initialize()

            # Check that all components are initialized
            assert system.config is not None
            assert system.mcp_client_manager is not None
            assert system.llm_provider is not None
            assert system.operator_interface is not None
            assert system.environment is not None
            assert system.agent is not None

            # Check LLM provider
            assert system.llm_provider.model_name == "mock_model"

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_environment_reset(self, mock_config):
        """Test environment reset functionality."""
        system = MCPRLAgentSystem(mock_config)

        try:
            await system.initialize()

            # Reset environment
            obs = await system.environment.reset()

            assert obs is not None
            assert obs.state is not None
            assert obs.state.conversation is not None
            assert not obs.done
            assert obs.reward == 0.0

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_single_step_execution(self, mock_config):
        """Test a single step of agent-environment interaction."""
        system = MCPRLAgentSystem(mock_config)

        try:
            await system.initialize()

            # Reset environment
            obs = await system.environment.reset()

            # Agent selects action
            action = system.agent.select_action(obs.state)
            assert isinstance(action, int)

            # Take step in environment
            next_obs = await system.environment.step(action)

            assert next_obs is not None
            assert next_obs.state is not None
            assert isinstance(next_obs.reward, float)
            assert isinstance(next_obs.done, bool)

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_episode_completion(self, mock_config):
        """Test completion of a full episode."""
        system = MCPRLAgentSystem(mock_config)

        try:
            await system.initialize()

            # Reset environment
            obs = await system.environment.reset()
            step_count = 0
            max_steps = 15  # Safety limit

            while not obs.done and step_count < max_steps:
                # Agent selects action
                action = system.agent.select_action(obs.state)

                # Take step in environment
                obs = await system.environment.step(action)
                step_count += 1

            # Episode should either complete naturally or hit step limit
            assert step_count > 0
            assert step_count <= max_steps

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_training_loop(self, mock_config):
        """Test a short training loop."""
        system = MCPRLAgentSystem(mock_config)

        try:
            await system.initialize()

            # Run short training loop
            await system.run_training(num_episodes=2)

            # Check that agent has been updated
            assert system.agent.total_steps > 0

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_config_loading(self, mock_config):
        """Test configuration loading and validation."""
        # Test loading configuration
        config_manager = ConfigurationManager()
        config_manager.load_config(mock_config)
        config = config_manager.get_config()

        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].id == "mock_basic"
        assert config.llm.provider.value == "mock"
        assert config.operator.type.value == "mock"

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_config):
        """Test that system shuts down gracefully."""
        system = MCPRLAgentSystem(mock_config)

        await system.initialize()
        assert system.running == False  # Not started yet

        # Shutdown should work even if not running
        await system.shutdown()
        assert system.running == False

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config):
        """Test error handling in system components."""
        # Test with invalid config path
        system = MCPRLAgentSystem("nonexistent_config.yaml")

        # Should create default config
        await system.initialize()
        assert system.config is not None

        await system.shutdown()