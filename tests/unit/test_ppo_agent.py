"""Tests for PPO agent functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from mcp_rl_agent.rl.ppo_agent import PPOAgent, PolicyNetwork, ExperienceBuffer
from mcp_rl_agent.interfaces import AgentState, Conversation, MCPTool, MessageType, Message


class TestPolicyNetwork:
    """Test policy network functionality."""

    def test_initialization(self):
        """Test network initialization."""
        network = PolicyNetwork(
            input_dim=512,
            max_actions=10,
            hidden_size=128,
            n_layers=2
        )

        assert network.input_dim == 512
        assert network.max_actions == 10

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 512)
        logits, value = network(x)

        assert logits.shape == (batch_size, 10)
        assert value.shape == (batch_size,)

    def test_action_masking(self):
        """Test action masking functionality."""
        network = PolicyNetwork(input_dim=64, max_actions=5)

        x = torch.randn(2, 64)
        action_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)

        logits, value = network(x, action_mask)

        # Check that masked actions have very low probability
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, 2] < 1e-6  # Action 2 should be masked for first sample
        assert probs[0, 3] < 1e-6  # Action 3 should be masked for first sample

    def test_get_action_and_value(self):
        """Test action selection and value estimation."""
        network = PolicyNetwork(input_dim=64, max_actions=5)

        x = torch.randn(2, 64)
        action, log_prob, entropy, value = network.get_action_and_value(x)

        assert action.shape == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2,)

        # Test with specific action
        specific_action = torch.tensor([0, 1])
        action, log_prob, entropy, value = network.get_action_and_value(x, action=specific_action)

        assert torch.equal(action, specific_action)


class TestExperienceBuffer:
    """Test experience buffer functionality."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ExperienceBuffer(buffer_size=100, state_dim=64)

        assert buffer.buffer_size == 100
        assert buffer.state_dim == 64
        assert buffer.size == 0
        assert buffer.ptr == 0

    def test_store_and_retrieve(self):
        """Test storing and retrieving experiences."""
        buffer = ExperienceBuffer(buffer_size=10, state_dim=4)

        # Store some experiences
        for i in range(5):
            state = np.array([i, i+1, i+2, i+3], dtype=np.float32)
            action_mask = np.array([1, 1, 0], dtype=np.float32)

            buffer.store(
                state=state,
                action=i % 3,
                reward=float(i),
                value=float(i * 0.5),
                log_prob=float(i * 0.1),
                done=i == 4,
                action_mask=action_mask
            )

        assert buffer.size == 5

        # Get batch
        batch = buffer.get_batch()

        assert batch["states"].shape == (5, 4)
        assert batch["actions"].shape == (5,)
        assert batch["rewards"].shape == (5,)
        assert batch["values"].shape == (5,)
        assert batch["log_probs"].shape == (5,)
        assert batch["dones"].shape == (5,)
        assert batch["action_masks"].shape[0] == 5

        # Check data integrity
        assert batch["states"][0, 0] == 0
        assert batch["actions"][2] == 2
        assert batch["rewards"][3] == 3.0
        assert batch["dones"][-1] == True

    def test_buffer_overflow(self):
        """Test buffer behavior when full."""
        buffer = ExperienceBuffer(buffer_size=3, state_dim=2)

        # Fill buffer beyond capacity
        for i in range(5):
            buffer.store(
                state=np.array([i, i], dtype=np.float32),
                action=i,
                reward=float(i),
                value=0.0,
                log_prob=0.0,
                done=False
            )

        # Size should be capped at buffer_size
        assert buffer.size == 3
        assert buffer.ptr == 5  # But pointer keeps incrementing

    def test_clear(self):
        """Test buffer clearing."""
        buffer = ExperienceBuffer(buffer_size=10, state_dim=2)

        # Add some data
        buffer.store(
            state=np.array([1, 2], dtype=np.float32),
            action=0,
            reward=1.0,
            value=0.5,
            log_prob=0.1,
            done=False
        )

        assert buffer.size == 1

        buffer.clear()
        assert buffer.size == 0
        assert buffer.ptr == 0


class TestPPOAgent:
    """Test PPO agent functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_steps": 10,
            "batch_size": 4,
            "n_epochs": 2,
            "hidden_size": 64,
            "n_layers": 2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "state_dim": 32,
            "max_actions": 5
        }

    def test_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(self.config)

        assert agent.learning_rate == 0.001
        assert agent.gamma == 0.99
        assert agent.state_dim == 32
        assert agent.max_actions == 5
        assert agent.update_count == 0
        assert agent.total_steps == 0

    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(self.config)

        # Create mock state
        conversation = Conversation(
            id="test_conv",
            messages=[Message(MessageType.HUMAN, "Hello", 0.0)],
            active=True
        )

        tools = [
            MCPTool("tool1", "Description 1", {}, "server1"),
            MCPTool("tool2", "Description 2", {}, "server1")
        ]

        state = AgentState(
            conversation=conversation,
            available_actions=tools,
            context_embedding=np.random.randn(32).astype(np.float32),
            step=0
        )

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < len(tools)

    def test_state_to_tensor_conversion(self):
        """Test state to tensor conversion."""
        agent = PPOAgent(self.config)

        # Test with embedding
        conversation = Conversation(id="test", messages=[], active=True)
        state = AgentState(
            conversation=conversation,
            available_actions=[],
            context_embedding=np.random.randn(32).astype(np.float32),
            step=0
        )

        tensor = agent._state_to_tensor(state)
        assert tensor.shape == (32,)
        assert tensor.dtype == torch.float32

        # Test without embedding
        state_no_embed = AgentState(
            conversation=conversation,
            available_actions=[],
            context_embedding=None,
            step=0
        )

        tensor_no_embed = agent._state_to_tensor(state_no_embed)
        assert tensor_no_embed.shape == (32,)
        assert torch.all(tensor_no_embed == 0)

    def test_action_mask_generation(self):
        """Test action mask generation."""
        agent = PPOAgent(self.config)

        tools = [
            MCPTool("tool1", "Description 1", {}, "server1"),
            MCPTool("tool2", "Description 2", {}, "server1")
        ]

        conversation = Conversation(id="test", messages=[], active=True)
        state = AgentState(
            conversation=conversation,
            available_actions=tools,
            step=0
        )

        mask = agent._get_action_mask(state)

        assert mask.shape == (agent.max_actions,)
        assert mask[0] == 1  # First two actions should be available
        assert mask[1] == 1
        assert mask[2] == 0  # Rest should be masked

    def test_update_with_no_experience(self):
        """Test update when no experience available."""
        agent = PPOAgent(self.config)

        # Empty update
        metrics = agent.update([], [], [], [], [])
        assert len(metrics) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test CUDA compatibility if available."""
        config = self.config.copy()
        agent = PPOAgent(config)

        # Should automatically detect and use CUDA if available
        assert agent.device.type in ["cuda", "cpu"]
        assert agent.network.training