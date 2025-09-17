"""Tests for LLM providers."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from mcp_rl_agent.llm.base import MockLLMProvider, LLMProviderFactory
from mcp_rl_agent.interfaces import Message, MessageType


class TestMockLLMProvider:
    """Test mock LLM provider functionality."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = MockLLMProvider(
            model_name="test_model",
            responses=["Response 1", "Response 2"],
            delay=0.0
        )

        assert provider.model_name == "test_model"
        assert len(provider.responses) == 2
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation."""
        provider = MockLLMProvider(
            model_name="test_model",
            responses=["Hello", "How are you?"],
            delay=0.0
        )

        messages = [
            Message(MessageType.HUMAN, "Hi there", 0.0)
        ]

        # First response
        response1 = await provider.generate_response(messages)
        assert response1 == "Hello"
        assert provider.call_count == 1

        # Second response
        response2 = await provider.generate_response(messages)
        assert response2 == "How are you?"
        assert provider.call_count == 2

        # Should cycle back to first
        response3 = await provider.generate_response(messages)
        assert response3 == "Hello"

    @pytest.mark.asyncio
    async def test_embed_text(self):
        """Test text embedding."""
        provider = MockLLMProvider()

        embedding = await provider.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 768  # Standard embedding size

        # Same text should produce same embedding
        embedding2 = await provider.embed_text("Hello world")
        np.testing.assert_array_equal(embedding, embedding2)

        # Different text should produce different embedding
        embedding3 = await provider.embed_text("Different text")
        assert not np.array_equal(embedding, embedding3)


class TestLLMProviderFactory:
    """Test LLM provider factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        config = {
            "provider": "mock",
            "model_name": "test_model",
            "max_tokens": 512,
            "temperature": 0.5
        }

        provider = LLMProviderFactory.create_provider(config)

        assert isinstance(provider, MockLLMProvider)
        assert provider.model_name == "test_model"
        assert provider.max_tokens == 512
        assert provider.temperature == 0.5

    def test_unknown_provider(self):
        """Test error for unknown provider type."""
        config = {
            "provider": "unknown_provider",
            "model_name": "test"
        }

        with pytest.raises(ValueError, match="Unknown LLM provider type"):
            LLMProviderFactory.create_provider(config)

    @patch('mcp_rl_agent.llm.huggingface.HuggingFaceLLMProvider')
    def test_create_huggingface_provider(self, mock_hf_class):
        """Test creating HuggingFace provider."""
        config = {
            "provider": "huggingface",
            "model_name": "microsoft/DialoGPT-medium"
        }

        provider = LLMProviderFactory.create_provider(config)
        mock_hf_class.assert_called_once_with(config)

    @patch('mcp_rl_agent.llm.claude.ClaudeLLMProvider')
    def test_create_claude_provider(self, mock_claude_class):
        """Test creating Claude provider."""
        config = {
            "provider": "claude",
            "model_name": "claude-3-sonnet-20240229"
        }

        provider = LLMProviderFactory.create_provider(config)
        mock_claude_class.assert_called_once_with(config)