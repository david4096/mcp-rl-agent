"""Base LLM provider implementation with common functionality."""

import asyncio
import time
from typing import List, Optional, Dict, Any
import numpy as np
import structlog

from ..interfaces import LLMProviderInterface, Message

logger = structlog.get_logger(__name__)


class BaseLLMProvider(LLMProviderInterface):
    """Base class for LLM providers with common functionality."""

    def __init__(self, model_name: str, max_tokens: int = 1024, temperature: float = 0.7):
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._call_count = 0
        self._total_tokens = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        """Get the number of API calls made."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Get the total tokens processed."""
        return self._total_tokens

    def _format_messages_for_api(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for API calls."""
        formatted_messages = []
        for message in messages:
            role = self._map_message_type_to_role(message.type.value)
            formatted_messages.append({
                "role": role,
                "content": message.content
            })
        return formatted_messages

    def _map_message_type_to_role(self, message_type: str) -> str:
        """Map message types to API roles."""
        mapping = {
            "human": "user",
            "agent": "assistant",
            "system": "system"
        }
        return mapping.get(message_type, "user")

    def _increment_usage(self, tokens: int = 0) -> None:
        """Track usage statistics."""
        self._call_count += 1
        self._total_tokens += tokens

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text. Base implementation uses simple hashing."""
        # This is a placeholder implementation
        # Real implementations would use proper embedding models
        hash_value = hash(text)
        # Create a simple pseudo-embedding
        np.random.seed(abs(hash_value) % (2**32))
        embedding = np.random.normal(0, 1, 768)  # Standard embedding size
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding.astype(np.float32)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        model_name: str = "mock_model",
        responses: Optional[List[str]] = None,
        delay: float = 0.1,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.responses = responses or [
            "I understand. Let me help you with that.",
            "That's an interesting request. I'll work on it.",
            "I've completed the requested action.",
            "Let me think about the best approach for this.",
            "I'll use the available tools to accomplish this task."
        ]
        self.delay = delay
        self._response_index = 0

    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a mock response."""
        logger.debug("Generating mock response", message_count=len(messages))
        print(f"🔮 MockLLM generating response (model: {self.model_name}, messages: {len(messages)})")

        # Simulate processing delay
        if self.delay > 0:
            print(f"⏳ Simulating {self.delay}s processing delay...")
            await asyncio.sleep(self.delay)

        # Cycle through predefined responses
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        # Track usage
        estimated_tokens = len(response.split()) * 2  # Rough estimate
        self._increment_usage(estimated_tokens)

        logger.debug("Mock response generated", response_length=len(response))
        print(f"✅ MockLLM response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        return response


class RateLimitedLLMProvider(BaseLLMProvider):
    """Base class for LLM providers that need rate limiting."""

    def __init__(self, model_name: str, requests_per_minute: int = 60, **kwargs):
        super().__init__(model_name, **kwargs)
        self.requests_per_minute = requests_per_minute
        self._request_times: List[float] = []

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()

        # Remove requests older than 1 minute
        cutoff_time = now - 60
        self._request_times = [t for t in self._request_times if t > cutoff_time]

        # If we're at the rate limit, wait
        if len(self._request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.info("Rate limit reached, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)

        # Record this request
        self._request_times.append(now)


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(config: Dict[str, Any]) -> LLMProviderInterface:
        """Create an LLM provider based on configuration."""
        provider_type = config.get("provider", "mock").lower()
        model_name = config.get("model_name", "mock_model")

        logger.info("Creating LLM provider", provider_type=provider_type, model_name=model_name)

        if provider_type == "mock":
            provider = MockLLMProvider(
                model_name=model_name,
                max_tokens=config.get("max_tokens", 1024),
                temperature=config.get("temperature", 0.7),
                responses=config.get("responses"),
                delay=config.get("delay", 0.1)
            )
            logger.info("Mock LLM provider created successfully", model_name=model_name)
            return provider
        elif provider_type == "huggingface":
            from .huggingface import HuggingFaceLLMProvider
            provider = HuggingFaceLLMProvider(config)
            logger.info("HuggingFace LLM provider created successfully", model_name=model_name)
            return provider
        elif provider_type == "claude":
            from .claude import ClaudeLLMProvider
            provider = ClaudeLLMProvider(config)
            logger.info("Claude LLM provider created successfully", model_name=model_name)
            return provider
        else:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")


class LLMProviderManager:
    """Manages multiple LLM providers with fallback support."""

    def __init__(self, primary_provider: LLMProviderInterface, fallback_providers: Optional[List[LLMProviderInterface]] = None):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self._current_provider = primary_provider

    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response with fallback support."""
        providers_to_try = [self.primary_provider] + self.fallback_providers

        last_error = None
        for i, provider in enumerate(providers_to_try):
            try:
                logger.debug("Attempting response generation", provider=provider.model_name, attempt=i + 1)
                response = await provider.generate_response(messages, max_tokens, temperature)

                # If we successfully used a fallback provider, log it
                if i > 0:
                    logger.warning("Used fallback provider", provider=provider.model_name, attempt=i + 1)

                self._current_provider = provider
                return response

            except Exception as e:
                last_error = e
                logger.warning("Provider failed", provider=provider.model_name, error=str(e))
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings using the current provider."""
        return await self._current_provider.embed_text(text)

    @property
    def model_name(self) -> str:
        return self._current_provider.model_name

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all providers."""
        stats = {}
        for provider in [self.primary_provider] + self.fallback_providers:
            if hasattr(provider, 'call_count'):
                stats[provider.model_name] = {
                    'calls': provider.call_count,
                    'tokens': provider.total_tokens
                }
        return stats