"""Claude LLM provider implementation."""

import asyncio
import os
from typing import List, Optional, Dict, Any
import numpy as np
import anthropic
import structlog

from ..interfaces import Message
from .base import RateLimitedLLMProvider

logger = structlog.get_logger(__name__)


class ClaudeLLMProvider(RateLimitedLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            model_name=config["model_name"],
            requests_per_minute=config.get("requests_per_minute", 60),
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.7)
        )

        # API configuration
        api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.api_url = config.get("api_url")
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.api_url
        )

        # Claude-specific parameters
        self.top_p = config.get("top_p", 0.95)
        self.system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")

    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response using Claude API."""
        await self._wait_for_rate_limit()

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        logger.debug("Generating Claude response", message_count=len(messages))

        try:
            # Format messages for Claude API
            claude_messages = self._format_messages_for_claude(messages)

            # Make API call
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                system=self._extract_system_message(messages),
                messages=claude_messages
            )

            # Extract text content
            text_content = ""
            for content in response.content:
                if content.type == "text":
                    text_content += content.text

            # Track usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            self._increment_usage(total_tokens)

            logger.debug("Claude response generated",
                        response_length=len(text_content),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens)

            return text_content.strip()

        except anthropic.RateLimitError as e:
            logger.warning("Claude rate limit exceeded", error=str(e))
            # Wait and retry once
            await asyncio.sleep(60)  # Wait 1 minute
            return await self.generate_response(messages, max_tokens, temperature)

        except anthropic.APIError as e:
            logger.error("Claude API error", error=str(e))
            raise Exception(f"Claude API error: {e}")

        except Exception as e:
            logger.error("Error generating Claude response", error=str(e))
            raise

    def _format_messages_for_claude(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for Claude API (excluding system messages)."""
        claude_messages = []

        for message in messages:
            if message.type.value == "system":
                continue  # System messages are handled separately

            role = "user" if message.type.value == "human" else "assistant"
            claude_messages.append({
                "role": role,
                "content": message.content
            })

        return claude_messages

    def _extract_system_message(self, messages: List[Message]) -> str:
        """Extract system message content, or use default."""
        system_messages = [msg.content for msg in messages if msg.type.value == "system"]

        if system_messages:
            return "\n".join(system_messages)
        else:
            return self.system_prompt

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text.

        Note: Claude doesn't provide embeddings API, so we fall back to a simple approach.
        In a production system, you'd want to use a dedicated embeddings service.
        """
        logger.debug("Generating embeddings (using fallback method)", text_length=len(text))

        # Use the base class implementation (simple hash-based embedding)
        return await super().embed_text(text)


class ClaudeEmbeddingProvider:
    """Separate provider for embeddings that works with Claude systems.

    Since Claude doesn't provide embeddings directly, this uses a lightweight
    alternative like sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    async def _ensure_model_loaded(self):
        """Load the embeddings model if not already loaded."""
        if self._model is not None:
            return

        try:
            def load_model():
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)

            await asyncio.get_event_loop().run_in_executor(None, load_model)
            logger.info("Embeddings model loaded", model=self.model_name)

        except ImportError:
            logger.warning("sentence-transformers not available, falling back to transformers")
            # Fall back to transformers
            def load_transformers_model():
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

            await asyncio.get_event_loop().run_in_executor(None, load_transformers_model)

        except Exception as e:
            logger.error("Failed to load embeddings model", error=str(e))
            raise

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text."""
        await self._ensure_model_loaded()

        try:
            def embed():
                if hasattr(self._model, 'encode'):
                    # sentence-transformers model
                    embedding = self._model.encode(text, convert_to_numpy=True)
                    return embedding.astype(np.float32)
                else:
                    # transformers model
                    import torch
                    inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

                    with torch.no_grad():
                        outputs = self._model(**inputs)

                    # Mean pooling
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                    pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

                    # Normalize
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return normalized.cpu().numpy()[0].astype(np.float32)

            return await asyncio.get_event_loop().run_in_executor(None, embed)

        except Exception as e:
            logger.error("Error generating embeddings", error=str(e))
            # Fall back to simple hash-based embedding
            hash_value = hash(text)
            np.random.seed(abs(hash_value) % (2**32))
            embedding = np.random.normal(0, 1, 384)  # Smaller size for fallback
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)


class ClaudeWithEmbeddingsProvider(ClaudeLLMProvider):
    """Claude provider with proper embeddings support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        embeddings_model = config.get("embeddings_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_provider = ClaudeEmbeddingProvider(embeddings_model)

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings using the dedicated embeddings provider."""
        return await self.embedding_provider.embed_text(text)