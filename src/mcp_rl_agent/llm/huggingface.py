"""HuggingFace LLM provider implementation."""

import asyncio
import torch
from typing import List, Optional, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import structlog

from ..interfaces import Message
from .base import RateLimitedLLMProvider

logger = structlog.get_logger(__name__)


class HuggingFaceLLMProvider(RateLimitedLLMProvider):
    """HuggingFace Transformers LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            model_name=config["model_name"],
            requests_per_minute=config.get("requests_per_minute", 60),
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.7)
        )

        self.device = config.get("device", "auto")
        self.dtype = config.get("dtype", "auto")
        self.cache_dir = config.get("cache_dir")
        self.use_local = config.get("use_local", True)

        # Initialize model components
        self._tokenizer = None
        self._model = None
        self._embeddings_model = None
        self._pipeline = None

        # Model loading flags
        self._model_loaded = False
        self._embeddings_loaded = False

    async def _ensure_model_loaded(self) -> None:
        """Ensure the generation model is loaded."""
        if self._model_loaded:
            return

        logger.info("Loading HuggingFace model", model=self.model_name)

        try:
            # Load in a thread to avoid blocking the event loop
            def load_model():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )

                # Add padding token if missing
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                # Load model
                if self.use_local:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        device_map=self.device,
                        torch_dtype=self._get_torch_dtype(),
                        trust_remote_code=True
                    )
                else:
                    # Use pipeline for API-based inference
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self.model_name,
                        tokenizer=self._tokenizer,
                        device_map=self.device,
                        torch_dtype=self._get_torch_dtype(),
                        trust_remote_code=True
                    )

            await asyncio.get_event_loop().run_in_executor(None, load_model)

            self._model_loaded = True
            logger.info("HuggingFace model loaded successfully", model=self.model_name)

        except Exception as e:
            logger.error("Failed to load HuggingFace model", model=self.model_name, error=str(e))
            raise

    async def _ensure_embeddings_loaded(self) -> None:
        """Ensure the embeddings model is loaded."""
        if self._embeddings_loaded:
            return

        logger.info("Loading embeddings model", model=self.model_name)

        try:
            def load_embeddings():
                # Try to use the same model for embeddings, or fall back to a smaller one
                try:
                    if self._tokenizer is None:
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True
                        )

                    self._embeddings_model = AutoModel.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        device_map=self.device,
                        torch_dtype=self._get_torch_dtype(),
                        trust_remote_code=True
                    )
                except Exception:
                    # Fall back to a standard embeddings model
                    logger.warning("Falling back to sentence-transformers model for embeddings")
                    self._embeddings_model = AutoModel.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        cache_dir=self.cache_dir,
                        device_map=self.device,
                        torch_dtype=torch.float32
                    )

            await asyncio.get_event_loop().run_in_executor(None, load_embeddings)

            self._embeddings_loaded = True
            logger.info("Embeddings model loaded successfully")

        except Exception as e:
            logger.error("Failed to load embeddings model", error=str(e))
            # Fall back to base implementation
            pass

    def _get_torch_dtype(self):
        """Get the appropriate torch dtype."""
        if self.dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.dtype == "float16":
            return torch.float16
        elif self.dtype == "float32":
            return torch.float32
        elif self.dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response using HuggingFace model."""
        await self._wait_for_rate_limit()
        await self._ensure_model_loaded()

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        logger.debug("Generating HuggingFace response", message_count=len(messages))

        try:
            # Format messages into a single prompt
            prompt = self._format_messages_to_prompt(messages)

            if self._pipeline:
                # Use pipeline for generation
                response = await self._generate_with_pipeline(prompt, max_tokens, temperature)
            else:
                # Use model directly
                response = await self._generate_with_model(prompt, max_tokens, temperature)

            # Track usage
            estimated_tokens = len(response.split()) * 2
            self._increment_usage(estimated_tokens)

            logger.debug("HuggingFace response generated", response_length=len(response))
            return response

        except Exception as e:
            logger.error("Error generating HuggingFace response", error=str(e))
            raise

    def _format_messages_to_prompt(self, messages: List[Message]) -> str:
        """Format messages into a single prompt string."""
        prompt_parts = []

        for message in messages:
            if message.type.value == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.type.value == "human":
                prompt_parts.append(f"Human: {message.content}")
            elif message.type.value == "agent":
                prompt_parts.append(f"Assistant: {message.content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    async def _generate_with_pipeline(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using pipeline."""
        def generate():
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False
            )
            return outputs[0]["generated_text"].strip()

        return await asyncio.get_event_loop().run_in_executor(None, generate)

    async def _generate_with_model(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using model directly."""
        def generate():
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()

        return await asyncio.get_event_loop().run_in_executor(None, generate)

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings using HuggingFace model."""
        await self._ensure_embeddings_loaded()

        if not self._embeddings_loaded:
            # Fall back to base implementation
            return await super().embed_text(text)

        logger.debug("Generating embeddings", text_length=len(text))

        try:
            def embed():
                inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self._embeddings_model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._embeddings_model(**inputs)

                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                # Apply attention mask and mean pool
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

                # Normalize
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                return normalized.cpu().numpy()[0].astype(np.float32)

            return await asyncio.get_event_loop().run_in_executor(None, embed)

        except Exception as e:
            logger.error("Error generating embeddings", error=str(e))
            # Fall back to base implementation
            return await super().embed_text(text)

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model:
            del self._model
        if hasattr(self, '_embeddings_model') and self._embeddings_model:
            del self._embeddings_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()