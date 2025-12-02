"""
LLM Client infrastructure supporting multiple providers.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import LLMProvider, LLMResponse, LLMConfig, RateLimitConfig
from ..core.exceptions import BiasEngineError


logger = logging.getLogger(__name__)


class LLMClientError(BiasEngineError):
    """Base exception for LLM client errors."""
    pass


class RateLimitError(LLMClientError):
    """Raised when rate limits are exceeded."""
    pass


class AuthenticationError(LLMClientError):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(LLMClientError):
    """Raised when requested model is not available."""
    pass


class RateLimiter:
    """Token bucket rate limiter for LLM requests."""

    def __init__(self, config: RateLimitConfig):
        self.requests_per_minute = config.requests_per_minute
        self.tokens_per_minute = config.tokens_per_minute
        self.burst_size = config.burst_size

        self.request_tokens = config.requests_per_minute
        self.token_tokens = config.tokens_per_minute
        self.last_refill = time.time()

        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 100) -> None:
        """Acquire permission for a request with estimated token usage."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_refill

            # Refill tokens based on time passed
            if time_passed > 0:
                request_refill = time_passed * (self.requests_per_minute / 60.0)
                token_refill = time_passed * (self.tokens_per_minute / 60.0)

                self.request_tokens = min(
                    self.burst_size,
                    self.request_tokens + request_refill
                )
                self.token_tokens = min(
                    self.tokens_per_minute,
                    self.token_tokens + token_refill
                )

                self.last_refill = now

            # Check if we can proceed
            if self.request_tokens < 1:
                wait_time = (1 - self.request_tokens) * 60.0 / self.requests_per_minute
                raise RateLimitError(f"Request rate limit exceeded. Wait {wait_time:.2f} seconds.")

            if self.token_tokens < estimated_tokens:
                wait_time = (estimated_tokens - self.token_tokens) * 60.0 / self.tokens_per_minute
                raise RateLimitError(f"Token rate limit exceeded. Wait {wait_time:.2f} seconds.")

            # Consume tokens
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig, rate_limiter: Optional[RateLimiter] = None):
        self.config = config
        self.rate_limiter = rate_limiter
        self.client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @abstractmethod
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make the actual API request to the LLM provider."""
        pass

    @abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse the provider-specific response into our standard format."""
        pass

    @abstractmethod
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token usage for rate limiting."""
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        start_time = time.time()

        # Estimate token usage for rate limiting
        estimated_tokens = self._estimate_tokens(messages)

        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated_tokens)

        try:
            # Make the request
            response_data = await self._make_request(messages, **kwargs)

            # Parse response
            response = self._parse_response(response_data)
            response.response_time = time.time() - start_time

            return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            elif e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {e}")
            elif e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise LLMClientError(f"HTTP error: {e}")

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise LLMClientError(f"Request failed: {e}")


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, config: LLMConfig, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(config, rate_limiter)
        self.base_url = config.base_url or "https://api.openai.com/v1"

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse OpenAI response."""
        choice = response["choices"][0]
        usage = response.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            provider=LLMProvider.OPENAI,
            model=self.config.model,
            tokens_used={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            response_time=0.0  # Will be set by caller
        )

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough token estimation for OpenAI models."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)  # Rough approximation


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(self, config: LLMConfig, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(config, rate_limiter)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make request to Anthropic API."""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Convert messages to Anthropic format
        system_message = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append(msg)

        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": converted_messages
        }

        if system_message:
            payload["system"] = system_message

        response = await self.client.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse Anthropic response."""
        content = response["content"][0]["text"] if response["content"] else ""
        usage = response.get("usage", {})

        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC,
            model=self.config.model,
            tokens_used={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            },
            response_time=0.0
        )

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Token estimation for Anthropic models."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)


class AzureClient(BaseLLMClient):
    """Azure OpenAI client."""

    def __init__(self, config: LLMConfig, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(config, rate_limiter)
        if not config.base_url:
            raise ValueError("Azure client requires base_url in config")

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make request to Azure OpenAI API."""
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Azure uses deployment name in URL
        url = f"{self.config.base_url}/openai/deployments/{self.config.model}/chat/completions"

        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse Azure OpenAI response (same format as OpenAI)."""
        choice = response["choices"][0]
        usage = response.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            provider=LLMProvider.AZURE,
            model=self.config.model,
            tokens_used={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            response_time=0.0
        )

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Token estimation for Azure models."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)


class LLMClient:
    """Factory class for creating LLM clients."""

    _client_classes = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.AZURE: AzureClient,
    }

    @classmethod
    def create(
        self,
        config: LLMConfig,
        rate_limit_config: Optional[RateLimitConfig] = None
    ) -> BaseLLMClient:
        """Create an LLM client based on the provider."""
        if config.provider not in self._client_classes:
            raise ValueError(f"Unsupported provider: {config.provider}")

        rate_limiter = None
        if rate_limit_config:
            rate_limiter = RateLimiter(rate_limit_config)

        client_class = self._client_classes[config.provider]
        return client_class(config, rate_limiter)

    @classmethod
    def get_supported_providers(cls) -> List[LLMProvider]:
        """Get list of supported providers."""
        return list(cls._client_classes.keys())