"""Anthropic LLM Integration.

Provides seamless integration with Anthropic's Claude API.

Features:
- Full support for Claude 3, Claude 2, and Instant models
- Streaming completions
- Token counting
- Cost tracking
- Automatic retries
- Async support

Example:
    >>> from agent_memory_toolkit.llm import AnthropicProvider, AnthropicConfig
    >>> 
    >>> provider = AnthropicProvider(AnthropicConfig(
    ...     model=AnthropicModel.CLAUDE_3_5_SONNET,
    ...     api_key="sk-ant-...",
    ... ))
    >>> 
    >>> response = provider.complete("What is machine learning?")
    >>> print(response)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    TokenUsage,
    ModelCapabilities,
    StreamingCallback,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)


class AnthropicModel(str, Enum):
    """Available Anthropic models."""
    # Claude 3.5 family
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_5_HAIKU_LATEST = "claude-3-5-haiku-latest"
    
    # Claude 3 family
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_OPUS_LATEST = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Claude 2 family (legacy)
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT = "claude-instant-1.2"


# Model capabilities lookup
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "claude-3-5-sonnet-20241022": ModelCapabilities(
        max_tokens=8192,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=False,  # Use tool_choice for structured output
        context_window=200000,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    "claude-3-5-haiku-20241022": ModelCapabilities(
        max_tokens=8192,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=False,
        context_window=200000,
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    ),
    "claude-3-opus-20240229": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=False,
        context_window=200000,
        input_price_per_million=15.0,
        output_price_per_million=75.0,
    ),
    "claude-3-sonnet-20240229": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=False,
        context_window=200000,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    "claude-3-haiku-20240307": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=False,
        context_window=200000,
        input_price_per_million=0.25,
        output_price_per_million=1.25,
    ),
    "claude-2.1": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        context_window=200000,
        input_price_per_million=8.0,
        output_price_per_million=24.0,
    ),
    "claude-2.0": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        context_window=100000,
        input_price_per_million=8.0,
        output_price_per_million=24.0,
    ),
    "claude-instant-1.2": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        context_window=100000,
        input_price_per_million=0.80,
        output_price_per_million=2.40,
    ),
}


@dataclass
class AnthropicConfig(LLMConfig):
    """Configuration for Anthropic provider.
    
    Attributes:
        model: Anthropic model to use (e.g., "claude-3-5-sonnet-20241022")
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        base_url: Custom API base URL (for proxies)
        
    Example:
        >>> config = AnthropicConfig(
        ...     model=AnthropicModel.CLAUDE_3_5_SONNET,
        ...     api_key="sk-ant-...",
        ...     temperature=0.7,
        ... )
    """
    model: str = "claude-3-5-sonnet-20241022"
    
    # Anthropic-specific settings
    metadata: Optional[dict[str, str]] = None  # Request metadata


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider.
    
    Provides integration with Anthropic's Claude API.
    Supports Claude 3.5, Claude 3, Claude 2, and Instant models.
    
    Example:
        >>> provider = AnthropicProvider(AnthropicConfig(model="claude-3-5-sonnet-20241022"))
        >>> response = provider.complete("Hello, world!")
        >>> print(response)
        
        >>> # With system prompt
        >>> response = provider.complete_with_system(
        ...     system="You are a helpful assistant.",
        ...     prompt="What is Python?",
        ... )
    """
    
    def __init__(self, config: Optional[AnthropicConfig] = None):
        """Initialize Anthropic provider.
        
        Args:
            config: Provider configuration. If None, uses defaults with
                    ANTHROPIC_API_KEY environment variable.
        """
        if config is None:
            config = AnthropicConfig()
        
        super().__init__(config)
        self.config: AnthropicConfig = config
        
        # Get API key from config or environment
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AuthenticationError("Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key in config.")
        
        # Lazy-load the Anthropic client
        self._client = None
    
    @property
    def client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic integration. "
                    "Install with: pip install anthropic"
                )
            
            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
            }
            
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            
            if self.config.timeout:
                kwargs["timeout"] = self.config.timeout
            
            if self.config.max_retries:
                kwargs["max_retries"] = self.config.max_retries
            
            self._client = Anthropic(**kwargs)
        
        return self._client
    
    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Get capabilities of the current model."""
        model = self.config.model
        
        # Check exact match first
        if model in MODEL_CAPABILITIES:
            return MODEL_CAPABILITIES[model]
        
        # Check for "latest" aliases
        for base_model, caps in MODEL_CAPABILITIES.items():
            if model.replace("-latest", "") in base_model:
                return caps
        
        # Default capabilities for Claude 3
        return ModelCapabilities(
            max_tokens=4096,
            supports_system_prompt=True,
            supports_streaming=True,
            context_window=200000,
        )
    
    def _calculate_cost(self, usage: TokenUsage) -> TokenUsage:
        """Calculate cost for token usage."""
        caps = self.model_capabilities
        
        if caps.input_price_per_million and caps.output_price_per_million:
            usage.prompt_cost = (usage.prompt_tokens / 1_000_000) * caps.input_price_per_million
            usage.completion_cost = (usage.completion_tokens / 1_000_000) * caps.output_price_per_million
            usage.total_cost = usage.prompt_cost + usage.completion_cost
        
        return usage
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt.
        
        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        response = self.complete_full(prompt=prompt, max_tokens=max_tokens)
        return response.content
    
    def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500,
    ) -> str:
        """Generate completion with system prompt.
        
        Args:
            system: System prompt for context
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        response = self.complete_full(prompt=prompt, system=system, max_tokens=max_tokens)
        return response.content
    
    def complete_full(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Full completion with all options.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences
            **kwargs: Additional options
            
        Returns:
            LLMResponse with content and metadata
        """
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        # System prompt
        if system:
            request_kwargs["system"] = system
        
        # Temperature
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        else:
            request_kwargs["temperature"] = self.config.temperature
        
        # Top P
        if self.config.top_p != 1.0:
            request_kwargs["top_p"] = self.config.top_p
        
        # Stop sequences
        if stop_sequences:
            request_kwargs["stop_sequences"] = stop_sequences
        elif self.config.stop_sequences:
            request_kwargs["stop_sequences"] = self.config.stop_sequences
        
        # Metadata
        if self.config.metadata:
            request_kwargs["metadata"] = self.config.metadata
        
        # Extra body parameters
        if self.config.extra_body:
            request_kwargs.update(self.config.extra_body)
        
        # Override with kwargs
        request_kwargs.update(kwargs)
        
        # Make request
        start_time = time.time()
        try:
            response = self.client.messages.create(**request_kwargs)
        except Exception as e:
            self._handle_error(e)
            raise  # Re-raise if _handle_error doesn't
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        
        # Build usage
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        usage = self._calculate_cost(usage)
        self._update_usage(usage)
        
        return LLMResponse(
            content=content,
            finish_reason=response.stop_reason or "end_turn",
            usage=usage,
            model=response.model,
            raw_response=response,
            latency_ms=latency_ms,
        )
    
    def complete_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        callback: Optional[StreamingCallback] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream completion tokens.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Max tokens to generate
            callback: Optional callback for each chunk
            **kwargs: Additional options
            
        Yields:
            Text chunks as they are generated
        """
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if system:
            request_kwargs["system"] = system
        
        request_kwargs["temperature"] = self.config.temperature
        request_kwargs.update(kwargs)
        
        try:
            with self.client.messages.stream(**request_kwargs) as stream:
                for text in stream.text_stream:
                    if callback:
                        callback(text)
                    yield text
        except Exception as e:
            self._handle_error(e)
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Note: Anthropic's tokenizer is not publicly available.
        This uses the API's count_tokens endpoint when available,
        or falls back to a rough estimate.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            # Use the client's token counting if available
            if hasattr(self.client, "count_tokens"):
                result = self.client.count_tokens(text)
                return result.count
        except Exception:
            pass
        
        # Fallback: rough estimate (Claude uses ~4 chars per token on average)
        return len(text) // 4
    
    def _handle_error(self, error: Exception) -> None:
        """Handle Anthropic API errors."""
        try:
            from anthropic import (
                RateLimitError as AnthropicRateLimitError,
                AuthenticationError as AnthropicAuthError,
                BadRequestError as AnthropicBadRequestError,
            )
            
            if isinstance(error, AnthropicRateLimitError):
                raise RateLimitError(str(error), cause=error)
            elif isinstance(error, AnthropicAuthError):
                raise AuthenticationError(str(error), cause=error)
            elif isinstance(error, AnthropicBadRequestError):
                raise InvalidRequestError(str(error), cause=error)
        except ImportError:
            pass
        
        raise LLMError(str(error), cause=error)


class AsyncAnthropicProvider:
    """Async Anthropic provider for high-concurrency applications.
    
    Example:
        >>> provider = AsyncAnthropicProvider(AnthropicConfig(model="claude-3-5-sonnet-20241022"))
        >>> response = await provider.complete("Hello!")
    """
    
    def __init__(self, config: Optional[AnthropicConfig] = None):
        """Initialize async Anthropic provider."""
        if config is None:
            config = AnthropicConfig()
        
        self.config = config
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self._api_key:
            raise AuthenticationError("Anthropic API key not provided.")
        
        self._client = None
        self._sync_provider = AnthropicProvider(config)
    
    @property
    def client(self):
        """Get or create the async Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
            
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            
            self._client = AsyncAnthropic(**kwargs)
        
        return self._client
    
    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt."""
        response = await self.complete_full(prompt=prompt, max_tokens=max_tokens)
        return response.content
    
    async def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500,
    ) -> str:
        """Generate completion with system prompt."""
        response = await self.complete_full(prompt=prompt, system=system, max_tokens=max_tokens)
        return response.content
    
    async def complete_full(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """Full async completion."""
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if system:
            request_kwargs["system"] = system
        
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        else:
            request_kwargs["temperature"] = self.config.temperature
        
        request_kwargs.update(kwargs)
        
        start_time = time.time()
        response = await self.client.messages.create(**request_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        
        return LLMResponse(
            content=content,
            finish_reason=response.stop_reason or "end_turn",
            usage=usage,
            model=response.model,
            latency_ms=latency_ms,
        )
