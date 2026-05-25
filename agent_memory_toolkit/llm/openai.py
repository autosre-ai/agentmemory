"""OpenAI LLM Integration.

Provides seamless integration with OpenAI's API for GPT-4, GPT-3.5, and other models.

Features:
- Full support for chat and completion APIs
- Streaming completions
- Token counting with tiktoken
- Cost tracking
- Automatic retries with exponential backoff
- Async support

Example:
    >>> from agent_memory_toolkit.llm import OpenAIProvider, OpenAIConfig
    >>> 
    >>> provider = OpenAIProvider(OpenAIConfig(
    ...     model=OpenAIModel.GPT_4O,
    ...     api_key="sk-...",
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
from typing import Any, Iterator, Optional, TYPE_CHECKING

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


class OpenAIModel(str, Enum):
    """Available OpenAI models."""
    # GPT-4 Omni
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    
    # GPT-4 Turbo
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    
    # GPT-4
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    
    # GPT-3.5
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"
    
    # O1 models (reasoning)
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"


# Model capabilities lookup
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "gpt-4o": ModelCapabilities(
        max_tokens=16384,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        context_window=128000,
        input_price_per_million=5.0,
        output_price_per_million=15.0,
    ),
    "gpt-4o-mini": ModelCapabilities(
        max_tokens=16384,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        context_window=128000,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
    "gpt-4-turbo": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        context_window=128000,
        input_price_per_million=10.0,
        output_price_per_million=30.0,
    ),
    "gpt-4": ModelCapabilities(
        max_tokens=8192,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        context_window=8192,
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    ),
    "gpt-4-32k": ModelCapabilities(
        max_tokens=32768,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        context_window=32768,
        input_price_per_million=60.0,
        output_price_per_million=120.0,
    ),
    "gpt-3.5-turbo": ModelCapabilities(
        max_tokens=4096,
        supports_system_prompt=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        context_window=16385,
        input_price_per_million=0.50,
        output_price_per_million=1.50,
    ),
    "o1-preview": ModelCapabilities(
        max_tokens=32768,
        supports_system_prompt=False,  # O1 doesn't use system prompts
        supports_streaming=False,       # O1 doesn't support streaming
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        context_window=128000,
        input_price_per_million=15.0,
        output_price_per_million=60.0,
    ),
    "o1-mini": ModelCapabilities(
        max_tokens=65536,
        supports_system_prompt=False,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        context_window=128000,
        input_price_per_million=3.0,
        output_price_per_million=12.0,
    ),
}


@dataclass
class OpenAIConfig(LLMConfig):
    """Configuration for OpenAI provider.
    
    Attributes:
        model: OpenAI model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        organization: OpenAI organization ID
        base_url: Custom API base URL (for Azure OpenAI or proxies)
        
    Example:
        >>> config = OpenAIConfig(
        ...     model=OpenAIModel.GPT_4O,
        ...     api_key="sk-...",
        ...     temperature=0.7,
        ... )
    """
    model: str = "gpt-4o"
    organization: Optional[str] = None
    
    # Azure OpenAI settings
    api_version: Optional[str] = None  # For Azure: "2024-02-01"
    deployment_name: Optional[str] = None  # For Azure deployments
    
    # Response format
    response_format: Optional[dict] = None  # {"type": "json_object"}
    
    # Seed for reproducibility
    seed: Optional[int] = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider.
    
    Provides integration with OpenAI's chat and completion APIs.
    Supports GPT-4, GPT-3.5, and O1 models.
    
    Example:
        >>> provider = OpenAIProvider(OpenAIConfig(model="gpt-4o"))
        >>> response = provider.complete("Hello, world!")
        >>> print(response)
        
        >>> # With system prompt
        >>> response = provider.complete_with_system(
        ...     system="You are a helpful assistant.",
        ...     prompt="What is Python?",
        ... )
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize OpenAI provider.
        
        Args:
            config: Provider configuration. If None, uses defaults with
                    OPENAI_API_KEY environment variable.
        """
        if config is None:
            config = OpenAIConfig()
        
        super().__init__(config)
        self.config: OpenAIConfig = config
        
        # Get API key from config or environment
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise AuthenticationError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key in config.")
        
        # Lazy-load the OpenAI client
        self._client = None
        self._tokenizer = None
    
    @property
    def client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI integration. "
                    "Install with: pip install openai"
                )
            
            kwargs = {
                "api_key": self._api_key,
            }
            
            if self.config.organization:
                kwargs["organization"] = self.config.organization
            
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            
            if self.config.timeout:
                kwargs["timeout"] = self.config.timeout
            
            if self.config.max_retries:
                kwargs["max_retries"] = self.config.max_retries
            
            self._client = OpenAI(**kwargs)
        
        return self._client
    
    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Get capabilities of the current model."""
        model = self.config.model
        
        # Check exact match first
        if model in MODEL_CAPABILITIES:
            return MODEL_CAPABILITIES[model]
        
        # Check prefix matches for versioned models
        for base_model, caps in MODEL_CAPABILITIES.items():
            if model.startswith(base_model):
                return caps
        
        # Default capabilities
        return ModelCapabilities(
            max_tokens=4096,
            supports_system_prompt=True,
            supports_streaming=True,
            context_window=8192,
        )
    
    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build messages list for chat completion."""
        messages = []
        
        caps = self.model_capabilities
        
        # Add system message if supported
        if system and caps.supports_system_prompt:
            messages.append({"role": "system", "content": system})
        elif system:
            # For models like O1 that don't support system prompts,
            # prepend to user message
            prompt = f"{system}\n\n{prompt}"
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
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
            **kwargs: Additional options (response_format, seed, etc.)
            
        Returns:
            LLMResponse with content and metadata
        """
        messages = self._build_messages(prompt, system)
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        
        # O1 models don't support certain parameters
        caps = self.model_capabilities
        is_o1 = self.config.model.startswith("o1")
        
        if max_tokens is not None:
            if is_o1:
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            if is_o1:
                request_kwargs["max_completion_tokens"] = self.config.max_tokens
            else:
                request_kwargs["max_tokens"] = self.config.max_tokens
        
        # These parameters are not supported by O1 models
        if not is_o1:
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            else:
                request_kwargs["temperature"] = self.config.temperature
            
            if self.config.top_p != 1.0:
                request_kwargs["top_p"] = self.config.top_p
            
            if self.config.frequency_penalty != 0.0:
                request_kwargs["frequency_penalty"] = self.config.frequency_penalty
            
            if self.config.presence_penalty != 0.0:
                request_kwargs["presence_penalty"] = self.config.presence_penalty
        
        if stop_sequences:
            request_kwargs["stop"] = stop_sequences
        elif self.config.stop_sequences:
            request_kwargs["stop"] = self.config.stop_sequences
        
        # Response format for JSON mode
        if self.config.response_format and caps.supports_json_mode:
            request_kwargs["response_format"] = self.config.response_format
        
        # Seed for reproducibility
        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed
        
        # Extra body parameters
        if self.config.extra_body:
            request_kwargs.update(self.config.extra_body)
        
        # Override with kwargs
        request_kwargs.update(kwargs)
        
        # Make request
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            self._handle_error(e)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Build usage
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        usage = self._calculate_cost(usage)
        self._update_usage(usage)
        
        return LLMResponse(
            content=content,
            finish_reason=choice.finish_reason or "stop",
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
        # O1 models don't support streaming
        if self.config.model.startswith("o1"):
            response = self.complete_full(prompt=prompt, system=system, max_tokens=max_tokens, **kwargs)
            if callback:
                callback(response.content)
            yield response.content
            return
        
        messages = self._build_messages(prompt, system)
        
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }
        
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            request_kwargs["max_tokens"] = self.config.max_tokens
        
        request_kwargs["temperature"] = self.config.temperature
        request_kwargs.update(kwargs)
        
        try:
            stream = self.client.chat.completions.create(**request_kwargs)
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if callback:
                        callback(content)
                    yield content
        except Exception as e:
            self._handle_error(e)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if self._tokenizer is None:
            try:
                import tiktoken
            except ImportError:
                # Fallback to rough estimate
                return len(text) // 4
            
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.config.model)
            except KeyError:
                # Use cl100k_base for unknown models (GPT-4 family)
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
        return len(self._tokenizer.encode(text))
    
    def _handle_error(self, error: Exception) -> None:
        """Handle OpenAI API errors."""
        try:
            from openai import (
                RateLimitError as OpenAIRateLimitError,
                AuthenticationError as OpenAIAuthError,
                BadRequestError as OpenAIBadRequestError,
                NotFoundError as OpenAINotFoundError,
            )
            
            if isinstance(error, OpenAIRateLimitError):
                raise RateLimitError(str(error), cause=error)
            elif isinstance(error, OpenAIAuthError):
                raise AuthenticationError(str(error), cause=error)
            elif isinstance(error, OpenAIBadRequestError):
                raise InvalidRequestError(str(error), cause=error)
            elif isinstance(error, OpenAINotFoundError):
                raise LLMError(f"Model not found: {self.config.model}", cause=error)
        except ImportError:
            pass
        
        raise LLMError(str(error), cause=error)


class AsyncOpenAIProvider:
    """Async OpenAI provider for high-concurrency applications.
    
    Example:
        >>> provider = AsyncOpenAIProvider(OpenAIConfig(model="gpt-4o"))
        >>> response = await provider.complete("Hello!")
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize async OpenAI provider."""
        if config is None:
            config = OpenAIConfig()
        
        self.config = config
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self._api_key:
            raise AuthenticationError("OpenAI API key not provided.")
        
        self._client = None
        self._sync_provider = OpenAIProvider(config)
    
    @property
    def client(self):
        """Get or create the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            
            kwargs = {"api_key": self._api_key}
            
            if self.config.organization:
                kwargs["organization"] = self.config.organization
            
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            
            self._client = AsyncOpenAI(**kwargs)
        
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
        messages = self._sync_provider._build_messages(prompt, system)
        
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        
        is_o1 = self.config.model.startswith("o1")
        
        if max_tokens:
            key = "max_completion_tokens" if is_o1 else "max_tokens"
            request_kwargs[key] = max_tokens
        
        if not is_o1:
            request_kwargs["temperature"] = temperature or self.config.temperature
        
        request_kwargs.update(kwargs)
        
        start_time = time.time()
        response = await self.client.chat.completions.create(**request_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        
        return LLMResponse(
            content=content,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=response.model,
            latency_ms=latency_ms,
        )
