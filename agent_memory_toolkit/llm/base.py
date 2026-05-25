"""Base classes and protocols for LLM integrations.

This module defines the core abstractions that all LLM providers implement,
ensuring a consistent interface across different backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Iterator


class CompletionMode(str, Enum):
    """LLM completion modes."""
    CHAT = "chat"           # Chat/conversation mode
    COMPLETION = "completion"  # Text completion mode
    INSTRUCT = "instruct"   # Instruction-following mode


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Optional cost tracking (in USD)
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None
    total_cost: Optional[float] = None
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two token usages together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_cost=(self.prompt_cost or 0) + (other.prompt_cost or 0) if self.prompt_cost or other.prompt_cost else None,
            completion_cost=(self.completion_cost or 0) + (other.completion_cost or 0) if self.completion_cost or other.completion_cost else None,
            total_cost=(self.total_cost or 0) + (other.total_cost or 0) if self.total_cost or other.total_cost else None,
        )


@dataclass
class ModelCapabilities:
    """Capabilities of an LLM model."""
    max_tokens: int = 4096
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    context_window: int = 4096
    
    # Pricing per 1M tokens (USD)
    input_price_per_million: Optional[float] = None
    output_price_per_million: Optional[float] = None


@dataclass
class LLMResponse:
    """Response from an LLM completion."""
    content: str
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: Optional[str] = None
    raw_response: Optional[Any] = None
    
    # Metadata
    latency_ms: Optional[float] = None
    cached: bool = False


@dataclass
class LLMConfig:
    """Base configuration for LLM providers."""
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    
    # Request options
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base exception for LLM errors."""
    
    def __init__(
        self, 
        message: str, 
        cause: Optional[Exception] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.cause = cause
        self.status_code = status_code
        self.retry_after = retry_after


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(LLMError):
    """Authentication failed."""
    pass


class InvalidRequestError(LLMError):
    """Invalid request parameters."""
    pass


class ModelNotFoundError(LLMError):
    """Model not found or not accessible."""
    pass


# Type for streaming callbacks
StreamingCallback = Callable[[str], None]


class LLMProvider(Protocol):
    """Protocol for LLM providers.
    
    This is the core interface that all LLM providers must implement.
    It's designed to be compatible with the existing protocol in
    agent_memory_toolkit.compression.summarization.
    """
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt.
        
        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        ...
    
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
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    Provides common functionality and implements the LLMProvider protocol.
    Concrete implementations should inherit from this class.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self._total_usage = TokenUsage()
    
    @property
    def total_usage(self) -> TokenUsage:
        """Get cumulative token usage."""
        return self._total_usage
    
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt."""
        ...
    
    @abstractmethod
    def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500,
    ) -> str:
        """Generate completion with system prompt."""
        ...
    
    @abstractmethod
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
            **kwargs: Provider-specific options
            
        Returns:
            LLMResponse with content and metadata
        """
        ...
    
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
            callback: Optional callback for each token
            **kwargs: Provider-specific options
            
        Yields:
            Tokens as they are generated
        """
        # Default implementation: non-streaming fallback
        response = self.complete_full(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            **kwargs,
        )
        if callback:
            callback(response.content)
        yield response.content
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        ...
    
    @property
    @abstractmethod
    def model_capabilities(self) -> ModelCapabilities:
        """Get capabilities of the current model."""
        ...
    
    def _update_usage(self, usage: TokenUsage) -> None:
        """Update cumulative usage."""
        self._total_usage = self._total_usage + usage
    
    def reset_usage(self) -> TokenUsage:
        """Reset usage counter and return final count."""
        final = self._total_usage
        self._total_usage = TokenUsage()
        return final


class AsyncLLMProvider(Protocol):
    """Protocol for async LLM providers."""
    
    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt."""
        ...
    
    async def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500,
    ) -> str:
        """Generate completion with system prompt."""
        ...
