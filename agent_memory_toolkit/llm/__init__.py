"""LLM Framework Integrations - Connect Agent Memory Toolkit to LLM providers.

This module provides seamless integrations with popular LLM frameworks and providers,
enabling AI-powered memory operations like summarization, extraction, and reasoning.

Supported Providers:
- OpenAI: GPT-4, GPT-3.5, and other OpenAI models
- Anthropic: Claude 3, Claude 2, and other Anthropic models
- LangChain: Universal adapter for any LangChain-compatible model

Features:
- Unified interface across all providers
- Automatic retries and error handling
- Token counting and cost tracking
- Streaming support
- Memory-optimized prompting
"""

from .base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    TokenUsage,
    CompletionMode,
    StreamingCallback,
    ModelCapabilities,
)

from .openai import (
    OpenAIProvider,
    OpenAIConfig,
    OpenAIModel,
    AsyncOpenAIProvider,
)

from .anthropic import (
    AnthropicProvider,
    AnthropicConfig,
    AnthropicModel,
    AsyncAnthropicProvider,
)

from .langchain import (
    LangChainProvider,
    LangChainConfig,
    LangChainAdapter,
    ChatModelAdapter,
    LLMAdapter,
)

__all__ = [
    # Base classes and types
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMError",
    "TokenUsage",
    "CompletionMode",
    "StreamingCallback",
    "ModelCapabilities",
    # OpenAI
    "OpenAIProvider",
    "OpenAIConfig",
    "OpenAIModel",
    "AsyncOpenAIProvider",
    # Anthropic
    "AnthropicProvider",
    "AnthropicConfig",
    "AnthropicModel",
    "AsyncAnthropicProvider",
    # LangChain
    "LangChainProvider",
    "LangChainConfig",
    "LangChainAdapter",
    "ChatModelAdapter",
    "LLMAdapter",
]
