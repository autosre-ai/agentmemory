"""LangChain LLM Integration.

Provides universal adapter for any LangChain-compatible LLM or chat model.

Features:
- Works with any LangChain LLM or ChatModel
- Automatic adaptation between completion and chat APIs
- Memory-aware prompting
- Token counting integration
- Seamless fallback between sync and async

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from agent_memory_toolkit.llm import LangChainProvider
    >>> 
    >>> chat = ChatOpenAI(model="gpt-4o", temperature=0.7)
    >>> provider = LangChainProvider(chat)
    >>> 
    >>> response = provider.complete("What is machine learning?")
    >>> print(response)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, TYPE_CHECKING, Union

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    TokenUsage,
    ModelCapabilities,
    StreamingCallback,
)


# Type definitions for LangChain models
class LangChainLLM(Protocol):
    """Protocol for LangChain LLM models."""
    
    def invoke(self, input: str, **kwargs) -> str:
        """Invoke the model."""
        ...
    
    def get_num_tokens(self, text: str) -> int:
        """Get token count."""
        ...


class LangChainChatModel(Protocol):
    """Protocol for LangChain chat models."""
    
    def invoke(self, messages: list, **kwargs) -> Any:
        """Invoke the chat model."""
        ...
    
    def get_num_tokens_from_messages(self, messages: list) -> int:
        """Get token count from messages."""
        ...


@dataclass
class LangChainConfig(LLMConfig):
    """Configuration for LangChain adapter.
    
    Attributes:
        model_name: Name of the underlying model (for logging/tracking)
        
    Example:
        >>> config = LangChainConfig(
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ... )
    """
    model_name: Optional[str] = None


class LangChainProvider(BaseLLMProvider):
    """Universal LangChain model adapter.
    
    Works with any LangChain LLM or ChatModel, automatically detecting
    the model type and adapting the API calls accordingly.
    
    Example:
        >>> # With ChatOpenAI
        >>> from langchain_openai import ChatOpenAI
        >>> chat = ChatOpenAI(model="gpt-4o")
        >>> provider = LangChainProvider(chat)
        >>> 
        >>> # With any LLM
        >>> from langchain_anthropic import ChatAnthropic
        >>> claude = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> provider = LangChainProvider(claude)
        >>> 
        >>> # Use with memory toolkit
        >>> response = provider.complete("Hello, world!")
        >>> response = provider.complete_with_system(
        ...     system="You are helpful.",
        ...     prompt="What is Python?",
        ... )
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[LangChainConfig] = None,
    ):
        """Initialize LangChain adapter.
        
        Args:
            model: LangChain LLM or ChatModel instance
            config: Optional configuration
        """
        if config is None:
            config = LangChainConfig()
        
        super().__init__(config)
        self.config: LangChainConfig = config
        self._model = model
        
        # Detect model type
        self._is_chat_model = self._detect_chat_model(model)
        
        # Try to extract model name
        self._model_name = self._extract_model_name(model)
    
    def _detect_chat_model(self, model: Any) -> bool:
        """Detect if model is a chat model."""
        # Check for chat model indicators
        if hasattr(model, "invoke"):
            # Check method signature or model type
            model_class = type(model).__name__.lower()
            if "chat" in model_class:
                return True
            
            # Check for messages in invoke signature
            try:
                import inspect
                sig = inspect.signature(model.invoke)
                params = list(sig.parameters.keys())
                if "messages" in params or "input" in params:
                    return True
            except Exception:
                pass
        
        # Check for known LangChain chat model base classes
        try:
            from langchain_core.language_models import BaseChatModel
            if isinstance(model, BaseChatModel):
                return True
        except ImportError:
            pass
        
        return False
    
    def _extract_model_name(self, model: Any) -> str:
        """Extract model name from LangChain model."""
        # Try various attributes
        for attr in ["model_name", "model", "model_id", "_model_name"]:
            if hasattr(model, attr):
                value = getattr(model, attr)
                if isinstance(value, str) and value:
                    return value
        
        # Fall back to class name
        return type(model).__name__
    
    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Get capabilities of the current model."""
        # Try to get from model config
        caps = ModelCapabilities()
        
        try:
            if hasattr(self._model, "model_kwargs"):
                kwargs = self._model.model_kwargs
                if "max_tokens" in kwargs:
                    caps.max_tokens = kwargs["max_tokens"]
        except Exception:
            pass
        
        caps.supports_system_prompt = self._is_chat_model
        caps.supports_streaming = hasattr(self._model, "stream")
        
        return caps
    
    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> list:
        """Build LangChain messages."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            return messages
        except ImportError:
            # Fallback: return dict-based messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return messages
    
    def _extract_content(self, response: Any) -> str:
        """Extract text content from LangChain response."""
        # Handle AIMessage
        if hasattr(response, "content"):
            return str(response.content)
        
        # Handle string response
        if isinstance(response, str):
            return response
        
        # Handle dict response
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        
        # Fallback
        return str(response)
    
    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from response if available."""
        usage = TokenUsage()
        
        # Try to get usage from response metadata
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                if "usage" in metadata:
                    u = metadata["usage"]
                    usage.prompt_tokens = u.get("prompt_tokens", u.get("input_tokens", 0))
                    usage.completion_tokens = u.get("completion_tokens", u.get("output_tokens", 0))
                    usage.total_tokens = u.get("total_tokens", 
                        usage.prompt_tokens + usage.completion_tokens)
                elif "token_usage" in metadata:
                    u = metadata["token_usage"]
                    usage.prompt_tokens = u.get("prompt_tokens", 0)
                    usage.completion_tokens = u.get("completion_tokens", 0)
                    usage.total_tokens = u.get("total_tokens", 
                        usage.prompt_tokens + usage.completion_tokens)
        
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
        # Build invoke kwargs
        invoke_kwargs: dict[str, Any] = {}
        
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens
        
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        
        if stop_sequences:
            invoke_kwargs["stop"] = stop_sequences
        
        invoke_kwargs.update(kwargs)
        
        start_time = time.time()
        
        try:
            if self._is_chat_model:
                # Chat model: use messages
                messages = self._build_messages(prompt, system)
                response = self._model.invoke(messages, **invoke_kwargs)
            else:
                # LLM: use prompt directly (prepend system if present)
                full_prompt = prompt
                if system:
                    full_prompt = f"{system}\n\n{prompt}"
                response = self._model.invoke(full_prompt, **invoke_kwargs)
        except Exception as e:
            raise LLMError(f"LangChain invocation failed: {e}", cause=e)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract content and usage
        content = self._extract_content(response)
        usage = self._extract_usage(response)
        self._update_usage(usage)
        
        # Determine finish reason
        finish_reason = "stop"
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if isinstance(meta, dict):
                finish_reason = meta.get("finish_reason", 
                    meta.get("stop_reason", "stop"))
        
        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            model=self._model_name,
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
        if not hasattr(self._model, "stream"):
            # Fallback to non-streaming
            response = self.complete_full(prompt=prompt, system=system, max_tokens=max_tokens, **kwargs)
            if callback:
                callback(response.content)
            yield response.content
            return
        
        invoke_kwargs: dict[str, Any] = {}
        if max_tokens:
            invoke_kwargs["max_tokens"] = max_tokens
        invoke_kwargs.update(kwargs)
        
        try:
            if self._is_chat_model:
                messages = self._build_messages(prompt, system)
                stream = self._model.stream(messages, **invoke_kwargs)
            else:
                full_prompt = prompt if not system else f"{system}\n\n{prompt}"
                stream = self._model.stream(full_prompt, **invoke_kwargs)
            
            for chunk in stream:
                content = self._extract_content(chunk)
                if content:
                    if callback:
                        callback(content)
                    yield content
        except Exception as e:
            raise LLMError(f"LangChain streaming failed: {e}", cause=e)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Uses the LangChain model's token counter if available.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        # Try model's built-in token counter
        if hasattr(self._model, "get_num_tokens"):
            try:
                return self._model.get_num_tokens(text)
            except Exception:
                pass
        
        # Fallback to rough estimate
        return len(text) // 4


class LangChainAdapter(LangChainProvider):
    """Alias for LangChainProvider for backwards compatibility."""
    pass


class ChatModelAdapter(LangChainProvider):
    """Adapter specifically for LangChain chat models.
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> chat = ChatOpenAI(model="gpt-4o")
        >>> adapter = ChatModelAdapter(chat)
    """
    
    def __init__(self, chat_model: Any, config: Optional[LangChainConfig] = None):
        """Initialize chat model adapter.
        
        Args:
            chat_model: LangChain ChatModel instance
            config: Optional configuration
        """
        super().__init__(chat_model, config)
        self._is_chat_model = True  # Force chat model mode


class LLMAdapter(LangChainProvider):
    """Adapter specifically for LangChain LLM models (non-chat).
    
    Example:
        >>> from langchain_openai import OpenAI
        >>> llm = OpenAI(model="gpt-3.5-turbo-instruct")
        >>> adapter = LLMAdapter(llm)
    """
    
    def __init__(self, llm: Any, config: Optional[LangChainConfig] = None):
        """Initialize LLM adapter.
        
        Args:
            llm: LangChain LLM instance
            config: Optional configuration
        """
        super().__init__(llm, config)
        self._is_chat_model = False  # Force LLM mode
