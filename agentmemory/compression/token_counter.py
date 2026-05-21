"""Token counting utilities using tiktoken."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import tiktoken


class TokenCounter:
    """Count tokens using tiktoken encodings.
    
    Supports multiple models and encoding types with caching for performance.
    
    Example:
        >>> counter = TokenCounter(model="gpt-4")
        >>> counter.count("Hello, world!")
        4
        >>> counter.count_messages([{"role": "user", "content": "Hi"}])
        8
    """
    
    # Default tokens per message overhead for chat models
    # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    TOKENS_PER_MESSAGE = {
        "gpt-4": 3,
        "gpt-4o": 3,
        "gpt-4-turbo": 3,
        "gpt-4-32k": 3,
        "gpt-3.5-turbo": 4,
        "gpt-3.5-turbo-16k": 4,
        "claude-3": 3,  # Approximation
        "claude-3.5": 3,  # Approximation
    }
    
    TOKENS_PER_NAME = {
        "gpt-4": 1,
        "gpt-4o": 1,
        "gpt-4-turbo": 1,
        "gpt-3.5-turbo": -1,
    }
    
    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "claude-3": "cl100k_base",  # Approximation
        "claude-3.5": "cl100k_base",  # Approximation
    }
    
    def __init__(
        self,
        model: str = "gpt-4",
        encoding_name: Optional[str] = None,
    ):
        """Initialize token counter.
        
        Args:
            model: Model name for encoding selection
            encoding_name: Override encoding name (e.g., "cl100k_base")
        """
        self.model = model
        self._encoding_name = encoding_name or self._get_encoding_for_model(model)
        self._encoding: Optional[tiktoken.Encoding] = None
        
    def _get_encoding_for_model(self, model: str) -> str:
        """Get encoding name for a model."""
        # Check exact match
        if model in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model]
        
        # Check prefix match
        for prefix, encoding in self.MODEL_ENCODINGS.items():
            if model.startswith(prefix):
                return encoding
        
        # Default to cl100k_base
        return "cl100k_base"
    
    @property
    def encoding(self) -> "tiktoken.Encoding":
        """Lazy-load tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "tiktoken is required for token counting. "
                    "Install with: pip install tiktoken"
                )
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding
    
    def count(self, text: str) -> int:
        """Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_messages(
        self,
        messages: list[dict],
        include_reply_priming: bool = True,
    ) -> int:
        """Count tokens in a list of chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            include_reply_priming: Add tokens for assistant reply priming
            
        Returns:
            Total token count including message overhead
        """
        tokens_per_message = self.TOKENS_PER_MESSAGE.get(self.model, 3)
        tokens_per_name = self.TOKENS_PER_NAME.get(self.model, 1)
        
        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    total += self.count(value)
                if key == "name":
                    total += tokens_per_name
        
        # Every reply is primed with <|start|>assistant<|message|>
        if include_reply_priming:
            total += 3
            
        return total
    
    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        truncation_marker: str = "...",
    ) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            truncation_marker: String to append if truncated
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        # Reserve space for truncation marker
        marker_tokens = self.count(truncation_marker)
        available_tokens = max_tokens - marker_tokens
        
        if available_tokens <= 0:
            return truncation_marker[:max_tokens] if max_tokens > 0 else ""
            
        truncated_tokens = tokens[:available_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        return truncated_text + truncation_marker
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> list[str]:
        """Split text into chunks of approximately chunk_size tokens.
        
        Args:
            text: Text to split
            chunk_size: Target tokens per chunk
            overlap: Tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start = end - overlap if overlap > 0 else end
            
        return chunks
    
    def estimate_compression_ratio(
        self,
        original: str,
        compressed: str,
    ) -> float:
        """Calculate compression ratio.
        
        Args:
            original: Original text
            compressed: Compressed text
            
        Returns:
            Compression ratio (0.0 = no compression, 1.0 = full compression)
        """
        original_tokens = self.count(original)
        compressed_tokens = self.count(compressed)
        
        if original_tokens == 0:
            return 0.0
            
        return 1.0 - (compressed_tokens / original_tokens)


# Module-level convenience functions with caching
@functools.lru_cache(maxsize=8)
def get_counter(model: str = "gpt-4") -> TokenCounter:
    """Get a cached TokenCounter instance."""
    return TokenCounter(model=model)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using specified model encoding."""
    return get_counter(model).count(text)


def count_message_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """Count tokens in chat messages."""
    return get_counter(model).count_messages(messages)
