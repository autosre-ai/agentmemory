"""Context Injection - Auto-inject relevant memories into conversation context.

Provides automatic retrieval and injection of relevant memories into
the conversation context, making information available to the agent
without explicit tool calls.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class InjectionConfig:
    """Configuration for context injection."""
    
    # Whether to enable auto-injection
    enabled: bool = True
    
    # Maximum memories to inject per turn
    max_memories: int = 5
    
    # Minimum relevance score (0.0-1.0)
    min_score: float = 0.3
    
    # Maximum tokens for injected context
    max_tokens: int = 500
    
    # Injection frequency: 'every-turn', 'first-turn', 'on-demand'
    frequency: str = "every-turn"
    
    # Whether to include confidence scores
    show_confidence: bool = False
    
    # Whether to include source information
    show_source: bool = False
    
    # Format template for injected memories
    format_template: str = "- {content}"


class ContextInjector:
    """Auto-inject relevant memories into conversation context.
    
    This class manages background retrieval of relevant memories and
    formats them for injection into the conversation context. It can
    operate in different modes:
    
    - 'every-turn': Inject relevant memories before each turn
    - 'first-turn': Only inject on the first turn of a session
    - 'on-demand': Only inject when explicitly requested
    
    Example:
        >>> from agent_memory_toolkit.hermes_plugin.context_injection import ContextInjector
        >>> from agent_memory_toolkit import MemoryStore
        >>> 
        >>> store = MemoryStore("memories.db")
        >>> injector = ContextInjector(store)
        >>> 
        >>> # Queue a prefetch for the next turn
        >>> injector.queue_prefetch("What are the user's preferences?")
        >>> 
        >>> # Get the prefetched context
        >>> context = injector.get_context()
        >>> print(context)
        ## Relevant Memories
        - User prefers dark mode
        - User works with Python primarily
    """
    
    def __init__(
        self,
        store: Any,  # MemoryStore
        config: Optional[InjectionConfig] = None,
    ):
        """Initialize the context injector.
        
        Args:
            store: MemoryStore instance for retrieving memories
            config: Injection configuration
        """
        self._store = store
        self._config = config or InjectionConfig()
        
        # Prefetch state
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        
        # Turn tracking
        self._turn_count = 0
        self._last_query: str = ""
    
    @property
    def config(self) -> InjectionConfig:
        """Get the current configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def queue_prefetch(self, query: str) -> None:
        """Queue a background prefetch for the next turn.
        
        Args:
            query: The query to use for memory retrieval
        """
        if not self._config.enabled:
            return
        
        # Check frequency setting
        if self._config.frequency == "first-turn" and self._turn_count > 0:
            return
        if self._config.frequency == "on-demand":
            return
        
        self._last_query = query
        
        def _run():
            try:
                self._do_prefetch(query)
            except Exception as e:
                logger.debug(f"Context injection prefetch failed: {e}")
        
        # Cancel any existing prefetch
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0.5)
        
        self._prefetch_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="context-injection-prefetch"
        )
        self._prefetch_thread.start()
    
    def _do_prefetch(self, query: str) -> None:
        """Execute the prefetch operation."""
        if not self._store:
            return
        
        # Search for relevant memories
        results = self._store.search(
            query,
            limit=self._config.max_memories * 2,  # Get extra for filtering
        )
        
        if not results:
            with self._prefetch_lock:
                self._prefetch_result = ""
            return
        
        # Filter by minimum score
        filtered = [
            r for r in results
            if r.score >= self._config.min_score
        ][:self._config.max_memories]
        
        if not filtered:
            with self._prefetch_lock:
                self._prefetch_result = ""
            return
        
        # Format the memories
        formatted = self._format_memories(filtered)
        
        with self._prefetch_lock:
            self._prefetch_result = formatted
    
    def _format_memories(self, results: List[Any]) -> str:
        """Format retrieved memories for injection."""
        lines = []
        
        for r in results:
            content = r.memory.content
            
            # Apply template
            line = self._config.format_template.format(
                content=content,
                score=f"{r.score:.2f}",
                confidence=r.memory.metadata.confidence if hasattr(r.memory.metadata, 'confidence') else 1.0,
            )
            
            # Add confidence if configured
            if self._config.show_confidence:
                confidence = getattr(r.memory.metadata, 'confidence', 1.0)
                line += f" (confidence: {confidence:.0%})"
            
            # Add source if configured
            if self._config.show_source:
                source = getattr(r.memory.metadata, 'source', None)
                if source:
                    line += f" [source: {source}]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_context(self, wait: bool = True, timeout: float = 3.0) -> str:
        """Get the prefetched context.
        
        Args:
            wait: Whether to wait for prefetch to complete
            timeout: Maximum time to wait in seconds
            
        Returns:
            Formatted context string, or empty string if no relevant memories
        """
        if wait and self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=timeout)
        
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        
        if not result:
            return ""
        
        self._turn_count += 1
        return f"## Relevant Memories\n{result}"
    
    def get_context_on_demand(self, query: str) -> str:
        """Synchronously retrieve and format relevant context.
        
        Use this for on-demand retrieval instead of background prefetch.
        
        Args:
            query: The query to use for memory retrieval
            
        Returns:
            Formatted context string
        """
        if not self._store:
            return ""
        
        try:
            results = self._store.search(
                query,
                limit=self._config.max_memories,
            )
            
            # Filter by minimum score
            filtered = [
                r for r in results
                if r.score >= self._config.min_score
            ]
            
            if not filtered:
                return ""
            
            formatted = self._format_memories(filtered)
            return f"## Relevant Memories\n{formatted}"
            
        except Exception as e:
            logger.debug(f"On-demand context retrieval failed: {e}")
            return ""
    
    def reset(self) -> None:
        """Reset the injector state (e.g., for new session)."""
        with self._prefetch_lock:
            self._prefetch_result = ""
        self._turn_count = 0
        self._last_query = ""


class SmartContextInjector(ContextInjector):
    """Enhanced context injector with smart query generation.
    
    Automatically analyzes the user message to generate effective
    memory retrieval queries.
    """
    
    # Keywords that indicate memory-relevant queries
    MEMORY_KEYWORDS = [
        "remember", "recall", "mentioned", "said", "told", "prefer",
        "like", "want", "need", "usually", "always", "never",
        "favorite", "last time", "before", "previously",
    ]
    
    def analyze_message(self, message: str) -> Tuple[str, float]:
        """Analyze a message to determine if memory retrieval is helpful.
        
        Args:
            message: The user message
            
        Returns:
            Tuple of (optimized query, relevance score)
        """
        # Check for memory-related keywords
        message_lower = message.lower()
        keyword_count = sum(1 for kw in self.MEMORY_KEYWORDS if kw in message_lower)
        
        # Calculate relevance score
        relevance = min(keyword_count * 0.2 + 0.3, 1.0)
        
        # Generate optimized query
        # Remove filler words and focus on key concepts
        query = self._extract_key_concepts(message)
        
        return query, relevance
    
    def _extract_key_concepts(self, message: str) -> str:
        """Extract key concepts from a message for querying."""
        # Simple extraction - remove common words
        filler_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "although", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them", "my", "your",
            "his", "its", "our", "their", "this", "that", "these", "those",
            "what", "which", "who", "whom", "whose", "please", "thanks",
            "thank", "hi", "hello", "hey",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', message.lower())
        key_words = [w for w in words if w not in filler_words and len(w) > 2]
        
        # Take most significant words
        return " ".join(key_words[:10])
    
    def smart_prefetch(self, message: str) -> None:
        """Analyze message and queue prefetch if relevant.
        
        Args:
            message: The user message
        """
        query, relevance = self.analyze_message(message)
        
        # Only prefetch if relevance is high enough
        if relevance >= 0.3 and query.strip():
            self.queue_prefetch(query)


__all__ = [
    "InjectionConfig",
    "ContextInjector",
    "SmartContextInjector",
]
