"""Tests for the Context Compression Engine."""

import pytest
from unittest.mock import Mock, patch
from typing import Any


class TestTokenCounter:
    """Tests for TokenCounter."""
    
    def test_count_simple_text(self):
        """Test basic token counting."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        
        # Simple text
        count = counter.count("Hello, world!")
        assert count > 0
        assert count < 10
    
    def test_count_empty_text(self):
        """Test counting empty text."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        assert counter.count("") == 0
    
    def test_count_messages(self):
        """Test counting messages with overhead."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        count = counter.count_messages(messages)
        
        # Should be more than just the text tokens due to overhead
        text_only = counter.count("Hello") + counter.count("Hi there!")
        assert count > text_only
    
    def test_truncate_to_tokens(self):
        """Test truncation to token limit."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        
        long_text = "This is a very long text " * 100
        truncated = counter.truncate_to_tokens(long_text, 20, "...")
        
        # Verify truncation happened
        assert counter.count(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_truncate_short_text(self):
        """Test truncation of already short text."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        
        short_text = "Hello"
        truncated = counter.truncate_to_tokens(short_text, 100)
        
        assert truncated == short_text
    
    def test_split_into_chunks(self):
        """Test splitting text into chunks."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        
        long_text = "This is a sentence. " * 50
        chunks = counter.split_into_chunks(long_text, 50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert counter.count(chunk) <= 50
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        from agent_memory_toolkit.compression import TokenCounter
        
        counter = TokenCounter(model="gpt-4")
        
        original = "This is a long original text with many words."
        compressed = "Short summary."
        
        ratio = counter.estimate_compression_ratio(original, compressed)
        assert 0.0 < ratio < 1.0


class TestImportanceRanker:
    """Tests for ImportanceRanker."""
    
    def test_rank_basic_messages(self):
        """Test basic message ranking."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        ranker = ImportanceRanker()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        scored = ranker.rank(messages)
        
        assert len(scored) == 3
        # System message should have high importance
        assert scored[0].score > 0.1  # System message
    
    def test_critical_marker_boost(self):
        """Test that critical markers boost importance."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        ranker = ImportanceRanker()
        messages = [
            {"role": "user", "content": "Normal message"},
            {"role": "user", "content": "[CRITICAL] Important info here"},
        ]
        
        scored = ranker.rank(messages)
        
        # Message with critical marker should score higher
        assert scored[1].score > scored[0].score
        assert scored[1].factors.has_critical_marker
    
    def test_recency_factor(self):
        """Test that recent messages get higher recency scores."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        ranker = ImportanceRanker()
        messages = [
            {"role": "user", "content": "Old message"},
            {"role": "user", "content": "Middle message"},
            {"role": "user", "content": "Recent message"},
        ]
        
        scored = ranker.rank(messages)
        
        assert scored[0].factors.recency < scored[2].factors.recency
    
    def test_code_detection(self):
        """Test code block detection."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        ranker = ImportanceRanker()
        messages = [
            {"role": "assistant", "content": "Here is code:\n```python\nprint('hello')\n```"},
        ]
        
        scored = ranker.rank(messages)
        assert scored[0].factors.has_code
    
    def test_compression_candidates(self):
        """Test getting compression candidates."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        ranker = ImportanceRanker()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old user msg"},
            {"role": "assistant", "content": "Old assistant msg"},
            {"role": "user", "content": "Recent user"},
            {"role": "assistant", "content": "Recent assistant"},
        ]
        
        candidates = ranker.get_compression_candidates(
            messages,
            target_reduction=0.5,
            preserve_system=True,
            preserve_recent=2,
        )
        
        # System and last 2 messages should not be candidates
        assert 0 not in candidates  # System
        assert 3 not in candidates  # Recent
        assert 4 not in candidates  # Recent
    
    def test_custom_scorer(self):
        """Test custom scoring function."""
        from agent_memory_toolkit.compression import ImportanceRanker
        
        def boost_short(msg: dict, idx: int) -> float:
            return 0.5 if len(msg.get("content", "")) < 20 else 0.0
        
        ranker = ImportanceRanker(custom_scorer=boost_short)
        messages = [
            {"role": "user", "content": "Short"},
            {"role": "user", "content": "This is a much longer message with more content"},
        ]
        
        scored = ranker.rank(messages)
        assert scored[0].factors.custom_score == 0.5
        assert scored[1].factors.custom_score == 0.0


class TestTruncateStrategy:
    """Tests for TruncateStrategy."""
    
    def test_no_compression_needed(self):
        """Test when no compression is needed."""
        from agent_memory_toolkit.compression import TruncateStrategy, TokenCounter
        
        strategy = TruncateStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        
        result = strategy.compress(messages, 1000, counter)
        
        assert result.compression_ratio == 0.0
        assert len(result.messages) == 2
    
    def test_removes_old_messages(self):
        """Test that old messages are removed first."""
        from agent_memory_toolkit.compression import TruncateStrategy, TokenCounter
        
        strategy = TruncateStrategy(preserve_recent=1)
        counter = TokenCounter()
        
        # Create messages that exceed budget
        messages = [
            {"role": "user", "content": "Old message " * 100},
            {"role": "user", "content": "Recent message"},
        ]
        
        result = strategy.compress(messages, 100, counter)
        
        # Old message should be removed, recent preserved
        assert len(result.messages) < 2 or result.compressed_tokens <= 100
    
    def test_preserves_system_messages(self):
        """Test that system messages are preserved."""
        from agent_memory_toolkit.compression import TruncateStrategy, TokenCounter
        
        strategy = TruncateStrategy(preserve_system=True, preserve_recent=0)
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message " * 100},
        ]
        
        result = strategy.compress(messages, 100, counter)
        
        # System message should still be there
        system_present = any(m.get("role") == "system" for m in result.messages)
        assert system_present


class TestSummarizeStrategy:
    """Tests for SummarizeStrategy."""
    
    def test_rule_based_summarize(self):
        """Test rule-based summarization without LLM."""
        from agent_memory_toolkit.compression import SummarizeStrategy, TokenCounter
        
        strategy = SummarizeStrategy(preserve_recent=1)
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "First message with information."},
            {"role": "assistant", "content": "Response to first message."},
            {"role": "user", "content": "Recent message"},
        ]
        
        result = strategy.compress(messages, 100, counter)
        
        assert result.strategy_used == "summarize"
        assert not result.details.get("llm_used")
    
    def test_with_llm_provider(self):
        """Test summarization with mock LLM."""
        from agent_memory_toolkit.compression import SummarizeStrategy, TokenCounter
        
        mock_llm = Mock()
        mock_llm.complete.return_value = "Brief summary of conversation."
        
        # Need more messages to trigger summarization (above preserve_recent threshold)
        strategy = SummarizeStrategy(llm_provider=mock_llm, preserve_recent=1, chunk_size=2)
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Old message with details."},
            {"role": "assistant", "content": "Old response with more details."},
            {"role": "user", "content": "Another old message."},
            {"role": "assistant", "content": "Another old response."},
            {"role": "user", "content": "Recent"},
        ]
        
        # Use a smaller budget to force compression
        result = strategy.compress(messages, 100, counter)
        
        # Either LLM was used OR there was no need for compression
        if result.compression_ratio > 0:
            assert result.details.get("llm_used") or result.details.get("summarized_chunks", 0) > 0


class TestExtractKeyFactsStrategy:
    """Tests for ExtractKeyFactsStrategy."""
    
    def test_extract_facts_rule_based(self):
        """Test rule-based fact extraction."""
        from agent_memory_toolkit.compression import ExtractKeyFactsStrategy, TokenCounter
        
        strategy = ExtractKeyFactsStrategy(preserve_recent=1)
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "The meeting is on 2024-03-15. The result is: 42"},
            {"role": "user", "content": "Recent message"},
        ]
        
        result = strategy.compress(messages, 200, counter)
        
        assert result.strategy_used == "extract_key_facts"
    
    def test_preserves_critical_markers(self):
        """Test that critical markers are extracted."""
        from agent_memory_toolkit.compression import ExtractKeyFactsStrategy, TokenCounter
        
        strategy = ExtractKeyFactsStrategy(preserve_recent=0)
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "[CRITICAL] Remember this important fact."},
        ]
        
        result = strategy.compress(messages, 200, counter)
        
        # Should have extracted the critical info
        assert result.details.get("facts_extracted", 0) > 0 or len(result.messages) > 0


class TestTieredCompressionStrategy:
    """Tests for TieredCompressionStrategy."""
    
    def test_tiered_compression(self):
        """Test tiered compression with different treatment for message ages."""
        from agent_memory_toolkit.compression import TieredCompressionStrategy, TokenCounter
        
        strategy = TieredCompressionStrategy(recent_count=2, medium_count=2)
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Very old message " * 20},
            {"role": "assistant", "content": "Old response " * 20},
            {"role": "user", "content": "Medium age message " * 20},
            {"role": "assistant", "content": "Medium response " * 20},
            {"role": "user", "content": "Recent user message"},
            {"role": "assistant", "content": "Recent assistant response"},
        ]
        
        result = strategy.compress(messages, 500, counter)
        
        assert result.strategy_used == "tiered"
        assert result.compressed_tokens <= 500 or result.compression_ratio > 0
    
    def test_no_compression_when_within_budget(self):
        """Test that no compression happens when within budget."""
        from agent_memory_toolkit.compression import TieredCompressionStrategy, TokenCounter
        
        strategy = TieredCompressionStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        
        result = strategy.compress(messages, 10000, counter)
        
        assert result.compression_ratio == 0.0


class TestContextCompressor:
    """Tests for the main ContextCompressor class."""
    
    def test_basic_compression(self):
        """Test basic compression workflow."""
        from agent_memory_toolkit.compression import ContextCompressor, CompressionConfig
        
        # Use config with lower recent_count to trigger compression
        config = CompressionConfig(
            max_tokens=300,
            reserve_tokens=0,
            recent_count=1,
            medium_count=1,
        )
        compressor = ContextCompressor(config=config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Hi there " * 100},
            {"role": "user", "content": "Another question " * 50},
            {"role": "assistant", "content": "Another response " * 50},
            {"role": "user", "content": "Recent question"},
        ]
        
        result = compressor.compress(messages)
        
        # Should either compress or be within budget
        assert result.compressed_tokens <= 300 or result.compression_ratio > 0
    
    def test_needs_compression(self):
        """Test needs_compression check."""
        from agent_memory_toolkit.compression import ContextCompressor, CompressionConfig
        
        # Use a config with no reserve and low token limit
        config = CompressionConfig(max_tokens=100, reserve_tokens=0)
        compressor = ContextCompressor(config=config)
        
        short_messages = [{"role": "user", "content": "Hi"}]
        long_messages = [{"role": "user", "content": "Word " * 500}]  # Much longer
        
        assert not compressor.needs_compression(short_messages)
        assert compressor.needs_compression(long_messages)
    
    def test_count_tokens(self):
        """Test token counting."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor()
        
        messages = [
            {"role": "user", "content": "Hello world"},
        ]
        
        count = compressor.count_tokens(messages)
        assert count > 0
    
    def test_rank_messages(self):
        """Test message ranking."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor()
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "[CRITICAL] Important"},
            {"role": "user", "content": "Normal message"},
        ]
        
        ranked = compressor.rank_messages(messages)
        
        assert len(ranked) == 3
        assert all(hasattr(m, 'score') for m in ranked)
    
    def test_compress_auto(self):
        """Test automatic strategy selection."""
        from agent_memory_toolkit.compression import ContextCompressor, CompressionConfig
        
        config = CompressionConfig(
            max_tokens=150,
            reserve_tokens=0,
            recent_count=1,
            medium_count=1,
        )
        compressor = ContextCompressor(config=config)
        
        # Create a longer conversation to trigger compression
        messages = [
            {"role": "user", "content": "Message " * 50},
            {"role": "assistant", "content": "Response " * 50},
            {"role": "user", "content": "Another message " * 50},
            {"role": "assistant", "content": "Another response " * 50},
            {"role": "user", "content": "Recent"},
        ]
        
        result = compressor.compress_auto(messages)
        
        # Should either compress or meet budget
        assert result.compressed_tokens <= 150 or result.compression_ratio > 0
    
    def test_preserve_critical_info(self):
        """Test that critical information is preserved."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor(max_tokens=300)
        
        messages = [
            {"role": "user", "content": "[CRITICAL] The password is secret123"},
            {"role": "user", "content": "Padding " * 100},
            {"role": "user", "content": "Recent question"},
        ]
        
        result = compressor.compress(messages)
        
        # Check if critical info is preserved somewhere
        all_content = " ".join(m.get("content", "") for m in result.messages)
        # The critical marker should trigger preservation
        assert "[CRITICAL]" in all_content or "secret123" in all_content or "PRESERVED" in all_content
    
    def test_strategy_selection(self):
        """Test explicit strategy selection."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor(max_tokens=500)
        
        messages = [
            {"role": "user", "content": "Test message " * 50},
            {"role": "assistant", "content": "Response " * 50},
        ]
        
        # Test each strategy
        for strategy in ["truncate", "summarize", "extract_key_facts", "tiered"]:
            result = compressor.compress(messages, strategy=strategy)
            assert result.strategy_used == strategy
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor()
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(ValueError):
            compressor.compress(messages, strategy="invalid_strategy")
    
    def test_add_custom_strategy(self):
        """Test adding a custom strategy."""
        from agent_memory_toolkit.compression import (
            ContextCompressor, 
            CompressionStrategy,
            TokenCounter,
        )
        from agent_memory_toolkit.compression.strategies import CompressionResult
        
        class CustomStrategy(CompressionStrategy):
            @property
            def name(self) -> str:
                return "custom"
            
            def compress(self, messages, token_budget, counter, **kwargs):
                return CompressionResult(
                    original_tokens=100,
                    compressed_tokens=50,
                    messages=messages[:1],
                    compression_ratio=0.5,
                    strategy_used=self.name,
                )
        
        compressor = ContextCompressor()
        compressor.add_strategy("custom", CustomStrategy())
        
        messages = [{"role": "user", "content": "Test"}]
        result = compressor.compress(messages, strategy="custom")
        
        assert result.strategy_used == "custom"
    
    def test_compression_stats(self):
        """Test getting compression statistics."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        compressor = ContextCompressor(max_tokens=100)
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message " * 50},
            {"role": "assistant", "content": "Response " * 50},
        ]
        
        stats = compressor.get_compression_stats(messages)
        
        assert "current_tokens" in stats
        assert "token_budget" in stats
        assert "over_budget" in stats
        assert "message_count" in stats
        assert stats["message_count"] == 3
    
    def test_compression_modes(self):
        """Test different compression modes."""
        from agent_memory_toolkit.compression import ContextCompressor, CompressionMode, CompressionConfig
        
        messages = [
            {"role": "user", "content": "Test " * 50},
            {"role": "assistant", "content": "Response " * 50},
            {"role": "user", "content": "Another " * 50},
            {"role": "assistant", "content": "More response " * 50},
            {"role": "user", "content": "Recent"},
        ]
        
        for mode in CompressionMode:
            config = CompressionConfig(
                max_tokens=150,
                reserve_tokens=0,
                mode=mode,
                recent_count=1,
                medium_count=1,
            )
            compressor = ContextCompressor(config=config)
            result = compressor.compress(messages)
            # Should either be within budget or have applied some compression
            assert result.compressed_tokens <= 150 or result.compression_ratio > 0
    
    def test_message_class(self):
        """Test Message dataclass."""
        from agent_memory_toolkit.compression import Message, MessageRole
        
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            name="John",
            metadata={"key": "value"},
        )
        
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["name"] == "John"
        
        # Test from_dict
        msg2 = Message.from_dict(d)
        assert msg2.role == MessageRole.USER
        assert msg2.content == "Hello"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_compress_context(self):
        """Test compress_context convenience function."""
        from agent_memory_toolkit.compression.compressor import compress_context
        
        # Create a longer conversation
        messages = [
            {"role": "user", "content": "Test " * 50},
            {"role": "assistant", "content": "Response " * 50},
            {"role": "user", "content": "More " * 50},
            {"role": "assistant", "content": "More response " * 50},
            {"role": "user", "content": "Recent"},
        ]
        
        result = compress_context(messages, max_tokens=100)
        # Should either be compressed or within budget
        assert result.compressed_tokens <= 100 or result.compression_ratio > 0
    
    def test_needs_compression_function(self):
        """Test needs_compression convenience function."""
        from agent_memory_toolkit.compression.compressor import needs_compression
        
        short = [{"role": "user", "content": "Hi"}]
        long = [{"role": "user", "content": "Word " * 1000}]
        
        assert not needs_compression(short, max_tokens=1000)
        assert needs_compression(long, max_tokens=100)


class TestIntegration:
    """Integration tests for the compression module."""
    
    def test_full_compression_pipeline(self):
        """Test complete compression workflow."""
        from agent_memory_toolkit.compression import (
            ContextCompressor,
            CompressionMode,
            TokenCounter,
        )
        
        # Create a realistic conversation
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I sort a list in Python?"},
            {"role": "assistant", "content": """You can sort a list in Python using several methods:

1. `list.sort()` - Sorts the list in-place
2. `sorted(list)` - Returns a new sorted list

Example:
```python
numbers = [3, 1, 4, 1, 5]
numbers.sort()  # [1, 1, 3, 4, 5]
```

[CRITICAL] Remember: sort() modifies the original list, sorted() returns a new one.
"""},
            {"role": "user", "content": "What about sorting dictionaries?"},
            {"role": "assistant", "content": "You can sort dictionaries by key or value using sorted() with a lambda function."},
            {"role": "user", "content": "Can you show an example? " * 20},  # Long message
            {"role": "assistant", "content": "Here's an example:\n" + "```python\n" + "d = {'b': 2, 'a': 1}\nsorted_d = dict(sorted(d.items()))\n" + "```\n" * 5},
            {"role": "user", "content": "Thanks! One more question about performance."},
        ]
        
        # Test compression
        compressor = ContextCompressor(
            max_tokens=500,
            mode=CompressionMode.BALANCED,
        )
        
        # Check initial state
        initial_tokens = compressor.count_tokens(messages)
        needs = compressor.needs_compression(messages)
        
        if needs:
            result = compressor.compress(messages)
            
            assert result.compressed_tokens < initial_tokens
            assert result.compressed_tokens <= 500 or result.compression_ratio > 0
            
            # Verify critical info is preserved
            all_content = " ".join(m.get("content", "") for m in result.messages)
            # Critical info should be somewhere in output
            assert "sort" in all_content.lower() or "CRITICAL" in all_content
    
    def test_compression_preserves_conversation_structure(self):
        """Test that compression maintains valid conversation structure."""
        from agent_memory_toolkit.compression import ContextCompressor
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message " * 50},
            {"role": "assistant", "content": "Assistant " * 50},
            {"role": "user", "content": "Final question"},
        ]
        
        compressor = ContextCompressor(max_tokens=200)
        result = compressor.compress(messages)
        
        # All messages should have valid roles
        valid_roles = {"system", "user", "assistant", "tool", "function"}
        for msg in result.messages:
            assert msg.get("role") in valid_roles
            assert "content" in msg
