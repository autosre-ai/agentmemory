#!/usr/bin/env python3
"""
Context Compression Demo - Agent Memory Toolkit

Demonstrates the context compression engine for managing LLM token budgets
while preserving critical information.

Features shown:
1. Basic Compression
2. Token Counting
3. Compression Strategies (truncate, summarize, extract, tiered)
4. Importance Ranking
5. Critical Information Preservation
6. Auto-Compression
7. Custom Strategies
"""

import sys
from pathlib import Path

# Add parent to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentmemory import (
    ContextCompressor,
    CompressionConfig,
    CompressionMode,
    CompressionResult,
    TokenCounter,
    ImportanceRanker,
    Message,
    MessageRole,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_sample_conversation() -> list[dict]:
    """Create a sample conversation for testing."""
    return [
        {"role": "system", "content": "You are a helpful coding assistant specializing in Python."},
        {"role": "user", "content": "Can you help me understand decorators in Python?"},
        {"role": "assistant", "content": "Of course! Decorators in Python are a powerful feature that allows you to modify or enhance functions or classes without directly changing their source code. A decorator is essentially a function that takes another function as an argument and returns a new function that usually extends the behavior of the original function.\n\nHere's a simple example:\n\n```python\ndef my_decorator(func):\n    def wrapper(*args, **kwargs):\n        print('Before the function call')\n        result = func(*args, **kwargs)\n        print('After the function call')\n        return result\n    return wrapper\n\n@my_decorator\ndef say_hello(name):\n    print(f'Hello, {name}!')\n```\n\nWhen you call `say_hello('Alice')`, it will print:\n- Before the function call\n- Hello, Alice!\n- After the function call"},
        {"role": "user", "content": "That's helpful! Can you show me a decorator with arguments?"},
        {"role": "assistant", "content": "Certainly! When you want a decorator that accepts arguments, you need to add another layer of nesting. The outer function takes the decorator arguments, the middle function takes the function being decorated, and the inner function is the actual wrapper.\n\n```python\ndef repeat(times):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(times):\n                result = func(*args, **kwargs)\n            return result\n        return wrapper\n    return decorator\n\n@repeat(times=3)\ndef greet(name):\n    print(f'Hello, {name}!')\n```\n\nCalling `greet('Bob')` will print 'Hello, Bob!' three times."},
        {"role": "user", "content": "[CRITICAL] Remember: The API key is stored in environment variable API_KEY_SECRET"},
        {"role": "assistant", "content": "I've noted that the API key is stored in the environment variable API_KEY_SECRET. I won't include this in any code examples I provide."},
        {"role": "user", "content": "Now can you explain context managers?"},
        {"role": "assistant", "content": "Context managers in Python provide a clean way to manage resources like files, network connections, or database connections. They ensure that resources are properly acquired and released, even if an error occurs.\n\nThe most common way to use a context manager is with the `with` statement:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n# File is automatically closed here\n```\n\nYou can create your own context managers using either a class with `__enter__` and `__exit__` methods, or using the `contextlib.contextmanager` decorator:\n\n```python\nfrom contextlib import contextmanager\n\n@contextmanager\ndef timing():\n    import time\n    start = time.time()\n    yield\n    print(f'Elapsed: {time.time() - start:.2f}s')\n\nwith timing():\n    # Your code here\n    pass\n```"},
        {"role": "user", "content": "Can you combine decorators and context managers?"},
        {"role": "assistant", "content": "Yes! Decorators and context managers can work together beautifully. In fact, `contextlib.contextmanager` is itself a decorator. Here's an example that combines both concepts:\n\n```python\nfrom contextlib import contextmanager\nfrom functools import wraps\n\n@contextmanager\ndef database_transaction(db):\n    try:\n        yield db\n        db.commit()\n    except Exception:\n        db.rollback()\n        raise\n\ndef with_transaction(func):\n    @wraps(func)\n    def wrapper(db, *args, **kwargs):\n        with database_transaction(db):\n            return func(db, *args, **kwargs)\n    return wrapper\n\n@with_transaction\ndef update_user(db, user_id, data):\n    db.execute('UPDATE users SET ...', data)\n```\n\nThis pattern is very common in web frameworks and ORMs."},
    ]


def demo_token_counting():
    """Demo 1: Token counting basics."""
    print_header("1. TOKEN COUNTING")
    
    counter = TokenCounter(model="gpt-4")
    
    texts = [
        "Hello, world!",
        "This is a longer sentence with more words to count.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]
    
    print("\n  Counting tokens in text samples:")
    for text in texts:
        tokens = counter.count(text)
        print(f"\n    \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"      → {tokens} tokens")
    
    # Count messages
    messages = create_sample_conversation()[:3]
    total = counter.count_messages(messages)
    print(f"\n  Conversation (3 messages): {total} tokens")


def demo_basic_compression():
    """Demo 2: Basic compression."""
    print_header("2. BASIC COMPRESSION")
    
    # Create compressor with token limit
    compressor = ContextCompressor(max_tokens=500)
    
    messages = create_sample_conversation()
    
    print(f"\n  Original conversation: {len(messages)} messages")
    original_tokens = compressor.token_counter.count_messages(messages)
    print(f"  Original tokens: {original_tokens}")
    
    # Check if compression is needed
    needs_compression = compressor.needs_compression(messages)
    print(f"  Needs compression: {needs_compression}")
    
    if needs_compression:
        result = compressor.compress(messages)
        print(f"\n  After compression:")
        print(f"    Messages: {len(result.messages)}")
        print(f"    Tokens: {result.compressed_tokens}")
        print(f"    Compression ratio: {result.compression_ratio:.1%}")
        print(f"    Strategy used: {result.strategy_used}")


def demo_compression_strategies():
    """Demo 3: Different compression strategies."""
    print_header("3. COMPRESSION STRATEGIES")
    
    messages = create_sample_conversation()
    
    strategies = ["truncate", "summarize", "extract_key_facts", "tiered"]
    
    print(f"\n  Original: {len(messages)} messages")
    
    for strategy in strategies:
        compressor = ContextCompressor(max_tokens=600)
        
        try:
            result = compressor.compress(messages, strategy=strategy, token_budget=600)
            print(f"\n  {strategy.upper()}:")
            print(f"    Messages: {len(result.messages)}")
            print(f"    Tokens: {result.original_tokens} → {result.compressed_tokens}")
            print(f"    Ratio: {result.compression_ratio:.1%}")
        except Exception as e:
            print(f"\n  {strategy.upper()}: (requires LLM for summarization)")


def demo_importance_ranking():
    """Demo 4: Message importance ranking."""
    print_header("4. IMPORTANCE RANKING")
    
    ranker = ImportanceRanker()
    
    messages = create_sample_conversation()
    scored = ranker.rank(messages)
    
    print("\n  Messages ranked by importance:")
    for i, msg in enumerate(sorted(scored, key=lambda x: x.importance_score, reverse=True)[:5]):
        role = msg.message.get("role", "unknown")
        content = msg.message.get("content", "")[:50]
        score = msg.importance_score
        print(f"\n    {i+1}. [{role}] {content}...")
        print(f"       Score: {score:.3f}")
        print(f"       Factors: {dict(msg.factors._asdict()) if hasattr(msg.factors, '_asdict') else msg.factors}")


def demo_critical_preservation():
    """Demo 5: Preserving critical information."""
    print_header("5. CRITICAL INFORMATION PRESERVATION")
    
    # Compressor with small token budget
    config = CompressionConfig(
        max_tokens=400,
        preserve_system=True,
        preserve_critical=True,
        preserve_recent=2,
    )
    compressor = ContextCompressor(config=config)
    
    messages = create_sample_conversation()
    
    print("\n  Original messages with critical marker:")
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if "[CRITICAL]" in content:
            print(f"    Message {i}: {content[:60]}...")
    
    result = compressor.compress(messages)
    
    print("\n  After aggressive compression:")
    print(f"    Messages: {len(messages)} → {len(result.messages)}")
    
    # Check if critical info preserved
    compressed_content = " ".join(m.get("content", "") for m in result.messages)
    critical_preserved = "API_KEY_SECRET" in compressed_content
    print(f"    Critical info preserved: {'✓' if critical_preserved else '✗'}")
    
    # Show which messages were kept
    print("\n  Preserved messages:")
    for i, msg in enumerate(result.messages):
        role = msg.get("role", "")
        content = msg.get("content", "")[:60]
        print(f"    {i+1}. [{role}] {content}...")


def demo_compression_modes():
    """Demo 6: Compression mode presets."""
    print_header("6. COMPRESSION MODES")
    
    messages = create_sample_conversation()
    
    modes = [
        (CompressionMode.LOSSLESS, "Truncation only"),
        (CompressionMode.CONSERVATIVE, "Preserve more info"),
        (CompressionMode.BALANCED, "Balance compression/retention"),
        (CompressionMode.AGGRESSIVE, "Maximum compression"),
    ]
    
    print(f"\n  Original: {len(messages)} messages")
    
    for mode, description in modes:
        compressor = ContextCompressor(max_tokens=600, mode=mode)
        
        result = compressor.compress(messages)
        print(f"\n  {mode.value.upper()} ({description}):")
        print(f"    Strategy: {result.strategy_used}")
        print(f"    Messages: {len(result.messages)}")
        print(f"    Tokens: {result.compressed_tokens}")


def demo_auto_compression():
    """Demo 7: Automatic strategy selection."""
    print_header("7. AUTO-COMPRESSION")
    
    compressor = ContextCompressor(max_tokens=800)
    messages = create_sample_conversation()
    
    print("\n  Auto-selecting best compression strategy...")
    
    result = compressor.compress_auto(messages, budget=500)
    
    print(f"\n  Result:")
    print(f"    Strategy selected: {result.strategy_used}")
    print(f"    Messages: {len(messages)} → {len(result.messages)}")
    print(f"    Tokens: {result.original_tokens} → {result.compressed_tokens}")
    print(f"    Compression ratio: {result.compression_ratio:.1%}")
    print(f"    Within budget: {'✓' if result.compressed_tokens <= 500 else '✗'}")


def demo_compression_stats():
    """Demo 8: Getting compression statistics."""
    print_header("8. COMPRESSION STATISTICS")
    
    compressor = ContextCompressor(max_tokens=600)
    messages = create_sample_conversation()
    
    stats = compressor.get_compression_stats(messages)
    
    print("\n  Conversation analysis:")
    print(f"    Total messages: {stats.get('message_count', 0)}")
    print(f"    Total tokens: {stats.get('total_tokens', 0)}")
    print(f"    Token budget: {stats.get('token_budget', 0)}")
    print(f"    Over budget by: {stats.get('tokens_over_budget', 0)} tokens")
    print(f"    Compression needed: {'Yes' if stats.get('needs_compression', False) else 'No'}")
    
    print("\n  Token distribution by role:")
    by_role = stats.get('tokens_by_role', {})
    for role, count in by_role.items():
        pct = (count / stats.get('total_tokens', 1)) * 100
        bar = "█" * int(pct / 5)
        print(f"    {role:10}: {count:4} tokens ({pct:5.1f}%) {bar}")


def demo_message_class():
    """Demo 9: Using the Message class."""
    print_header("9. MESSAGE CLASS USAGE")
    
    # Create structured messages
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant.",
            is_critical=True,  # Never compress
        ),
        Message(
            role=MessageRole.USER,
            content="What's the weather like?",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="I don't have access to real-time weather data.",
            importance_boost=0.5,  # Boost importance
        ),
    ]
    
    print("\n  Creating structured messages:")
    for msg in messages:
        print(f"\n    Role: {msg.role.value}")
        print(f"    Content: {msg.content[:50]}...")
        print(f"    Critical: {msg.is_critical}")
        print(f"    Importance boost: {msg.importance_boost}")
    
    # Convert to dict format
    print("\n  Converting to dict format:")
    for msg in messages:
        d = msg.to_dict()
        print(f"    {d}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  AGENT MEMORY TOOLKIT - CONTEXT COMPRESSION DEMO")
    print("=" * 60)
    
    demo_token_counting()
    demo_basic_compression()
    demo_compression_strategies()
    demo_importance_ranking()
    demo_critical_preservation()
    demo_compression_modes()
    demo_auto_compression()
    demo_compression_stats()
    demo_message_class()
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\n  Compression features demonstrated:")
    print("    ✓ Token Counting")
    print("    ✓ Basic Compression")
    print("    ✓ Multiple Strategies (truncate, summarize, extract, tiered)")
    print("    ✓ Importance Ranking")
    print("    ✓ Critical Information Preservation")
    print("    ✓ Compression Modes (lossless → aggressive)")
    print("    ✓ Auto-Compression")
    print("    ✓ Compression Statistics")
    print("    ✓ Structured Message Class")
    print("=" * 60)


if __name__ == "__main__":
    main()
