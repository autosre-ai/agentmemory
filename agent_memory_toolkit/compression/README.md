# Context Compression Engine

Intelligent context compression for LLM conversations. Fits conversation history within token budgets while preserving critical information.

## Features

- **Multiple compression strategies**: truncate, summarize, extract-key-facts, tiered
- **Token counting**: Accurate counting using tiktoken
- **Importance ranking**: Smart prioritization of messages
- **Tiered compression**: Recent = full fidelity, older = lossy
- **Critical info preservation**: Markers like `[CRITICAL]` are never lost
- **LLM support**: Optional LLM-based summarization

## Quick Start

```python
from agent_memory import ContextCompressor, CompressionConfig

# Basic usage
compressor = ContextCompressor(max_tokens=4000)
result = compressor.compress(messages)
print(f"Compressed {result.original_tokens} → {result.compressed_tokens} tokens")

# With custom config
config = CompressionConfig(
    max_tokens=4000,
    reserve_tokens=500,  # Reserve for response
    recent_count=4,      # Keep last 4 messages at full fidelity
    medium_count=8,      # Summarize next 8
    preserve_system=True,
    preserve_critical=True,
)
compressor = ContextCompressor(config=config)
```

## Compression Strategies

### Truncate
Simple removal of old messages. Fastest but loses information.

```python
result = compressor.compress(messages, strategy="truncate")
```

### Summarize
Converts older messages into summaries. Requires LLM for best results.

```python
compressor = ContextCompressor(max_tokens=4000, llm_provider=my_llm)
result = compressor.compress(messages, strategy="summarize")
```

### Extract Key Facts
Extracts only key facts (dates, names, decisions) from older messages.

```python
result = compressor.compress(messages, strategy="extract_key_facts")
```

### Tiered (Default)
Combines strategies based on message age:
- Recent messages: Kept intact
- Medium-age: Summarized
- Old messages: Key facts only

```python
result = compressor.compress(messages, strategy="tiered")
```

## Token Counting

```python
from agent_memory import TokenCounter

counter = TokenCounter(model="gpt-4")

# Count text tokens
tokens = counter.count("Hello, world!")

# Count chat message tokens (with overhead)
tokens = counter.count_messages(messages)

# Truncate to fit
truncated = counter.truncate_to_tokens(long_text, 100)
```

## Importance Ranking

```python
from agent_memory import ImportanceRanker

ranker = ImportanceRanker()
scored = ranker.rank(messages)

for msg in scored:
    print(f"Score: {msg.score:.2f} - {msg.content[:50]}")
```

## Critical Information Markers

Messages containing these markers are always preserved:
- `[CRITICAL]`
- `[IMPORTANT]`
- `[REMEMBER]`
- `[PRESERVE]`
- `[KEY]`

## API Reference

### ContextCompressor

Main class for compression.

```python
ContextCompressor(
    max_tokens: int = 4000,
    model: str = "gpt-4",
    llm_provider: Optional[LLMProvider] = None,
    mode: CompressionMode = CompressionMode.BALANCED,
    config: Optional[CompressionConfig] = None,
)
```

Methods:
- `compress(messages, strategy=None, token_budget=None) -> CompressionResult`
- `compress_auto(messages, token_budget=None) -> CompressionResult`
- `needs_compression(messages) -> bool`
- `count_tokens(messages) -> int`
- `rank_messages(messages) -> list[ScoredMessage]`
- `get_compression_stats(messages) -> dict`

### CompressionResult

```python
@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    messages: list[dict]
    compression_ratio: float
    strategy_used: str
    details: dict
```

### CompressionMode

```python
class CompressionMode(str, Enum):
    AGGRESSIVE = "aggressive"   # Maximum compression
    BALANCED = "balanced"       # Balance compression/retention
    CONSERVATIVE = "conservative"  # Preserve more information
    LOSSLESS = "lossless"       # Only truncate, no summarization
```
