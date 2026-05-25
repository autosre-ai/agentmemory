# LLM Framework Integrations

Agent Memory Toolkit provides seamless integrations with popular LLM providers and frameworks, enabling AI-powered memory operations like summarization, extraction, and reasoning.

## Overview

The LLM module provides unified interfaces for:

- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-Turbo, and O1 models
- **Anthropic**: Claude 3.5, Claude 3, and Claude 2 models
- **LangChain**: Universal adapter for any LangChain-compatible model

All providers implement a common interface, making it easy to swap between them.

## Quick Start

```python
from agent_memory_toolkit.llm import (
    # OpenAI
    OpenAIProvider,
    OpenAIConfig,
    OpenAIModel,
    # Anthropic
    AnthropicProvider,
    AnthropicConfig,
    AnthropicModel,
    # LangChain
    LangChainProvider,
)

# OpenAI
provider = OpenAIProvider(OpenAIConfig(
    model=OpenAIModel.GPT_4O,
    api_key="sk-...",  # or set OPENAI_API_KEY env var
))

# Anthropic
provider = AnthropicProvider(AnthropicConfig(
    model=AnthropicModel.CLAUDE_3_5_SONNET,
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
))

# LangChain (any model)
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o")
provider = LangChainProvider(chat)

# Use any provider the same way
response = provider.complete("What is machine learning?")
print(response)

# With system prompt
response = provider.complete_with_system(
    system="You are a helpful AI assistant.",
    prompt="Explain quantum computing.",
)
```

## OpenAI Integration

### Basic Usage

```python
from agent_memory_toolkit.llm import OpenAIProvider, OpenAIConfig, OpenAIModel

# Create provider with default settings
provider = OpenAIProvider()  # Uses OPENAI_API_KEY env var

# Or with explicit configuration
provider = OpenAIProvider(OpenAIConfig(
    model=OpenAIModel.GPT_4O,
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1024,
))

# Simple completion
response = provider.complete("Tell me about Python programming.")
print(response)

# Completion with system prompt
response = provider.complete_with_system(
    system="You are an expert programmer. Give concise, technical answers.",
    prompt="What are Python decorators?",
)
print(response)
```

### Available Models

```python
from agent_memory_toolkit.llm import OpenAIModel

# GPT-4 Omni (recommended)
OpenAIModel.GPT_4O           # Latest GPT-4o
OpenAIModel.GPT_4O_MINI      # Smaller, faster, cheaper

# GPT-4 Turbo
OpenAIModel.GPT_4_TURBO      # Latest GPT-4 Turbo
OpenAIModel.GPT_4_TURBO_PREVIEW

# GPT-4
OpenAIModel.GPT_4            # Original GPT-4
OpenAIModel.GPT_4_32K        # Extended context

# GPT-3.5
OpenAIModel.GPT_35_TURBO     # Fast and affordable
OpenAIModel.GPT_35_TURBO_16K

# O1 Models (reasoning)
OpenAIModel.O1_PREVIEW       # Best for complex reasoning
OpenAIModel.O1_MINI          # Faster O1 variant
```

### Full Response with Metadata

```python
# Get complete response with usage statistics
response = provider.complete_full(
    prompt="Explain machine learning",
    system="You are a data science expert.",
    max_tokens=500,
    temperature=0.5,
)

print(f"Response: {response.content}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.finish_reason}")
print(f"Latency: {response.latency_ms:.0f}ms")

# Token usage and cost
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Estimated cost: ${response.usage.total_cost:.6f}")
```

### Streaming

```python
# Stream tokens as they're generated
for chunk in provider.complete_stream(
    prompt="Write a short poem about code.",
    system="You are a creative poet.",
):
    print(chunk, end="", flush=True)

# With callback
def on_token(token: str):
    # Process each token
    print(token, end="", flush=True)

for _ in provider.complete_stream(
    prompt="Explain recursion",
    callback=on_token,
):
    pass
```

### Token Counting

```python
# Count tokens before sending
text = "This is a sample text to count tokens for."
token_count = provider.count_tokens(text)
print(f"Token count: {token_count}")

# Track cumulative usage
provider.complete("Question 1")
provider.complete("Question 2")
total = provider.total_usage
print(f"Total tokens used: {total.total_tokens}")
print(f"Total cost: ${total.total_cost:.4f}")

# Reset counter
final_usage = provider.reset_usage()
```

### Configuration Options

```python
from agent_memory_toolkit.llm import OpenAIConfig

config = OpenAIConfig(
    # Model selection
    model="gpt-4o",
    
    # API settings
    api_key="sk-...",
    organization="org-...",
    base_url=None,  # Custom endpoint (Azure, proxy)
    
    # Generation parameters
    temperature=0.7,
    max_tokens=1024,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop_sequences=["END"],
    
    # Retry configuration
    max_retries=3,
    retry_delay=1.0,
    timeout=60.0,
    
    # Response format (JSON mode)
    response_format={"type": "json_object"},
    
    # Reproducibility
    seed=42,
)
```

### Async Support

```python
from agent_memory_toolkit.llm import AsyncOpenAIProvider, OpenAIConfig
import asyncio

async def main():
    provider = AsyncOpenAIProvider(OpenAIConfig(model="gpt-4o"))
    
    # Async completion
    response = await provider.complete("Hello, world!")
    print(response)
    
    # With system prompt
    response = await provider.complete_with_system(
        system="You are helpful.",
        prompt="What is Python?",
    )
    
    # Full response
    response = await provider.complete_full(
        prompt="Explain AI",
        max_tokens=500,
    )
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

## Anthropic Integration

### Basic Usage

```python
from agent_memory_toolkit.llm import AnthropicProvider, AnthropicConfig, AnthropicModel

# Create provider
provider = AnthropicProvider(AnthropicConfig(
    model=AnthropicModel.CLAUDE_3_5_SONNET,
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
))

# Simple completion
response = provider.complete("Tell me about neural networks.")
print(response)

# With system prompt
response = provider.complete_with_system(
    system="You are a machine learning expert.",
    prompt="Explain backpropagation.",
)
```

### Available Models

```python
from agent_memory_toolkit.llm import AnthropicModel

# Claude 3.5 (recommended)
AnthropicModel.CLAUDE_3_5_SONNET        # Best balance
AnthropicModel.CLAUDE_3_5_SONNET_LATEST
AnthropicModel.CLAUDE_3_5_HAIKU         # Fastest
AnthropicModel.CLAUDE_3_5_HAIKU_LATEST

# Claude 3
AnthropicModel.CLAUDE_3_OPUS            # Most capable
AnthropicModel.CLAUDE_3_OPUS_LATEST
AnthropicModel.CLAUDE_3_SONNET
AnthropicModel.CLAUDE_3_HAIKU

# Claude 2 (legacy)
AnthropicModel.CLAUDE_2_1
AnthropicModel.CLAUDE_2_0
AnthropicModel.CLAUDE_INSTANT
```

### Full Response with Metadata

```python
response = provider.complete_full(
    prompt="Explain transformers in AI",
    system="You are a deep learning researcher.",
    max_tokens=1000,
    temperature=0.5,
)

print(f"Response: {response.content}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.finish_reason}")
print(f"Latency: {response.latency_ms:.0f}ms")

# Token usage
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cost: ${response.usage.total_cost:.6f}")
```

### Streaming

```python
# Stream response
for chunk in provider.complete_stream(
    prompt="Write a haiku about coding.",
    system="You are a poet.",
):
    print(chunk, end="", flush=True)
```

### Configuration Options

```python
from agent_memory_toolkit.llm import AnthropicConfig

config = AnthropicConfig(
    # Model
    model="claude-3-5-sonnet-20241022",
    
    # API settings
    api_key="sk-ant-...",
    base_url=None,
    
    # Generation parameters
    temperature=0.7,
    max_tokens=1024,
    top_p=1.0,
    stop_sequences=["END"],
    
    # Retry configuration
    max_retries=3,
    timeout=60.0,
    
    # Anthropic-specific
    metadata={"user_id": "user123"},
)
```

### Async Support

```python
from agent_memory_toolkit.llm import AsyncAnthropicProvider, AnthropicConfig
import asyncio

async def main():
    provider = AsyncAnthropicProvider(AnthropicConfig(
        model="claude-3-5-sonnet-20241022"
    ))
    
    response = await provider.complete("Hello!")
    print(response)
    
    response = await provider.complete_with_system(
        system="You are helpful.",
        prompt="What is Python?",
    )

asyncio.run(main())
```

## LangChain Integration

The LangChain adapter provides universal compatibility with any LangChain model.

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from agent_memory_toolkit.llm import LangChainProvider

# OpenAI via LangChain
chat_openai = ChatOpenAI(model="gpt-4o", temperature=0.7)
provider = LangChainProvider(chat_openai)

# Anthropic via LangChain
chat_claude = ChatAnthropic(model="claude-3-5-sonnet-20241022")
provider = LangChainProvider(chat_claude)

# Any LangChain model works!
from langchain_community.llms import Ollama
ollama = Ollama(model="llama2")
provider = LangChainProvider(ollama)

# Use the unified interface
response = provider.complete("Hello, world!")
response = provider.complete_with_system(
    system="You are helpful.",
    prompt="What is Python?",
)
```

### Specialized Adapters

```python
from agent_memory_toolkit.llm import ChatModelAdapter, LLMAdapter

# For chat models (ChatOpenAI, ChatAnthropic, etc.)
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o")
adapter = ChatModelAdapter(chat)

# For completion LLMs (OpenAI, Cohere, etc.)
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct")
adapter = LLMAdapter(llm)
```

### Full Response

```python
response = provider.complete_full(
    prompt="Explain neural networks",
    system="You are an AI researcher.",
    max_tokens=500,
    temperature=0.7,
)

print(f"Content: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Latency: {response.latency_ms:.0f}ms")
```

### Streaming

```python
# Stream if the underlying model supports it
for chunk in provider.complete_stream(
    prompt="Write a short story.",
    system="You are a creative writer.",
):
    print(chunk, end="", flush=True)
```

### Using with Memory Toolkit Components

```python
from agent_memory_toolkit.llm import OpenAIProvider, OpenAIConfig
from agent_memory_toolkit.compression import AbstractiveSummarizer

# Create LLM provider
llm = OpenAIProvider(OpenAIConfig(model="gpt-4o-mini"))

# Use with summarizer
summarizer = AbstractiveSummarizer(llm_provider=llm)
summary = summarizer.summarize(memories, level=SummaryLevel.STANDARD)

# Use with extractor
from agent_memory_toolkit import LLMExtractor
extractor = LLMExtractor(llm_provider=llm)
result = extractor.extract(conversation)
```

## Error Handling

All providers raise consistent exceptions:

```python
from agent_memory_toolkit.llm import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
)

try:
    response = provider.complete("Hello")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except InvalidRequestError as e:
    print(f"Bad request: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Cost Tracking

Track costs across multiple requests:

```python
# Use any provider
provider = OpenAIProvider(OpenAIConfig(model="gpt-4o"))

# Make multiple requests
for question in questions:
    provider.complete(question)

# Check cumulative usage
usage = provider.total_usage
print(f"Total tokens: {usage.total_tokens}")
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total cost: ${usage.total_cost:.4f}")

# Reset and continue
provider.reset_usage()
```

## Model Capabilities

Check what a model supports:

```python
caps = provider.model_capabilities

print(f"Max tokens: {caps.max_tokens}")
print(f"Context window: {caps.context_window}")
print(f"Supports streaming: {caps.supports_streaming}")
print(f"Supports system prompt: {caps.supports_system_prompt}")
print(f"Supports function calling: {caps.supports_function_calling}")
print(f"Supports vision: {caps.supports_vision}")
print(f"Input price per 1M tokens: ${caps.input_price_per_million}")
print(f"Output price per 1M tokens: ${caps.output_price_per_million}")
```

## Environment Variables

The providers look for API keys in environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Custom Providers

Implement the `LLMProvider` protocol to create custom providers:

```python
from agent_memory_toolkit.llm import LLMProvider, LLMResponse, TokenUsage

class MyCustomProvider:
    """Custom LLM provider."""
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        # Your implementation
        return "Generated response"
    
    def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500,
    ) -> str:
        # Your implementation
        return "Generated response with system"

# Use with memory toolkit
provider = MyCustomProvider()
summarizer = AbstractiveSummarizer(llm_provider=provider)
```

## Best Practices

### 1. Use Environment Variables for API Keys

```python
# Good - uses environment variable
provider = OpenAIProvider()

# Avoid hardcoding keys
provider = OpenAIProvider(OpenAIConfig(api_key="sk-..."))  # Not recommended
```

### 2. Configure Appropriate Timeouts

```python
config = OpenAIConfig(
    timeout=120.0,  # Longer timeout for complex prompts
    max_retries=5,  # More retries for production
)
```

### 3. Use Streaming for Long Responses

```python
# Better UX for long responses
for chunk in provider.complete_stream(prompt):
    display_to_user(chunk)
```

### 4. Monitor Token Usage

```python
# Track usage for cost management
response = provider.complete_full(prompt)
log_usage(response.usage)

# Check before sending large prompts
token_count = provider.count_tokens(long_prompt)
if token_count > 10000:
    # Split or summarize
    pass
```

### 5. Handle Errors Gracefully

```python
import time

def complete_with_retry(provider, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return provider.complete(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(e.retry_after or 60)
            else:
                raise
```
