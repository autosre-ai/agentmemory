"""Integration modules for LangChain and LlamaIndex frameworks.

This module provides drop-in memory backends for popular AI agent frameworks,
allowing them to use the Agent Memory Toolkit's MemoryStore as their memory backend.

Example (LangChain):
    >>> from agentmemory.integrations.langchain import AgentMemoryToolkitMemory
    >>> from agentmemory import MemoryStore
    >>> 
    >>> store = MemoryStore("agent_memory.db")
    >>> memory = AgentMemoryToolkitMemory(store=store)
    >>> 
    >>> # Use with LangChain ConversationChain or other components
    >>> from langchain.chains import ConversationChain
    >>> chain = ConversationChain(llm=llm, memory=memory)

Example (LlamaIndex):
    >>> from agentmemory.integrations.llamaindex import AgentMemoryToolkitStore
    >>> from agentmemory import MemoryStore
    >>> 
    >>> store = MemoryStore("agent_memory.db")
    >>> memory_store = AgentMemoryToolkitStore(store=store)
    >>> 
    >>> # Use with LlamaIndex chat engine or other components
"""

# Lazy imports to avoid requiring both frameworks
def __getattr__(name: str):
    if name == "AgentMemoryToolkitMemory":
        from .langchain import AgentMemoryToolkitMemory
        return AgentMemoryToolkitMemory
    elif name == "AgentMemoryToolkitChatMemory":
        from .langchain import AgentMemoryToolkitChatMemory
        return AgentMemoryToolkitChatMemory
    elif name == "AgentMemoryToolkitStore":
        from .llamaindex import AgentMemoryToolkitStore
        return AgentMemoryToolkitStore
    elif name == "AgentMemoryToolkitChatStore":
        from .llamaindex import AgentMemoryToolkitChatStore
        return AgentMemoryToolkitChatStore
    elif name == "AgentMemoryToolkitVectorStore":
        from .llamaindex import AgentMemoryToolkitVectorStore
        return AgentMemoryToolkitVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentMemoryToolkitMemory",
    "AgentMemoryToolkitChatMemory",
    "AgentMemoryToolkitStore",
    "AgentMemoryToolkitChatStore",
    "AgentMemoryToolkitVectorStore",
]
