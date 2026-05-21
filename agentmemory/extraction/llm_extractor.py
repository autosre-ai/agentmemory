"""
LLM-Based Memory Extractor

Uses OpenAI or Anthropic APIs for intelligent memory extraction.
Provides higher accuracy for complex or nuanced statements.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, Protocol

from .domains import CognitiveDomain, Memory, DOMAIN_PROMPTS


class LLMClient(Protocol):
    """Protocol for LLM clients (OpenAI, Anthropic, etc.)."""
    
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate completion from prompt."""
        ...


class OpenAIClient:
    """OpenAI API client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model to use for extraction
        """
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client
    
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """Anthropic API client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-latest"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model to use for extraction
        """
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client
    
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate completion."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system if system else "You are a memory extraction assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


class LLMExtractor:
    """
    Extract structured memories using LLM intelligence.
    
    Provides high-accuracy extraction for complex statements,
    implicit information, and nuanced context.
    """
    
    SYSTEM_PROMPT = """You are a precise memory extraction system. Your task is to extract 
structured facts from text and categorize them into cognitive domains.

Output JSON array of memories, each with:
- domain: one of [biography, preferences, work, social, temporal, procedural]
- key: short identifier (snake_case)
- value: the extracted fact
- confidence: 0.0 to 1.0 (how certain you are)

Rules:
1. Only extract explicit or strongly implied facts
2. Don't invent or assume information not present
3. Use appropriate confidence scores (0.9+ for explicit, 0.7-0.9 for implied)
4. Keep values concise but complete
5. Prefer specific over vague extractions

Example input: "I'm John, a Python developer at Google working on the search team. I prefer dark mode."

Example output:
[
  {"domain": "biography", "key": "name", "value": "John", "confidence": 0.95},
  {"domain": "work", "key": "role", "value": "Python developer", "confidence": 0.95},
  {"domain": "work", "key": "company", "value": "Google", "confidence": 0.95},
  {"domain": "work", "key": "team", "value": "search team", "confidence": 0.9},
  {"domain": "preferences", "key": "ui_mode", "value": "dark mode", "confidence": 0.9}
]
"""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM extractor.
        
        Args:
            client: Pre-configured LLM client (takes precedence)
            provider: Provider to use if no client ("openai" or "anthropic")
            api_key: API key for the provider
            model: Model to use (provider-specific defaults apply)
        """
        if client:
            self._client = client
        elif provider.lower() == "anthropic":
            self._client = AnthropicClient(
                api_key=api_key,
                model=model or "claude-3-5-sonnet-latest"
            )
        else:
            self._client = OpenAIClient(
                api_key=api_key,
                model=model or "gpt-4o-mini"
            )
    
    def extract(self, text: str, source: Optional[str] = None) -> list[Memory]:
        """
        Extract memories from text using LLM.
        
        Args:
            text: Input text to analyze
            source: Optional source identifier
            
        Returns:
            List of extracted Memory objects
        """
        if not text.strip():
            return []
        
        prompt = f"""Extract all memorable facts from this text:

---
{text}
---

Return a JSON array of memories. If no memories can be extracted, return [].
"""
        
        try:
            response = self._client.complete(prompt, self.SYSTEM_PROMPT)
            return self._parse_response(response, source)
        except Exception as e:
            # Return empty list on error, let caller handle logging
            return []
    
    def extract_domain(
        self,
        text: str,
        domain: CognitiveDomain,
        source: Optional[str] = None
    ) -> list[Memory]:
        """
        Extract memories for a specific domain.
        
        Uses domain-specific prompts for more accurate extraction.
        
        Args:
            text: Input text to analyze
            domain: Target cognitive domain
            source: Optional source identifier
            
        Returns:
            List of extracted Memory objects for the domain
        """
        if not text.strip():
            return []
        
        domain_prompt = DOMAIN_PROMPTS.get(domain, "")
        
        prompt = f"""Extract {domain.value.upper()} information from this text:

{domain_prompt}

Text:
---
{text}
---

Return a JSON array of memories with domain="{domain.value}".
"""
        
        try:
            response = self._client.complete(prompt, self.SYSTEM_PROMPT)
            memories = self._parse_response(response, source)
            # Filter to ensure only the requested domain
            return [m for m in memories if m.domain == domain]
        except Exception:
            return []
    
    def extract_all_domains(
        self,
        text: str,
        source: Optional[str] = None
    ) -> dict[CognitiveDomain, list[Memory]]:
        """
        Extract memories for all domains separately.
        
        More accurate but slower (6 LLM calls).
        
        Args:
            text: Input text to analyze
            source: Optional source identifier
            
        Returns:
            Dictionary mapping domains to extracted memories
        """
        results = {}
        for domain in CognitiveDomain:
            results[domain] = self.extract_domain(text, domain, source)
        return results
    
    def _parse_response(self, response: str, source: Optional[str]) -> list[Memory]:
        """Parse LLM response into Memory objects."""
        memories = []
        
        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return memories
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return memories
        
        if not isinstance(data, list):
            return memories
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            try:
                domain_str = item.get("domain", "").lower()
                domain = CognitiveDomain.from_string(domain_str)
                
                key = str(item.get("key", "")).strip()
                value = str(item.get("value", "")).strip()
                confidence = float(item.get("confidence", 0.8))
                
                if key and value:
                    memories.append(Memory(
                        domain=domain,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "llm"},
                    ))
            except (ValueError, KeyError):
                continue
        
        return memories


class MockLLMClient:
    """
    Mock LLM client for testing.
    
    Returns predefined responses based on input patterns.
    """
    
    def __init__(self, responses: Optional[dict[str, str]] = None):
        """
        Initialize with optional custom responses.
        
        Args:
            responses: Dict mapping input patterns to JSON responses
        """
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []
    
    def complete(self, prompt: str, system: str = "") -> str:
        """Return mock response."""
        self.calls.append((prompt, system))
        
        # Check for custom responses
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                return response
        
        # Default: extract simple patterns
        memories = []
        
        if "john" in prompt.lower():
            memories.append({
                "domain": "biography",
                "key": "name",
                "value": "John",
                "confidence": 0.95
            })
        
        if "python" in prompt.lower():
            memories.append({
                "domain": "work",
                "key": "skill",
                "value": "Python",
                "confidence": 0.9
            })
        
        if "dark mode" in prompt.lower():
            memories.append({
                "domain": "preferences",
                "key": "ui_mode",
                "value": "dark mode",
                "confidence": 0.85
            })
        
        return json.dumps(memories)
