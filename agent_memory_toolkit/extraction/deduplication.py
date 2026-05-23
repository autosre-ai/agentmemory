"""
Memory Deduplication

Identifies and removes duplicate memories using multiple strategies:
- Exact match
- Fuzzy matching
- Semantic similarity (optional)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .domains import CognitiveDomain, Memory


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    
    unique_memories: list[Memory]
    duplicates_removed: int
    duplicate_groups: list[list[Memory]]  # Groups of memories considered duplicates
    
    def __len__(self) -> int:
        return len(self.unique_memories)


class MemoryDeduplicator:
    """
    Deduplicate memories using configurable strategies.
    
    Strategies:
    - exact: Identical domain + key + value
    - fuzzy: Similar text with configurable threshold
    - semantic: Embedding-based similarity (requires embeddings)
    """
    
    def __init__(
        self,
        strategy: str = "fuzzy",
        similarity_threshold: float = 0.85,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        """
        Initialize deduplicator.
        
        Args:
            strategy: Deduplication strategy ("exact", "fuzzy", "semantic")
            similarity_threshold: Threshold for fuzzy/semantic matching (0-1)
            embedding_fn: Function to generate embeddings for semantic matching
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn
    
    def deduplicate(self, memories: list[Memory]) -> DeduplicationResult:
        """
        Remove duplicate memories.
        
        Args:
            memories: List of memories to deduplicate
            
        Returns:
            DeduplicationResult with unique memories and duplicate info
        """
        if not memories:
            return DeduplicationResult([], 0, [])
        
        if self.strategy == "exact":
            return self._dedupe_exact(memories)
        elif self.strategy == "semantic" and self.embedding_fn:
            return self._dedupe_semantic(memories)
        else:
            return self._dedupe_fuzzy(memories)
    
    def _dedupe_exact(self, memories: list[Memory]) -> DeduplicationResult:
        """Exact match deduplication."""
        seen: dict[tuple, Memory] = {}
        duplicate_groups: list[list[Memory]] = []
        
        for memory in memories:
            key = (memory.domain, memory.key.lower(), memory.value.lower())
            
            if key in seen:
                # Found duplicate - keep higher confidence one
                existing = seen[key]
                if memory.confidence > existing.confidence:
                    # Find or create duplicate group
                    found_group = False
                    for group in duplicate_groups:
                        if existing in group:
                            group.append(memory)
                            found_group = True
                            break
                    if not found_group:
                        duplicate_groups.append([existing, memory])
                    seen[key] = memory
                else:
                    for group in duplicate_groups:
                        if existing in group:
                            group.append(memory)
                            break
                    else:
                        duplicate_groups.append([existing, memory])
            else:
                seen[key] = memory
        
        unique = list(seen.values())
        return DeduplicationResult(
            unique_memories=unique,
            duplicates_removed=len(memories) - len(unique),
            duplicate_groups=duplicate_groups,
        )
    
    def _dedupe_fuzzy(self, memories: list[Memory]) -> DeduplicationResult:
        """Fuzzy match deduplication using text similarity."""
        # Group by domain first
        by_domain: dict[CognitiveDomain, list[Memory]] = defaultdict(list)
        for memory in memories:
            by_domain[memory.domain].append(memory)
        
        unique = []
        duplicate_groups: list[list[Memory]] = []
        
        for domain, domain_memories in by_domain.items():
            domain_unique = self._dedupe_domain_fuzzy(domain_memories, duplicate_groups)
            unique.extend(domain_unique)
        
        return DeduplicationResult(
            unique_memories=unique,
            duplicates_removed=len(memories) - len(unique),
            duplicate_groups=duplicate_groups,
        )
    
    def _dedupe_domain_fuzzy(
        self,
        memories: list[Memory],
        duplicate_groups: list[list[Memory]]
    ) -> list[Memory]:
        """Fuzzy dedupe within a domain."""
        if not memories:
            return []
        
        unique: list[Memory] = []
        
        for memory in memories:
            is_duplicate = False
            
            for i, existing in enumerate(unique):
                if memory.similar_to(existing, self.similarity_threshold):
                    # Keep higher confidence
                    if memory.confidence > existing.confidence:
                        # Find or create duplicate group
                        found = False
                        for group in duplicate_groups:
                            if existing in group:
                                group.append(memory)
                                found = True
                                break
                        if not found:
                            duplicate_groups.append([existing, memory])
                        unique[i] = memory
                    else:
                        for group in duplicate_groups:
                            if existing in group:
                                group.append(memory)
                                break
                        else:
                            duplicate_groups.append([existing, memory])
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(memory)
        
        return unique
    
    def _dedupe_semantic(self, memories: list[Memory]) -> DeduplicationResult:
        """Semantic similarity deduplication using embeddings."""
        if not self.embedding_fn:
            return self._dedupe_fuzzy(memories)
        
        # Group by domain
        by_domain: dict[CognitiveDomain, list[Memory]] = defaultdict(list)
        for memory in memories:
            by_domain[memory.domain].append(memory)
        
        unique = []
        duplicate_groups: list[list[Memory]] = []
        
        for domain, domain_memories in by_domain.items():
            # Get embeddings for all memories in domain
            texts = [f"{m.key}: {m.value}" for m in domain_memories]
            embeddings = [self.embedding_fn(t) for t in texts]
            
            domain_unique = self._dedupe_domain_semantic(
                domain_memories, embeddings, duplicate_groups
            )
            unique.extend(domain_unique)
        
        return DeduplicationResult(
            unique_memories=unique,
            duplicates_removed=len(memories) - len(unique),
            duplicate_groups=duplicate_groups,
        )
    
    def _dedupe_domain_semantic(
        self,
        memories: list[Memory],
        embeddings: list[list[float]],
        duplicate_groups: list[list[Memory]]
    ) -> list[Memory]:
        """Semantic dedupe within a domain."""
        if not memories:
            return []
        
        unique: list[Memory] = []
        unique_embeddings: list[list[float]] = []
        
        for memory, embedding in zip(memories, embeddings):
            is_duplicate = False
            
            for i, (existing, existing_emb) in enumerate(zip(unique, unique_embeddings)):
                similarity = self._cosine_similarity(embedding, existing_emb)
                
                if similarity >= self.similarity_threshold:
                    # Keep higher confidence
                    if memory.confidence > existing.confidence:
                        for group in duplicate_groups:
                            if existing in group:
                                group.append(memory)
                                break
                        else:
                            duplicate_groups.append([existing, memory])
                        unique[i] = memory
                        unique_embeddings[i] = embedding
                    else:
                        for group in duplicate_groups:
                            if existing in group:
                                group.append(memory)
                                break
                        else:
                            duplicate_groups.append([existing, memory])
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(memory)
                unique_embeddings.append(embedding)
        
        return unique
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def find_duplicates(
        self,
        new_memory: Memory,
        existing_memories: list[Memory]
    ) -> list[Memory]:
        """
        Find existing memories that are duplicates of a new memory.
        
        Args:
            new_memory: The new memory to check
            existing_memories: List of existing memories
            
        Returns:
            List of existing memories that are duplicates
        """
        duplicates = []
        
        for existing in existing_memories:
            if existing.domain != new_memory.domain:
                continue
            
            if self.strategy == "exact":
                if (existing.key.lower() == new_memory.key.lower() and
                    existing.value.lower() == new_memory.value.lower()):
                    duplicates.append(existing)
            
            elif self.strategy == "semantic" and self.embedding_fn:
                text1 = f"{new_memory.key}: {new_memory.value}"
                text2 = f"{existing.key}: {existing.value}"
                emb1 = self.embedding_fn(text1)
                emb2 = self.embedding_fn(text2)
                if self._cosine_similarity(emb1, emb2) >= self.similarity_threshold:
                    duplicates.append(existing)
            
            else:  # fuzzy
                if new_memory.similar_to(existing, self.similarity_threshold):
                    duplicates.append(existing)
        
        return duplicates
