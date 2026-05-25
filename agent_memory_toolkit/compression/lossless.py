"""Lossless Memory Compression - Reduce storage footprint without data loss.

This module provides lossless compression techniques for memory storage,
enabling efficient storage while maintaining full data fidelity.

Compression algorithms:
- ZLIB: Fast, good compression ratio (default)
- GZIP: Compatible, slightly larger overhead
- BROTLI: Best compression, slower but excellent for storage
- LZ4: Ultra-fast, lower compression ratio
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, TypeVar, Union

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


class CompressionAlgorithm(str, Enum):
    """Available lossless compression algorithms."""
    ZLIB = "zlib"       # Good balance of speed and compression
    GZIP = "gzip"       # Standard, compatible
    BROTLI = "brotli"   # Best compression ratio
    LZ4 = "lz4"         # Fastest, lower ratio
    NONE = "none"       # No compression


@dataclass
class CompressionStats:
    """Statistics about a compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: CompressionAlgorithm
    compression_time_ms: float
    checksum: str
    
    @property
    def space_saved(self) -> int:
        """Bytes saved by compression."""
        return self.original_size - self.compressed_size
    
    @property
    def space_saved_percent(self) -> float:
        """Percentage of space saved."""
        if self.original_size == 0:
            return 0.0
        return (self.space_saved / self.original_size) * 100


@dataclass
class CompressedMemory:
    """A compressed memory blob with metadata."""
    memory_id: str
    compressed_data: bytes
    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "compressed_data": base64.b64encode(self.compressed_data).decode("utf-8"),
            "algorithm": self.algorithm.value,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressedMemory":
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            compressed_data=base64.b64decode(data["compressed_data"]),
            algorithm=CompressionAlgorithm(data["algorithm"]),
            original_size=data["original_size"],
            compressed_size=data["compressed_size"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class LosslessCompressor(ABC):
    """Abstract base class for lossless compressors."""
    
    @property
    @abstractmethod
    def algorithm(self) -> CompressionAlgorithm:
        """The compression algorithm used."""
        ...
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        ...
    
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        ...


class ZlibCompressor(LosslessCompressor):
    """ZLIB compression - good balance of speed and compression."""
    
    def __init__(self, level: int = 6):
        """Initialize ZLIB compressor.
        
        Args:
            level: Compression level (1-9, default 6)
        """
        self.level = max(1, min(9, level))
    
    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.ZLIB
    
    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=self.level)
    
    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class GzipCompressor(LosslessCompressor):
    """GZIP compression - standard, widely compatible."""
    
    def __init__(self, level: int = 6):
        """Initialize GZIP compressor.
        
        Args:
            level: Compression level (1-9, default 6)
        """
        self.level = max(1, min(9, level))
    
    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.GZIP
    
    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)
    
    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class BrotliCompressor(LosslessCompressor):
    """Brotli compression - best compression ratio, ideal for storage.
    
    Requires: pip install brotli
    """
    
    def __init__(self, quality: int = 6, lgwin: int = 22):
        """Initialize Brotli compressor.
        
        Args:
            quality: Quality level (0-11, default 6)
            lgwin: Window size log (10-24, default 22)
        """
        if not BROTLI_AVAILABLE:
            raise ImportError(
                "Brotli is not installed. Install with: pip install brotli"
            )
        self.quality = max(0, min(11, quality))
        self.lgwin = max(10, min(24, lgwin))
    
    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.BROTLI
    
    def compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=self.quality, lgwin=self.lgwin)
    
    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)


class Lz4Compressor(LosslessCompressor):
    """LZ4 compression - ultra-fast, ideal for real-time use.
    
    Requires: pip install lz4
    """
    
    def __init__(self, level: int = 0):
        """Initialize LZ4 compressor.
        
        Args:
            level: Compression level (0-16, default 0 for fastest)
        """
        if not LZ4_AVAILABLE:
            raise ImportError(
                "LZ4 is not installed. Install with: pip install lz4"
            )
        self.level = max(0, min(16, level))
    
    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.LZ4
    
    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=self.level)
    
    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class NoCompressor(LosslessCompressor):
    """No compression - pass-through for testing/comparison."""
    
    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.NONE
    
    def compress(self, data: bytes) -> bytes:
        return data
    
    def decompress(self, data: bytes) -> bytes:
        return data


@dataclass
class MemoryCompressionConfig:
    """Configuration for memory compression."""
    
    # Algorithm selection
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB
    compression_level: int = 6
    
    # Minimum size to compress (small data may expand with compression)
    min_size_bytes: int = 100
    
    # Auto-select algorithm based on data characteristics
    auto_select: bool = False
    
    # Verify integrity after decompression
    verify_checksum: bool = True
    
    # Store metadata alongside compressed data
    include_metadata: bool = True


class MemoryCompressor:
    """Intelligent memory compression engine.
    
    Provides lossless compression for memory data with automatic
    algorithm selection, integrity verification, and statistics tracking.
    
    Example:
        >>> compressor = MemoryCompressor()
        >>> 
        >>> # Compress memory data
        >>> memory_data = {"content": "Important memory content...", "tags": ["work"]}
        >>> compressed = compressor.compress_memory(memory_data, memory_id="mem_123")
        >>> 
        >>> print(f"Compressed from {compressed.original_size} to {compressed.compressed_size}")
        >>> 
        >>> # Decompress when needed
        >>> restored = compressor.decompress_memory(compressed)
        >>> assert restored == memory_data
    
    Advanced usage:
        >>> # Use Brotli for maximum compression
        >>> config = MemoryCompressionConfig(
        ...     algorithm=CompressionAlgorithm.BROTLI,
        ...     compression_level=11,  # Maximum compression
        ... )
        >>> compressor = MemoryCompressor(config=config)
        >>> 
        >>> # Or use LZ4 for speed
        >>> config = MemoryCompressionConfig(
        ...     algorithm=CompressionAlgorithm.LZ4,
        ...     compression_level=0,  # Fastest
        ... )
    """
    
    def __init__(self, config: Optional[MemoryCompressionConfig] = None):
        """Initialize the memory compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config or MemoryCompressionConfig()
        self._compressors: dict[CompressionAlgorithm, LosslessCompressor] = {}
        self._stats_history: list[CompressionStats] = []
        
        # Initialize compressor for configured algorithm
        self._init_compressor(self.config.algorithm)
    
    def _init_compressor(self, algorithm: CompressionAlgorithm) -> LosslessCompressor:
        """Get or create a compressor for the given algorithm."""
        if algorithm in self._compressors:
            return self._compressors[algorithm]
        
        compressor: LosslessCompressor
        level = self.config.compression_level
        
        if algorithm == CompressionAlgorithm.ZLIB:
            compressor = ZlibCompressor(level=level)
        elif algorithm == CompressionAlgorithm.GZIP:
            compressor = GzipCompressor(level=level)
        elif algorithm == CompressionAlgorithm.BROTLI:
            compressor = BrotliCompressor(quality=level)
        elif algorithm == CompressionAlgorithm.LZ4:
            compressor = Lz4Compressor(level=level)
        else:
            compressor = NoCompressor()
        
        self._compressors[algorithm] = compressor
        return compressor
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def _select_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Auto-select the best algorithm for the data."""
        if not self.config.auto_select:
            return self.config.algorithm
        
        # For small data, use fast algorithms
        if len(data) < 1024:
            return CompressionAlgorithm.ZLIB
        
        # For large data, prefer better compression
        if len(data) > 100_000:
            if BROTLI_AVAILABLE:
                return CompressionAlgorithm.BROTLI
            return CompressionAlgorithm.ZLIB
        
        # Default to configured algorithm
        return self.config.algorithm
    
    def _serialize_memory(self, memory: Any) -> bytes:
        """Serialize memory data to bytes."""
        if isinstance(memory, bytes):
            return memory
        elif isinstance(memory, str):
            return memory.encode("utf-8")
        else:
            return json.dumps(memory, default=str, ensure_ascii=False).encode("utf-8")
    
    def _deserialize_memory(
        self, 
        data: bytes, 
        original_type: Optional[str] = None
    ) -> Any:
        """Deserialize bytes back to memory data."""
        if original_type == "bytes":
            return data
        elif original_type == "str":
            return data.decode("utf-8")
        else:
            return json.loads(data.decode("utf-8"))
    
    def compress_memory(
        self,
        memory: Any,
        memory_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CompressedMemory:
        """Compress a memory object.
        
        Args:
            memory: Memory data (dict, str, or bytes)
            memory_id: Unique identifier for the memory
            metadata: Optional metadata to store with compressed data
            
        Returns:
            CompressedMemory object
        """
        import time
        
        start_time = time.perf_counter()
        
        # Serialize to bytes
        data = self._serialize_memory(memory)
        original_size = len(data)
        
        # Store original type for deserialization
        mem_metadata = metadata.copy() if metadata else {}
        if isinstance(memory, bytes):
            mem_metadata["_original_type"] = "bytes"
        elif isinstance(memory, str):
            mem_metadata["_original_type"] = "str"
        else:
            mem_metadata["_original_type"] = "json"
        
        # Check if compression is worthwhile
        if original_size < self.config.min_size_bytes:
            algorithm = CompressionAlgorithm.NONE
            compressed_data = data
        else:
            # Select and apply compression
            algorithm = self._select_algorithm(data)
            compressor = self._init_compressor(algorithm)
            compressed_data = compressor.compress(data)
            
            # If compression didn't help, store uncompressed
            if len(compressed_data) >= original_size:
                algorithm = CompressionAlgorithm.NONE
                compressed_data = data
        
        compressed_size = len(compressed_data)
        checksum = self._compute_checksum(data)
        
        end_time = time.perf_counter()
        compression_time_ms = (end_time - start_time) * 1000
        
        # Track statistics
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size if original_size > 0 else 1.0,
            algorithm=algorithm,
            compression_time_ms=compression_time_ms,
            checksum=checksum,
        )
        self._stats_history.append(stats)
        
        return CompressedMemory(
            memory_id=memory_id,
            compressed_data=compressed_data,
            algorithm=algorithm,
            original_size=original_size,
            compressed_size=compressed_size,
            checksum=checksum,
            metadata=mem_metadata,
        )
    
    def decompress_memory(
        self,
        compressed: CompressedMemory,
        verify: Optional[bool] = None,
    ) -> Any:
        """Decompress a memory object.
        
        Args:
            compressed: CompressedMemory object
            verify: Override checksum verification setting
            
        Returns:
            Original memory data
            
        Raises:
            ValueError: If checksum verification fails
        """
        should_verify = verify if verify is not None else self.config.verify_checksum
        
        # Decompress
        if compressed.algorithm == CompressionAlgorithm.NONE:
            data = compressed.compressed_data
        else:
            compressor = self._init_compressor(compressed.algorithm)
            data = compressor.decompress(compressed.compressed_data)
        
        # Verify integrity
        if should_verify:
            actual_checksum = self._compute_checksum(data)
            if actual_checksum != compressed.checksum:
                raise ValueError(
                    f"Checksum mismatch for memory {compressed.memory_id}: "
                    f"expected {compressed.checksum}, got {actual_checksum}"
                )
        
        # Deserialize
        original_type = compressed.metadata.get("_original_type")
        return self._deserialize_memory(data, original_type)
    
    def compress_batch(
        self,
        memories: list[tuple[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[CompressedMemory]:
        """Compress multiple memories.
        
        Args:
            memories: List of (memory_id, memory_data) tuples
            metadata: Optional shared metadata
            
        Returns:
            List of CompressedMemory objects
        """
        return [
            self.compress_memory(mem_data, mem_id, metadata)
            for mem_id, mem_data in memories
        ]
    
    def decompress_batch(
        self,
        compressed_list: list[CompressedMemory],
        verify: Optional[bool] = None,
    ) -> list[tuple[str, Any]]:
        """Decompress multiple memories.
        
        Args:
            compressed_list: List of CompressedMemory objects
            verify: Override checksum verification setting
            
        Returns:
            List of (memory_id, memory_data) tuples
        """
        return [
            (cm.memory_id, self.decompress_memory(cm, verify))
            for cm in compressed_list
        ]
    
    def estimate_compression(
        self,
        data: Any,
        algorithms: Optional[list[CompressionAlgorithm]] = None,
    ) -> dict[CompressionAlgorithm, CompressionStats]:
        """Estimate compression for different algorithms.
        
        Useful for selecting the best algorithm for your data characteristics.
        
        Args:
            data: Sample data to compress
            algorithms: Algorithms to test (default: all available)
            
        Returns:
            Dictionary mapping algorithm to compression stats
        """
        import time
        
        serialized = self._serialize_memory(data)
        original_size = len(serialized)
        
        if algorithms is None:
            algorithms = [CompressionAlgorithm.ZLIB, CompressionAlgorithm.GZIP]
            if BROTLI_AVAILABLE:
                algorithms.append(CompressionAlgorithm.BROTLI)
            if LZ4_AVAILABLE:
                algorithms.append(CompressionAlgorithm.LZ4)
        
        results: dict[CompressionAlgorithm, CompressionStats] = {}
        
        for algo in algorithms:
            compressor = self._init_compressor(algo)
            
            start_time = time.perf_counter()
            compressed = compressor.compress(serialized)
            end_time = time.perf_counter()
            
            results[algo] = CompressionStats(
                original_size=original_size,
                compressed_size=len(compressed),
                compression_ratio=len(compressed) / original_size if original_size > 0 else 1.0,
                algorithm=algo,
                compression_time_ms=(end_time - start_time) * 1000,
                checksum=self._compute_checksum(serialized),
            )
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        if not self._stats_history:
            return {
                "total_operations": 0,
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                "total_saved_bytes": 0,
                "average_compression_ratio": 0.0,
                "average_compression_time_ms": 0.0,
            }
        
        total_original = sum(s.original_size for s in self._stats_history)
        total_compressed = sum(s.compressed_size for s in self._stats_history)
        
        return {
            "total_operations": len(self._stats_history),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "total_saved_bytes": total_original - total_compressed,
            "average_compression_ratio": total_compressed / total_original if total_original > 0 else 0.0,
            "average_compression_time_ms": sum(s.compression_time_ms for s in self._stats_history) / len(self._stats_history),
            "algorithms_used": {
                algo.value: sum(1 for s in self._stats_history if s.algorithm == algo)
                for algo in CompressionAlgorithm
                if any(s.algorithm == algo for s in self._stats_history)
            },
        }
    
    def clear_stats(self) -> None:
        """Clear compression statistics."""
        self._stats_history.clear()


# Convenience functions

def compress_memory(
    memory: Any,
    memory_id: str,
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB,
) -> CompressedMemory:
    """Compress a single memory.
    
    Convenience function for one-off compression.
    
    Args:
        memory: Memory data to compress
        memory_id: Unique identifier
        algorithm: Compression algorithm to use
        
    Returns:
        CompressedMemory object
    """
    config = MemoryCompressionConfig(algorithm=algorithm)
    compressor = MemoryCompressor(config=config)
    return compressor.compress_memory(memory, memory_id)


def decompress_memory(compressed: CompressedMemory) -> Any:
    """Decompress a compressed memory.
    
    Convenience function for one-off decompression.
    
    Args:
        compressed: CompressedMemory object
        
    Returns:
        Original memory data
    """
    compressor = MemoryCompressor()
    return compressor.decompress_memory(compressed)
