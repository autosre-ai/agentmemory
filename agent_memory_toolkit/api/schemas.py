"""Request/response schemas for the REST API.

This module re-exports all Pydantic models from models.py for API schema compatibility.
Provides both request schemas (input validation) and response schemas (output serialization).
"""

from .models import (
    # Authentication
    TokenRequest,
    TokenResponse,
    # Memory
    MemoryMetadataModel,
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    SearchResultModel,
    MemoryListResponse,
    SearchResponse,
    # Branch
    BranchCreate,
    BranchResponse,
    BranchListResponse,
    # Commit
    CommitCreate,
    CommitResponse,
    CommitListResponse,
    # Extraction
    ExtractionRequest,
    ExtractedMemoryModel,
    ExtractionResponse,
    # Security
    SecurityCheckRequest,
    SecurityCheckResponse,
    # Compression
    MessageModel,
    CompressionRequest,
    CompressionResponse,
    # Health & Info
    HealthResponse,
    InfoResponse,
    # Errors
    ErrorResponse,
    ValidationErrorResponse,
)

from pydantic import BaseModel, Field
from typing import Optional


# ==============================================================================
# Hybrid Search Schemas (additional POST /memories/search request body)
# ==============================================================================

class HybridSearchRequest(BaseModel):
    """Request for hybrid search combining FTS and vector similarity."""
    
    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    fts_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for FTS (BM25) scoring")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector similarity scoring")
    include_deleted: bool = Field(default=False, description="Include soft-deleted memories")
    rerank: bool = Field(default=False, description="Use cross-encoder reranking for improved accuracy")
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Number of candidates for reranking")
    method: str = Field(default="auto", description="Search method: auto, fts, vector, or hybrid")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the capital of France?",
                "limit": 10,
                "fts_weight": 0.4,
                "vector_weight": 0.6,
                "method": "hybrid",
                "rerank": False
            }
        }
    }


class HybridSearchResponse(BaseModel):
    """Response from hybrid search."""
    
    results: list[SearchResultModel] = Field(..., description="Search results with scores")
    query: str = Field(..., description="Original search query")
    method: str = Field(..., description="Search method used")
    count: int = Field(..., description="Number of results returned")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [],
                "query": "What is the capital of France?",
                "method": "hybrid",
                "count": 0
            }
        }
    }


__all__ = [
    # Authentication
    "TokenRequest",
    "TokenResponse",
    # Memory
    "MemoryMetadataModel",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryResponse",
    "SearchResultModel",
    "MemoryListResponse",
    "SearchResponse",
    # Branch
    "BranchCreate",
    "BranchResponse",
    "BranchListResponse",
    # Commit
    "CommitCreate",
    "CommitResponse",
    "CommitListResponse",
    # Extraction
    "ExtractionRequest",
    "ExtractedMemoryModel",
    "ExtractionResponse",
    # Security
    "SecurityCheckRequest",
    "SecurityCheckResponse",
    # Compression
    "MessageModel",
    "CompressionRequest",
    "CompressionResponse",
    # Health & Info
    "HealthResponse",
    "InfoResponse",
    # Errors
    "ErrorResponse",
    "ValidationErrorResponse",
    # Hybrid Search
    "HybridSearchRequest",
    "HybridSearchResponse",
]
