"""Pydantic models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# Authentication Models
# ==============================================================================

class TokenRequest(BaseModel):
    """Request for JWT token."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


# ==============================================================================
# Memory Models
# ==============================================================================

class MemoryMetadataModel(BaseModel):
    """Memory metadata."""
    source: Optional[str] = Field(default=None, description="Source of the memory")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MemoryCreate(BaseModel):
    """Request to create a new memory."""
    content: str = Field(..., min_length=1, description="Memory content")
    metadata: Optional[MemoryMetadataModel] = Field(default=None, description="Memory metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "The capital of France is Paris",
                "metadata": {
                    "source": "knowledge-base",
                    "confidence": 0.95,
                    "tags": ["geography", "facts"]
                }
            }
        }
    }


class MemoryUpdate(BaseModel):
    """Request to update a memory."""
    content: Optional[str] = Field(default=None, min_length=1, description="New content")
    metadata: Optional[MemoryMetadataModel] = Field(default=None, description="New metadata")


class MemoryResponse(BaseModel):
    """Memory response."""
    id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    metadata: MemoryMetadataModel = Field(..., description="Memory metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: int = Field(..., description="Version number")
    is_deleted: bool = Field(default=False, description="Soft deletion flag")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "The capital of France is Paris",
                "metadata": {
                    "source": "knowledge-base",
                    "confidence": 0.95,
                    "tags": ["geography", "facts"],
                    "extra": {}
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "version": 1,
                "is_deleted": False
            }
        }
    }


class SearchResultModel(BaseModel):
    """Search result with score."""
    memory: MemoryResponse = Field(..., description="The matched memory")
    score: float = Field(..., description="Relevance score")
    match_type: str = Field(..., description="Type of match (fts, vector, hybrid)")


class MemoryListResponse(BaseModel):
    """List of memories response."""
    memories: list[MemoryResponse] = Field(..., description="List of memories")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class SearchResponse(BaseModel):
    """Search results response."""
    results: list[SearchResultModel] = Field(..., description="Search results")
    query: str = Field(..., description="Search query")
    count: int = Field(..., description="Number of results")


# ==============================================================================
# Branch Models
# ==============================================================================

class BranchCreate(BaseModel):
    """Request to create a branch."""
    name: str = Field(..., min_length=1, max_length=100, description="Branch name")
    from_branch: Optional[str] = Field(default=None, description="Source branch to copy from")


class BranchResponse(BaseModel):
    """Branch response."""
    name: str = Field(..., description="Branch name")
    head_commit_id: Optional[str] = Field(default=None, description="Head commit ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_active: bool = Field(..., description="Whether branch is active")


class BranchListResponse(BaseModel):
    """List of branches response."""
    branches: list[BranchResponse] = Field(..., description="List of branches")
    current: str = Field(..., description="Current branch name")


# ==============================================================================
# Commit Models
# ==============================================================================

class CommitCreate(BaseModel):
    """Request to create a commit."""
    message: str = Field(..., min_length=1, max_length=500, description="Commit message")


class CommitResponse(BaseModel):
    """Commit response."""
    id: str = Field(..., description="Commit ID")
    branch: str = Field(..., description="Branch name")
    parent_id: Optional[str] = Field(default=None, description="Parent commit ID")
    message: str = Field(..., description="Commit message")
    created_at: datetime = Field(..., description="Creation timestamp")


class CommitListResponse(BaseModel):
    """List of commits response."""
    commits: list[CommitResponse] = Field(..., description="List of commits")
    branch: str = Field(..., description="Branch name")


# ==============================================================================
# Extraction Models
# ==============================================================================

class ExtractionRequest(BaseModel):
    """Request to extract memories from text."""
    text: str = Field(..., min_length=1, description="Text to extract from")
    mode: str = Field(default="rule", description="Extraction mode: rule, llm, hybrid")
    source: Optional[str] = Field(default=None, description="Source identifier")


class ExtractedMemoryModel(BaseModel):
    """Extracted memory."""
    domain: str = Field(..., description="Cognitive domain")
    key: str = Field(..., description="Memory key")
    value: str = Field(..., description="Memory value")
    confidence: float = Field(..., description="Confidence score")


class ExtractionResponse(BaseModel):
    """Extraction results response."""
    memories: list[ExtractedMemoryModel] = Field(..., description="Extracted memories")
    method: str = Field(..., description="Extraction method used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ==============================================================================
# Security Models
# ==============================================================================

class SecurityCheckRequest(BaseModel):
    """Request to check content for security issues."""
    content: str = Field(..., min_length=1, description="Content to validate")
    level: str = Field(default="medium", description="Security level: minimal, low, medium, high, paranoid")


class SecurityCheckResponse(BaseModel):
    """Security check results."""
    is_safe: bool = Field(..., description="Whether content is safe")
    rejection_reason: Optional[str] = Field(default=None, description="Rejection reason if not safe")
    adjusted_confidence: float = Field(..., description="Adjusted confidence score")
    validation_time_ms: float = Field(..., description="Validation time in milliseconds")
    detected_patterns: list[str] = Field(default_factory=list, description="Detected security patterns")


# ==============================================================================
# Compression Models
# ==============================================================================

class MessageModel(BaseModel):
    """Chat message for compression."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class CompressionRequest(BaseModel):
    """Request to compress a conversation."""
    messages: list[MessageModel] = Field(..., min_length=1, description="Messages to compress")
    max_tokens: int = Field(default=4000, gt=0, description="Maximum token budget")
    reserve_tokens: int = Field(default=500, ge=0, description="Reserve tokens for response")
    mode: str = Field(default="balanced", description="Compression mode: aggressive, balanced, conservative, lossless")


class CompressionResponse(BaseModel):
    """Compression results response."""
    messages: list[MessageModel] = Field(..., description="Compressed messages")
    original_tokens: int = Field(..., description="Original token count")
    compressed_tokens: int = Field(..., description="Compressed token count")
    compression_ratio: float = Field(..., description="Compression ratio")
    strategy_used: str = Field(..., description="Compression strategy used")
    tokens_saved: int = Field(..., description="Tokens saved")


# ==============================================================================
# Health & Info Models
# ==============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")


class InfoResponse(BaseModel):
    """API information response."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    modules: list[str] = Field(..., description="Available modules")


# ==============================================================================
# Error Models
# ==============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Any] = Field(default=None, description="Additional error details")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(default="Request validation failed", description="Error message")
    detail: list[dict[str, Any]] = Field(..., description="Validation error details")
