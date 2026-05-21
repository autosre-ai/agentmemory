"""Extraction routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractedMemoryModel,
    ErrorResponse,
)
from ..auth import get_current_user

router = APIRouter(prefix="/extract", tags=["Extraction"])


def get_memory_extractor():
    """Lazy import MemoryExtractor."""
    from agentmemory.extraction import MemoryExtractor
    return MemoryExtractor


@router.post(
    "/text",
    response_model=ExtractionResponse,
    summary="Extract memories from text",
    description="Extract structured memories from plain text using rule-based, LLM, or hybrid extraction.",
)
async def extract_from_text(
    request: ExtractionRequest,
    current_user: str = Depends(get_current_user),
) -> ExtractionResponse:
    """Extract memories from text."""
    MemoryExtractor = get_memory_extractor()
    
    # Validate mode
    valid_modes = ["rule", "llm", "hybrid"]
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Must be one of: {valid_modes}",
        )
    
    extractor = MemoryExtractor(mode=request.mode)
    result = extractor.extract(request.text, source=request.source)
    
    return ExtractionResponse(
        memories=[
            ExtractedMemoryModel(
                domain=m.domain.value,
                key=m.key,
                value=m.value,
                confidence=m.confidence,
            )
            for m in result.memories
        ],
        method=result.method,
        processing_time_ms=result.processing_time_ms,
    )


@router.post(
    "/conversation",
    response_model=ExtractionResponse,
    summary="Extract memories from conversation",
    description="Extract structured memories from a list of conversation messages.",
)
async def extract_from_conversation(
    messages: list[dict],
    mode: str = "rule",
    source: str | None = None,
    current_user: str = Depends(get_current_user),
) -> ExtractionResponse:
    """Extract memories from a conversation."""
    MemoryExtractor = get_memory_extractor()
    
    # Validate mode
    valid_modes = ["rule", "llm", "hybrid"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{mode}'. Must be one of: {valid_modes}",
        )
    
    # Combine messages into text
    conversation_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in messages
    )
    
    extractor = MemoryExtractor(mode=mode)
    result = extractor.extract(
        conversation_text,
        source=source or "api:conversation",
    )
    
    return ExtractionResponse(
        memories=[
            ExtractedMemoryModel(
                domain=m.domain.value,
                key=m.key,
                value=m.value,
                confidence=m.confidence,
            )
            for m in result.memories
        ],
        method=result.method,
        processing_time_ms=result.processing_time_ms,
    )
