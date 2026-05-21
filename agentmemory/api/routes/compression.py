"""Compression routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..models import (
    CompressionRequest,
    CompressionResponse,
    MessageModel,
)
from ..auth import get_current_user

router = APIRouter(prefix="/compress", tags=["Compression"])


def get_context_compressor():
    """Lazy import ContextCompressor."""
    from agentmemory.compression import ContextCompressor, CompressionConfig, CompressionMode
    return ContextCompressor, CompressionConfig, CompressionMode


@router.post(
    "/conversation",
    response_model=CompressionResponse,
    summary="Compress a conversation",
    description="Compress a conversation to fit within a token budget.",
)
async def compress_conversation(
    request: CompressionRequest,
    current_user: str = Depends(get_current_user),
) -> CompressionResponse:
    """Compress a conversation."""
    ContextCompressor, CompressionConfig, CompressionMode = get_context_compressor()
    
    mode_map = {
        "aggressive": CompressionMode.AGGRESSIVE,
        "balanced": CompressionMode.BALANCED,
        "conservative": CompressionMode.CONSERVATIVE,
        "lossless": CompressionMode.LOSSLESS,
    }
    
    compression_mode = mode_map.get(request.mode)
    if compression_mode is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Must be one of: {list(mode_map.keys())}",
        )
    
    config = CompressionConfig(
        max_tokens=request.max_tokens,
        reserve_tokens=request.reserve_tokens,
        mode=compression_mode,
    )
    
    # Convert to format expected by compressor
    messages_data = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]
    
    compressor = ContextCompressor(config=config)
    result = compressor.compress(messages_data)
    
    # Result messages are already dicts
    compressed_messages = [
        MessageModel(role=m.get("role", "user"), content=m.get("content", ""))
        for m in result.messages
    ]
    
    return CompressionResponse(
        messages=compressed_messages,
        original_tokens=result.original_tokens,
        compressed_tokens=result.compressed_tokens,
        compression_ratio=result.compression_ratio,
        strategy_used=result.strategy_used,
        tokens_saved=result.tokens_saved,
    )


@router.post(
    "/estimate",
    summary="Estimate token count",
    description="Estimate token count for a list of messages.",
)
async def estimate_tokens(
    messages: list[MessageModel],
    current_user: str = Depends(get_current_user),
):
    """Estimate token count for messages."""
    from agentmemory.compression import TokenCounter
    
    counter = TokenCounter()
    total_tokens = 0
    
    for msg in messages:
        tokens = counter.count(msg.content)
        total_tokens += tokens
    
    return {
        "message_count": len(messages),
        "total_tokens": total_tokens,
        "average_tokens_per_message": total_tokens / len(messages) if messages else 0,
    }


@router.get(
    "/modes",
    summary="List compression modes",
    description="Get available compression modes and their descriptions.",
)
async def list_compression_modes(
    current_user: str = Depends(get_current_user),
):
    """List available compression modes."""
    return {
        "modes": [
            {
                "name": "aggressive",
                "description": "Maximum compression, may lose some context",
            },
            {
                "name": "balanced",
                "description": "Balanced compression (recommended)",
            },
            {
                "name": "conservative",
                "description": "Conservative compression, preserves more context",
            },
            {
                "name": "lossless",
                "description": "No lossy compression, only removes obvious redundancy",
            },
        ],
        "default": "balanced",
    }
