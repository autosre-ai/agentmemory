"""Memory operations routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..models import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemoryMetadataModel,
    MemoryListResponse,
    SearchResponse,
    SearchResultModel,
    ErrorResponse,
)
from ..auth import get_current_user
from ..dependencies import get_memory_store

router = APIRouter(prefix="/memories", tags=["Memories"])


def memory_to_response(memory) -> MemoryResponse:
    """Convert a Memory object to MemoryResponse."""
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        metadata=MemoryMetadataModel(
            source=memory.metadata.source,
            confidence=memory.metadata.confidence,
            tags=memory.metadata.tags,
            extra=memory.metadata.extra,
        ),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        version=memory.version,
        is_deleted=memory.is_deleted,
    )


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new memory",
    description="Add a new memory to the store with optional metadata.",
)
async def create_memory(
    request: MemoryCreate,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> MemoryResponse:
    """Create a new memory."""
    metadata = None
    if request.metadata:
        metadata = {
            "source": request.metadata.source or f"api:{current_user}",
            "confidence": request.metadata.confidence,
            "tags": request.metadata.tags,
            "extra": request.metadata.extra,
        }
    else:
        metadata = {"source": f"api:{current_user}"}
    
    memory = store.add(request.content, metadata=metadata)
    return memory_to_response(memory)


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List memories",
    description="List memories with pagination and optional tag filtering.",
)
async def list_memories(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    include_deleted: bool = Query(default=False, description="Include deleted memories"),
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> MemoryListResponse:
    """List memories with pagination."""
    memories = store.list(
        limit=limit,
        offset=offset,
        tag=tag,
        include_deleted=include_deleted,
    )
    total = store.count(include_deleted=include_deleted)
    
    return MemoryListResponse(
        memories=[memory_to_response(m) for m in memories],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search memories",
    description="Full-text search using FTS5 with BM25 ranking.",
)
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    include_deleted: bool = Query(default=False, description="Include deleted memories"),
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> SearchResponse:
    """Search memories using full-text search."""
    results = store.search_fts(q, limit=limit, include_deleted=include_deleted)
    
    return SearchResponse(
        results=[
            SearchResultModel(
                memory=memory_to_response(r.memory),
                score=r.score,
                match_type=r.match_type,
            )
            for r in results
        ],
        query=q,
        count=len(results),
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Memory not found"},
    },
    summary="Get a memory",
    description="Retrieve a memory by its ID.",
)
async def get_memory(
    memory_id: str,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> MemoryResponse:
    """Get a memory by ID."""
    try:
        memory = store.get(memory_id)
        return memory_to_response(memory)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )
        raise


@router.put(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Memory not found"},
    },
    summary="Update a memory",
    description="Update an existing memory's content or metadata.",
)
async def update_memory(
    memory_id: str,
    request: MemoryUpdate,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> MemoryResponse:
    """Update a memory."""
    try:
        metadata = None
        if request.metadata:
            metadata = {
                "source": request.metadata.source,
                "confidence": request.metadata.confidence,
                "tags": request.metadata.tags,
                "extra": request.metadata.extra,
            }
        
        memory = store.update(
            memory_id,
            content=request.content,
            metadata=metadata,
        )
        return memory_to_response(memory)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )
        raise


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Memory not found"},
    },
    summary="Delete a memory",
    description="Soft-delete a memory. Use hard=true for permanent deletion.",
)
async def delete_memory(
    memory_id: str,
    hard: bool = Query(default=False, description="Permanently delete"),
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> None:
    """Delete a memory."""
    try:
        store.delete(memory_id, hard=hard)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )
        raise


@router.get(
    "/{memory_id}/history",
    summary="Get memory history",
    description="Get version history for a specific memory.",
)
async def get_memory_history(
    memory_id: str,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
):
    """Get version history for a memory."""
    try:
        history = store.get_memory_history(memory_id)
        return {"memory_id": memory_id, "versions": history}
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )
        raise
