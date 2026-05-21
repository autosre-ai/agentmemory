"""Branch operations routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..models import (
    BranchCreate,
    BranchResponse,
    BranchListResponse,
    CommitCreate,
    CommitResponse,
    CommitListResponse,
    ErrorResponse,
)
from ..auth import get_current_user
from ..dependencies import get_memory_store

router = APIRouter(prefix="/branches", tags=["Branches"])


@router.get(
    "",
    response_model=BranchListResponse,
    summary="List branches",
    description="List all branches in the memory store.",
)
async def list_branches(
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> BranchListResponse:
    """List all branches."""
    branches = store.list_branches()
    current = store.current_branch
    
    return BranchListResponse(
        branches=[
            BranchResponse(
                name=b.name,
                head_commit_id=b.head_commit_id,
                created_at=b.created_at,
                is_active=b.is_active,
            )
            for b in branches
        ],
        current=current,
    )


@router.post(
    "",
    response_model=BranchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a branch",
    description="Create a new branch, optionally from an existing branch.",
)
async def create_branch(
    request: BranchCreate,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> BranchResponse:
    """Create a new branch."""
    try:
        branch = store.create_branch(request.name, from_branch=request.from_branch)
        return BranchResponse(
            name=branch.name,
            head_commit_id=branch.head_commit_id,
            created_at=branch.created_at,
            is_active=branch.is_active,
        )
    except Exception as e:
        if "exists" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Branch '{request.name}' already exists",
            )
        raise


@router.post(
    "/{branch_name}/checkout",
    response_model=BranchResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Branch not found"},
    },
    summary="Checkout a branch",
    description="Switch to a different branch.",
)
async def checkout_branch(
    branch_name: str,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> BranchResponse:
    """Checkout (switch to) a branch."""
    try:
        store.checkout(branch_name)
        branches = store.list_branches()
        branch = next((b for b in branches if b.name == branch_name), None)
        
        if branch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Branch '{branch_name}' not found",
            )
        
        return BranchResponse(
            name=branch.name,
            head_commit_id=branch.head_commit_id,
            created_at=branch.created_at,
            is_active=branch.is_active,
        )
    except HTTPException:
        raise
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Branch '{branch_name}' not found",
            )
        raise


@router.delete(
    "/{branch_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Branch not found"},
        409: {"model": ErrorResponse, "description": "Cannot delete current branch"},
    },
    summary="Delete a branch",
    description="Delete a branch. Cannot delete the current branch.",
)
async def delete_branch(
    branch_name: str,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> None:
    """Delete a branch."""
    if branch_name == store.current_branch:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete the current branch",
        )
    
    try:
        store.delete_branch(branch_name)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Branch '{branch_name}' not found",
            )
        raise


# ==============================================================================
# Commit Operations
# ==============================================================================

@router.get(
    "/{branch_name}/commits",
    response_model=CommitListResponse,
    summary="List commits",
    description="List commits for a branch.",
)
async def list_commits(
    branch_name: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> CommitListResponse:
    """List commits for a branch."""
    # Temporarily switch to branch to get commits
    original_branch = store.current_branch
    try:
        if branch_name != original_branch:
            store.checkout(branch_name)
        
        commits = store.get_history(limit=limit)
        
        return CommitListResponse(
            commits=[
                CommitResponse(
                    id=c.id,
                    branch=c.branch,
                    parent_id=c.parent_id,
                    message=c.message,
                    created_at=c.created_at,
                )
                for c in commits
            ],
            branch=branch_name,
        )
    finally:
        if branch_name != original_branch:
            store.checkout(original_branch)


@router.post(
    "/{branch_name}/commits",
    response_model=CommitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a commit",
    description="Create a new commit on the branch.",
)
async def create_commit(
    branch_name: str,
    request: CommitCreate,
    store=Depends(get_memory_store),
    current_user: str = Depends(get_current_user),
) -> CommitResponse:
    """Create a new commit."""
    original_branch = store.current_branch
    try:
        if branch_name != original_branch:
            store.checkout(branch_name)
        
        commit = store.commit(request.message)
        
        return CommitResponse(
            id=commit.id,
            branch=commit.branch,
            parent_id=commit.parent_id,
            message=commit.message,
            created_at=commit.created_at,
        )
    finally:
        if branch_name != original_branch:
            store.checkout(original_branch)
