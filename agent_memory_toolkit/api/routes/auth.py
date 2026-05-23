"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..models import TokenRequest, TokenResponse, ErrorResponse
from ..auth import authenticate_user, create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
    },
    summary="Get access token",
    description="Authenticate with username and password to receive a JWT access token.",
)
async def login(request: TokenRequest) -> TokenResponse:
    """
    Authenticate and get a JWT access token.
    
    The token should be included in the Authorization header for subsequent requests:
    `Authorization: Bearer <token>`
    """
    if not authenticate_user(request.username, request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token, expires_in = create_access_token(request.username)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
    )
