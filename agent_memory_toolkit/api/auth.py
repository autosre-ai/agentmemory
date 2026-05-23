"""JWT authentication for the REST API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .config import get_config, APIConfig

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=True)


class AuthenticationError(Exception):
    """Authentication error."""
    pass


def create_access_token(
    username: str,
    config: Optional[APIConfig] = None,
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, int]:
    """
    Create a JWT access token.
    
    Args:
        username: The username to encode in the token
        config: API configuration (uses global config if not provided)
        expires_delta: Optional custom expiration time
        
    Returns:
        Tuple of (token, expires_in_seconds)
    """
    if config is None:
        config = get_config()
    
    if expires_delta is None:
        expires_delta = timedelta(minutes=config.jwt_expiration_minutes)
    
    expire = datetime.utcnow() + expires_delta
    expires_in = int(expires_delta.total_seconds())
    
    payload = {
        "sub": username,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    
    token = jwt.encode(
        payload,
        config.jwt_secret_key,
        algorithm=config.jwt_algorithm,
    )
    
    return token, expires_in


def verify_token(
    token: str,
    config: Optional[APIConfig] = None,
) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: The JWT token to verify
        config: API configuration (uses global config if not provided)
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    if config is None:
        config = get_config()
    
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret_key,
            algorithms=[config.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def authenticate_user(
    username: str,
    password: str,
    config: Optional[APIConfig] = None,
) -> bool:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Password
        config: API configuration (uses global config if not provided)
        
    Returns:
        True if authentication successful, False otherwise
    """
    if config is None:
        config = get_config()
    
    stored_password = config.api_users.get(username)
    if stored_password is None:
        return False
    
    # Simple comparison - in production use proper password hashing
    return password == stored_password


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to get the current authenticated user.
    
    Args:
        credentials: HTTP Bearer credentials from request
        
    Returns:
        The username from the token
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        payload = verify_token(credentials.credentials)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


class RequireAuth:
    """
    Dependency class for requiring authentication.
    Can be extended for role-based access control.
    """
    
    def __init__(self, required_roles: Optional[list[str]] = None):
        self.required_roles = required_roles or []
    
    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        """Verify authentication and optionally check roles."""
        try:
            payload = verify_token(credentials.credentials)
            username = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Role checking can be added here
            # For now, just return the username
            return username
            
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
