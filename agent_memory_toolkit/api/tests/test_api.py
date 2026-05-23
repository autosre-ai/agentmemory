"""Tests for the REST API."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import tempfile
import os

# Conditionally import based on whether fastapi is installed
try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def test_config(temp_db):
    """Create a test configuration."""
    from agent_memory_toolkit.api.config import APIConfig
    return APIConfig(
        db_path=temp_db,
        jwt_secret_key="test-secret-key-for-testing",
        jwt_expiration_minutes=60,
        rate_limit_enabled=False,  # Disable rate limiting for tests
    )


@pytest.fixture
def app(test_config):
    """Create a test FastAPI app."""
    from agent_memory_toolkit.api import create_app, set_config
    from agent_memory_toolkit.api.dependencies import reset_memory_store
    
    # Reset any existing store
    reset_memory_store()
    
    set_config(test_config)
    app = create_app(test_config)
    yield app
    
    # Cleanup
    reset_memory_store()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get authentication headers with a valid token."""
    response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealthAndInfo:
    """Tests for health and info endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "modules" in data
        assert "store" in data["modules"]


class TestAuthentication:
    """Tests for authentication endpoints."""
    
    def test_get_token_success(self, client):
        """Test successful token generation."""
        response = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "admin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_get_token_invalid_credentials(self, client):
        """Test token generation with invalid credentials."""
        response = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get("/api/v1/memories")
        assert response.status_code == 401  # Unauthorized
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401


class TestMemoryOperations:
    """Tests for memory CRUD operations."""
    
    def test_create_memory(self, client, auth_headers):
        """Test creating a new memory."""
        response = client.post(
            "/api/v1/memories",
            json={
                "content": "Test memory content",
                "metadata": {
                    "source": "test",
                    "confidence": 0.95,
                    "tags": ["test", "example"],
                }
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "Test memory content"
        assert "id" in data
        assert data["metadata"]["confidence"] == 0.95
    
    def test_list_memories(self, client, auth_headers):
        """Test listing memories."""
        # Create a memory first
        client.post(
            "/api/v1/memories",
            json={"content": "Memory for listing"},
            headers=auth_headers,
        )
        
        response = client.get("/api/v1/memories", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert len(data["memories"]) > 0
    
    def test_get_memory(self, client, auth_headers):
        """Test getting a specific memory."""
        # Create a memory
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "Memory to get"},
            headers=auth_headers,
        )
        memory_id = create_response.json()["id"]
        
        # Get the memory
        response = client.get(f"/api/v1/memories/{memory_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Memory to get"
    
    def test_get_memory_not_found(self, client, auth_headers):
        """Test getting a non-existent memory."""
        response = client.get(
            "/api/v1/memories/non-existent-id",
            headers=auth_headers,
        )
        assert response.status_code == 404
    
    def test_update_memory(self, client, auth_headers):
        """Test updating a memory."""
        # Create a memory
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "Original content"},
            headers=auth_headers,
        )
        memory_id = create_response.json()["id"]
        
        # Update the memory
        response = client.put(
            f"/api/v1/memories/{memory_id}",
            json={"content": "Updated content"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated content"
        assert data["version"] == 2
    
    def test_delete_memory(self, client, auth_headers):
        """Test deleting a memory."""
        # Create a memory
        create_response = client.post(
            "/api/v1/memories",
            json={"content": "Memory to delete"},
            headers=auth_headers,
        )
        memory_id = create_response.json()["id"]
        
        # Delete the memory
        response = client.delete(
            f"/api/v1/memories/{memory_id}",
            headers=auth_headers,
        )
        assert response.status_code == 204
        
        # Verify it's deleted
        get_response = client.get(
            f"/api/v1/memories/{memory_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 404
    
    def test_search_memories(self, client, auth_headers):
        """Test searching memories."""
        # Create memories
        client.post(
            "/api/v1/memories",
            json={"content": "The capital of France is Paris"},
            headers=auth_headers,
        )
        client.post(
            "/api/v1/memories",
            json={"content": "The capital of Germany is Berlin"},
            headers=auth_headers,
        )
        
        # Search
        response = client.get(
            "/api/v1/memories/search?q=France",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "France"


class TestBranchOperations:
    """Tests for branch operations."""
    
    def test_list_branches(self, client, auth_headers):
        """Test listing branches."""
        response = client.get("/api/v1/branches", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "branches" in data
        assert "current" in data
        assert data["current"] == "main"
    
    def test_create_branch(self, client, auth_headers):
        """Test creating a new branch."""
        response = client.post(
            "/api/v1/branches",
            json={"name": "feature-branch"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "feature-branch"
    
    def test_checkout_branch(self, client, auth_headers):
        """Test checking out a branch."""
        # Create a branch
        client.post(
            "/api/v1/branches",
            json={"name": "checkout-test"},
            headers=auth_headers,
        )
        
        # Checkout the branch
        response = client.post(
            "/api/v1/branches/checkout-test/checkout",
            headers=auth_headers,
        )
        assert response.status_code == 200


class TestExtraction:
    """Tests for extraction endpoints."""
    
    def test_extract_from_text(self, client, auth_headers):
        """Test extracting memories from text."""
        response = client.post(
            "/api/v1/extract/text",
            json={
                "text": "My name is John and I work at Google.",
                "mode": "rule",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "method" in data
        assert "processing_time_ms" in data


class TestSecurity:
    """Tests for security endpoints."""
    
    def test_security_check_safe(self, client, auth_headers):
        """Test security check with safe content."""
        response = client.post(
            "/api/v1/security/check",
            json={
                "content": "This is safe content about programming.",
                "level": "medium",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_safe" in data
        assert "validation_time_ms" in data
    
    def test_list_security_levels(self, client, auth_headers):
        """Test listing security levels."""
        response = client.get("/api/v1/security/levels", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "levels" in data
        assert "default" in data


class TestCompression:
    """Tests for compression endpoints."""
    
    def test_compress_conversation(self, client, auth_headers):
        """Test compressing a conversation."""
        response = client.post(
            "/api/v1/compress/conversation",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
                    {"role": "user", "content": "Can you help me with Python?"},
                ],
                "max_tokens": 1000,
                "mode": "balanced",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert "original_tokens" in data
        assert "compressed_tokens" in data
        assert "compression_ratio" in data
    
    def test_estimate_tokens(self, client, auth_headers):
        """Test estimating tokens."""
        response = client.post(
            "/api/v1/compress/estimate",
            json=[
                {"role": "user", "content": "Hello, world!"},
            ],
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_tokens" in data
        assert "message_count" in data
    
    def test_list_compression_modes(self, client, auth_headers):
        """Test listing compression modes."""
        response = client.get("/api/v1/compress/modes", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "modes" in data
        assert "default" in data


class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_headers(self, temp_db):
        """Test that rate limit headers are present."""
        from agent_memory_toolkit.api.config import APIConfig
        from agent_memory_toolkit.api import create_app, set_config
        from agent_memory_toolkit.api.dependencies import reset_memory_store
        
        reset_memory_store()
        
        config = APIConfig(
            db_path=temp_db,
            jwt_secret_key="test-secret",
            rate_limit_enabled=True,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
        )
        set_config(config)
        app = create_app(config)
        client = TestClient(app)
        
        # Get a token
        response = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "admin"},
        )
        token = response.json()["access_token"]
        
        # Make a request
        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": f"Bearer {token}"},
        )
        
        # Check headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        
        reset_memory_store()


class TestJWTAuth:
    """Tests for JWT authentication utilities."""
    
    def test_create_and_verify_token(self, test_config):
        """Test creating and verifying a token."""
        from agent_memory_toolkit.api.auth import create_access_token, verify_token
        
        token, expires_in = create_access_token("testuser", config=test_config)
        assert token
        assert expires_in > 0
        
        payload = verify_token(token, config=test_config)
        assert payload["sub"] == "testuser"
    
    def test_expired_token(self, test_config):
        """Test that expired tokens are rejected."""
        from agent_memory_toolkit.api.auth import create_access_token, verify_token, AuthenticationError
        from datetime import timedelta
        
        # Create a token that's already expired
        token, _ = create_access_token(
            "testuser",
            config=test_config,
            expires_delta=timedelta(seconds=-1),
        )
        
        with pytest.raises(AuthenticationError, match="expired"):
            verify_token(token, config=test_config)
    
    def test_invalid_token(self, test_config):
        """Test that invalid tokens are rejected."""
        from agent_memory_toolkit.api.auth import verify_token, AuthenticationError
        
        with pytest.raises(AuthenticationError, match="Invalid"):
            verify_token("invalid-token", config=test_config)


class TestOpenAPI:
    """Tests for OpenAPI documentation."""
    
    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_endpoint(self, client):
        """Test that docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test that redoc endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
