"""Tests for the deployment health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from agentmemory.api.app import create_app
from agentmemory.api.config import APIConfig


@pytest.fixture
def client():
    """Create test client with in-memory database."""
    config = APIConfig(
        db_path=":memory:",
        jwt_secret_key="test-secret-key-for-testing-only",
    )
    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints for Kubernetes probes."""

    def test_basic_health_check(self, client):
        """Test basic /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_trailing_slash(self, client):
        """Test /health/ with trailing slash."""
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_liveness_probe(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    def test_readiness_probe(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data
        assert "database" in data["checks"]

    def test_detailed_health(self, client):
        """Test detailed health check with component status."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data
        
        # Check components are present
        component_names = [c["name"] for c in data["components"]]
        assert "database" in component_names
        assert "disk" in component_names

    def test_startup_probe(self, client):
        """Test startup probe endpoint."""
        response = client.get("/health/startup")
        assert response.status_code == 200
        data = response.json()
        assert data["started"] is True
        assert "uptime_seconds" in data


class TestHealthNoAuth:
    """Verify health endpoints don't require authentication."""

    def test_health_no_auth_required(self, client):
        """Health endpoints should work without auth token."""
        endpoints = [
            "/health",
            "/health/",
            "/health/live",
            "/health/ready",
            "/health/detailed",
            "/health/startup",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return 401 or 403
            assert response.status_code not in [401, 403], f"{endpoint} requires auth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
