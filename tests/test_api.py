"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient
from md_platform.api.app import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "2.0.0"}
