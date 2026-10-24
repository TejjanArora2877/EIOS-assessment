import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db

client = TestClient(app)

def test_health_endpoint():
    init_db()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "storage_status" in data
    assert data["storage_status"]["database_connected"] is True
    assert data["storage_status"]["total_documents"] == 0
    assert data["storage_status"]["total_assessments"] == 0

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
