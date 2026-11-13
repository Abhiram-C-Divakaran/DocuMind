import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(anon_client: AsyncClient):
    response = await anon_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["redis"] in ("connected", "unavailable")
    assert data["worker"] in ("available", "unknown", "unavailable")
    assert isinstance(data["tesseract_available"], bool)
    assert isinstance(data["layoutlmv3_ready"], bool)
    assert isinstance(data["layoutlmv3_is_production_ready"], bool)
