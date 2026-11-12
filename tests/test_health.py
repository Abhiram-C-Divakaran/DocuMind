import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert data["components"]["database"] == "healthy"
    assert data["components"]["storage"] == "healthy"
    assert data["phase"] in [1, "2A", "2a"]

