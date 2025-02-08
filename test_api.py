import pytest
from httpx import AsyncClient, ASGITransport
from basic_api import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/")
    
    assert response.status_code == 200
    assert "Welcome to Conversion Prediction API!" in response.text


@pytest.mark.asyncio
async def test_predict_valid():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/predict", json={"days": 7})
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 7
    assert "model_metrics" in data
    assert "mape" in data["model_metrics"]
    assert "rmse" in data["model_metrics"]


@pytest.mark.asyncio
async def test_predict_invalid_days():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/predict", json={"days": 0})
    
    assert response.status_code == 422  # FastAPI will return an automatic validation error


@pytest.mark.asyncio
async def test_predict_large_days():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/predict", json={"days": 365})  # 1 year prediction request
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 365


@pytest.mark.asyncio
async def test_invalid_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/invalid")
    
    assert response.status_code == 404

# In order to run it:
# pytest test_api.py -v
