import pytest

from api_client import APIClient, DjangoAPIError


@pytest.fixture
def client():
    return APIClient(base_url=None)


def test_mock_mode_enabled_without_django_url(client):
    assert client.mock_mode is True


@pytest.mark.asyncio
async def test_get_menu_returns_items_from_mock(client):
    menu = await client.get_menu()
    assert len(menu) >= 10
    assert all(item.price > 0 for item in menu)
    ids = {item.id for item in menu}
    assert "plov" in ids


@pytest.mark.asyncio
async def test_check_delivery_time_returns_positive_minutes(client):
    minutes = await client.check_delivery_time("Ташкент, Чиланзар 5")
    assert minutes > 0


@pytest.mark.asyncio
async def test_create_order_computes_total_price(client):
    result = await client.create_order(
        items=[{"item_id": "plov", "quantity": 2}, {"item_id": "non", "quantity": 1}],
        address="Ташкент, Юнусабад 10",
        phone="+998901234567",
        delivery_time="через 40 минут",
    )
    assert result.status == "created"
    assert result.total_price == 35000 * 2 + 6000
    assert result.order_id.startswith("mock-")
    assert len(result.items) == 2


@pytest.mark.asyncio
async def test_create_order_unknown_item_raises(client):
    with pytest.raises(DjangoAPIError):
        await client.create_order(
            items=[{"item_id": "does-not-exist", "quantity": 1}],
            address="Ташкент",
            phone="+998901234567",
            delivery_time="сейчас",
        )


@pytest.mark.asyncio
async def test_real_mode_retries_and_raises_on_persistent_failure(monkeypatch):
    client = APIClient(base_url="http://example.invalid", max_retries=2, timeout_secs=0.1)

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("api_client.asyncio.sleep", fake_sleep)

    with pytest.raises(DjangoAPIError):
        await client.get_menu()
