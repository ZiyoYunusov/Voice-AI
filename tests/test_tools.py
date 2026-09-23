from unittest.mock import MagicMock

import pytest

import tools
from api_client import APIClient
from pipecat.services.llm_service import FunctionCallParams


def make_params(api_client: APIClient) -> tuple[FunctionCallParams, list]:
    results = []

    async def result_callback(value, **kwargs):
        results.append(value)

    params = FunctionCallParams(
        function_name="test",
        tool_call_id="call-1",
        arguments={},
        llm=MagicMock(),
        pipeline_worker=MagicMock(),
        context=MagicMock(),
        result_callback=result_callback,
        app_resources={"api_client": api_client},
    )
    return params, results


@pytest.fixture
def api_client():
    return APIClient(base_url=None)


async def test_get_menu_calls_result_callback_with_items(api_client):
    params, results = make_params(api_client)
    await tools.get_menu(params)
    assert len(results) == 1
    assert "items" in results[0]
    assert len(results[0]["items"]) >= 10


async def test_check_delivery_time_returns_estimate(api_client):
    params, results = make_params(api_client)
    await tools.check_delivery_time(params, address="Ташкент, Мирзо-Улугбек")
    assert "estimated_minutes" in results[0]
    assert results[0]["estimated_minutes"] > 0


async def test_create_order_returns_order_id(api_client):
    params, results = make_params(api_client)
    await tools.create_order(
        params,
        items=[{"item_id": "somsa", "quantity": 3}],
        address="Ташкент, Себзар",
        phone="+998901112233",
        delivery_time="к 20:00",
    )
    assert results[0]["status"] == "created"
    assert results[0]["total_price"] == 12000 * 3


async def test_create_order_unknown_item_returns_error_not_exception(api_client):
    params, results = make_params(api_client)
    await tools.create_order(
        params,
        items=[{"item_id": "unknown-dish", "quantity": 1}],
        address="Ташкент",
        phone="+998901112233",
        delivery_time="сейчас",
    )
    assert "error" in results[0]
