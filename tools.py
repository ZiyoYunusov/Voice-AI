"""Функции (tools) для LLM function calling.

Регистрируются как `LLMContext(tools=[get_menu, check_delivery_time, create_order])`.
Pipecat строит JSON-схему из докстринга и типов аргументов (кроме первого
параметра `params`) — работает одинаково с GoogleLLMService (Gemini) и
AnthropicLLMService (Claude), без ручных FunctionSchema.

APIClient и CallLogger берутся из `params.app_resources` (передаётся в
PipelineWorker(..., app_resources=...) в bot.py), а не как аргументы функции —
иначе они попали бы в JSON-схему, которую видит LLM.
"""

from typing import Any

from loguru import logger

from api_client import APIClient, DjangoAPIError
from pipecat.services.llm_service import FunctionCallParams


def _api_client(params: FunctionCallParams) -> APIClient:
    return params.app_resources["api_client"]


def _log_call(params: FunctionCallParams, result: Any) -> None:
    call_logger = params.app_resources.get("call_logger")
    if call_logger is not None:
        call_logger.add_function_call(params.function_name, dict(params.arguments), result)


async def get_menu(params: FunctionCallParams):
    """Получить список блюд меню с ценами.

    Вызывай в начале разговора и каждый раз, когда клиент спрашивает, что есть
    в меню. Никогда не придумывай блюда или цены — используй только этот список.
    """
    try:
        menu = await _api_client(params).get_menu()
        payload = {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "name_uz": item.name_uz,
                    "category": item.category,
                    "price": item.price,
                    "description": item.description,
                }
                for item in menu
            ]
        }
    except DjangoAPIError as e:
        logger.error(f"get_menu: ошибка API: {e}")
        payload = {"error": "menu_unavailable", "message": str(e)}

    _log_call(params, payload)
    await params.result_callback(payload)


async def check_delivery_time(params: FunctionCallParams, address: str):
    """Узнать ориентировочное время доставки по адресу.

    Args:
        address: Полный адрес доставки клиента, как он его назвал (город, улица, дом).
    """
    try:
        minutes = await _api_client(params).check_delivery_time(address)
        payload = {"estimated_minutes": minutes}
    except DjangoAPIError as e:
        logger.error(f"check_delivery_time: ошибка API: {e}")
        payload = {"error": "delivery_estimate_unavailable", "message": str(e)}

    _log_call(params, payload)
    await params.result_callback(payload)


async def create_order(
    params: FunctionCallParams,
    items: list[dict],
    address: str,
    phone: str,
    delivery_time: str,
):
    """Создать заказ. Вызывай только после того, как клиент вслух подтвердил весь заказ целиком.

    Args:
        items: Список блюд заказа. Каждый элемент — объект с полями item_id
            (идентификатор блюда из get_menu, например "plov") и quantity
            (количество, целое число больше нуля).
        address: Полный адрес доставки.
        phone: Контактный телефон клиента.
        delivery_time: Желаемое время доставки в свободной форме, например
            "как можно скорее" или "к 19:00".
    """
    try:
        result = await _api_client(params).create_order(
            items=items, address=address, phone=phone, delivery_time=delivery_time
        )
        payload = {
            "order_id": result.order_id,
            "status": result.status,
            "total_price": result.total_price,
            "estimated_delivery_minutes": result.estimated_delivery_minutes,
        }
    except DjangoAPIError as e:
        logger.error(f"create_order: ошибка API: {e}")
        payload = {"error": "order_creation_failed", "message": str(e)}

    _log_call(params, payload)
    await params.result_callback(payload)
