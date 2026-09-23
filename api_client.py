"""Клиент для Django API заказов с mock-режимом на data/menu.json."""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

MENU_PATH = Path(__file__).parent / "data" / "menu.json"


class DjangoAPIError(Exception):
    """Ошибка при обращении к Django API (недоступен, таймаут, плохой ответ)."""


@dataclass
class MenuItem:
    id: str
    name: str
    name_uz: str
    category: str
    price: int
    description: str


@dataclass
class OrderResult:
    order_id: str
    status: str
    total_price: int
    estimated_delivery_minutes: int
    items: list[dict[str, Any]] = field(default_factory=list)


def _load_mock_menu() -> list[MenuItem]:
    with open(MENU_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [MenuItem(**item) for item in raw]


class APIClient:
    """Обёртка над Django API. Без DJANGO_API_URL работает в mock-режиме."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout_secs: float = 8.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url or os.getenv("DJANGO_API_URL") or None
        self.token = token or os.getenv("DJANGO_API_TOKEN") or None
        self.timeout_secs = timeout_secs
        self.max_retries = max_retries
        self.mock_mode = self.base_url is None
        if self.mock_mode:
            logger.info("APIClient: DJANGO_API_URL не задан, работаю в mock-режиме")
            self._mock_menu = _load_mock_menu()
        else:
            logger.info(f"APIClient: работаю с Django API на {self.base_url}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _request_with_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Запрос к Django API с ретраями и экспоненциальной задержкой.

        Ретраится на таймаутах, сетевых ошибках и 429/5xx. Другие 4xx — сразу ошибка.
        """
        assert self.base_url is not None
        url = f"{self.base_url.rstrip('/')}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_secs) as client:
                    response = await client.request(
                        method, url, headers=self._headers(), **kwargs
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = DjangoAPIError(
                        f"{method} {path} -> HTTP {response.status_code}"
                    )
                    raise last_error
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, DjangoAPIError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        f"Django API запрос не удался (попытка {attempt + 1}/"
                        f"{self.max_retries}): {e}. Повтор через {delay}с"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Django API недоступен после {self.max_retries} попыток: {e}")

        raise DjangoAPIError(f"Django API недоступен: {last_error}") from last_error

    async def get_menu(self) -> list[MenuItem]:
        """Возвращает список блюд меню."""
        if self.mock_mode:
            return self._mock_menu

        response = await self._request_with_retry("GET", "/api/menu/")
        data = response.json()
        return [MenuItem(**item) for item in data]

    async def check_delivery_time(self, address: str) -> int:
        """Возвращает ориентировочное время доставки в минутах для адреса."""
        if self.mock_mode:
            # Простая эвристика для демо: время зависит от длины адреса.
            base = 30
            extra = min(len(address) % 20, 20)
            return base + extra

        response = await self._request_with_retry(
            "POST", "/api/delivery-estimate/", json={"address": address}
        )
        data = response.json()
        return int(data["estimated_minutes"])

    async def create_order(
        self,
        items: list[dict[str, Any]],
        address: str,
        phone: str,
        delivery_time: str,
    ) -> OrderResult:
        """Создаёт заказ. items: [{"item_id": str, "quantity": int}, ...]."""
        if self.mock_mode:
            menu_by_id = {item.id: item for item in self._mock_menu}
            total = 0
            enriched_items = []
            for entry in items:
                menu_item = menu_by_id.get(entry["item_id"])
                if menu_item is None:
                    raise DjangoAPIError(f"Неизвестное блюдо: {entry['item_id']}")
                quantity = int(entry["quantity"])
                total += menu_item.price * quantity
                enriched_items.append(
                    {
                        "item_id": menu_item.id,
                        "name": menu_item.name,
                        "quantity": quantity,
                        "price": menu_item.price,
                    }
                )
            order_id = f"mock-{uuid.uuid4().hex[:8]}"
            logger.info(
                f"[MOCK] Заказ создан: {order_id}, адрес={address}, телефон={phone}, "
                f"время={delivery_time}, сумма={total}"
            )
            return OrderResult(
                order_id=order_id,
                status="created",
                total_price=total,
                estimated_delivery_minutes=45,
                items=enriched_items,
            )

        payload = {
            "items": items,
            "address": address,
            "phone": phone,
            "delivery_time": delivery_time,
        }
        response = await self._request_with_retry("POST", "/api/orders/", json=payload)
        data = response.json()
        return OrderResult(
            order_id=data["order_id"],
            status=data["status"],
            total_price=data["total_price"],
            estimated_delivery_minutes=data["estimated_delivery_minutes"],
            items=data.get("items", []),
        )
