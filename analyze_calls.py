"""Считает метрики по логам звонков в logs/*.json.

Запуск: python analyze_calls.py
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    # На консоли с кодировкой по умолчанию (cp1251/cp866) кириллица в print()
    # может превращаться в "битые" символы вместо ошибки — принудительно
    # переключаемся на UTF-8, чтобы вывод отображался корректно.
    sys.stdout.reconfigure(encoding="utf-8")

LOGS_DIR = Path(__file__).parent / "logs"


def load_calls() -> list[dict[str, Any]]:
    calls = []
    for path in sorted(LOGS_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                calls.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Пропускаю {path.name}: не удалось прочитать ({e})")
    return calls


def last_assistant_question(call: dict[str, Any]) -> str:
    """Последняя реплика бота перед тем, как разговор закончился без заказа.

    Используется как приблизительная "точка отказа" — что бот спросил или
    сказал последним, если клиент не дошёл до заказа.
    """
    for entry in reversed(call.get("transcript", [])):
        if entry["role"] == "assistant":
            return entry["text"]
    return "(нет реплик ассистента)"


def print_report(calls: list[dict[str, Any]]) -> None:
    if not calls:
        print(f"В {LOGS_DIR} пока нет логов звонков (logs/*.json). Сделайте тестовый звонок.")
        return

    total = len(calls)
    outcomes = Counter(call["outcome"] for call in calls)
    orders_created = outcomes.get("order_created", 0)
    conversion_pct = round(100 * orders_created / total, 1)

    durations = [call["duration_secs"] for call in calls]
    avg_duration = round(statistics.mean(durations), 1)

    print("=" * 60)
    print(f"Всего звонков: {total}")
    print(f"Заказов создано: {orders_created} ({conversion_pct}% конверсия)")
    print(f"Средняя длительность звонка: {avg_duration}с")
    print()

    print("Распределение по итогу звонка:")
    for outcome, count in outcomes.most_common():
        pct = round(100 * count / total, 1)
        print(f"  {outcome:<15} {count:>3}  ({pct}%)")
    print()

    non_converted = [c for c in calls if c["outcome"] != "order_created"]
    if non_converted:
        print(f"Частые точки отказа (последняя реплика бота в {len(non_converted)} "
              f"незавершённых звонках):")
        failure_points = Counter(last_assistant_question(c) for c in non_converted)
        for text, count in failure_points.most_common(5):
            preview = text if len(text) <= 70 else text[:67] + "..."
            print(f'  [{count}x] "{preview}"')
        print()

    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_provider[call.get("llm_provider", "unknown")].append(call)

    if len(by_provider) > 1:
        print("Сравнение LLM-провайдеров:")
        for provider, provider_calls in sorted(by_provider.items()):
            p_total = len(provider_calls)
            p_orders = sum(1 for c in provider_calls if c["outcome"] == "order_created")
            p_conv = round(100 * p_orders / p_total, 1)
            p_avg_dur = round(statistics.mean(c["duration_secs"] for c in provider_calls), 1)
            print(
                f"  {provider:<12} звонков={p_total:<4} конверсия={p_conv}%  "
                f"средняя длительность={p_avg_dur}с"
            )
        print()

    by_language: dict[str, int] = Counter(call.get("language", "unknown") for call in calls)
    print("Звонков по языку: " + ", ".join(f"{lang}={count}" for lang, count in by_language.items()))
    print("=" * 60)


if __name__ == "__main__":
    print_report(load_calls())
