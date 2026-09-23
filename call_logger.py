"""Логирование звонков в logs/*.json: транскрипт, вызовы функций, итог, длительность."""

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

LOGS_DIR = Path(__file__).parent / "logs"

CallOutcome = Literal["order_created", "declined", "hangup", "in_progress"]


@dataclass
class TranscriptEntry:
    role: Literal["user", "assistant"]
    text: str
    timestamp: float


@dataclass
class FunctionCallEntry:
    name: str
    arguments: dict[str, Any]
    result: Any
    timestamp: float


class CallLogger:
    """Собирает события одного звонка и сохраняет их в JSON после завершения."""

    def __init__(self, language: str, llm_provider: str):
        self.call_id = f"call-{uuid.uuid4().hex[:12]}"
        self.language = language
        self.llm_provider = llm_provider
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.outcome: CallOutcome = "in_progress"
        self.transcript: list[TranscriptEntry] = []
        self.function_calls: list[FunctionCallEntry] = []

    def add_transcript_entry(self, role: Literal["user", "assistant"], text: str) -> None:
        if not text.strip():
            return
        self.transcript.append(TranscriptEntry(role=role, text=text, timestamp=time.time()))

    def add_function_call(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        self.function_calls.append(
            FunctionCallEntry(name=name, arguments=arguments, result=result, timestamp=time.time())
        )
        if name == "create_order" and isinstance(result, dict) and result.get("status") == "created":
            self.outcome = "order_created"

    def set_outcome(self, outcome: CallOutcome) -> None:
        self.outcome = outcome

    def finalize(self) -> Path:
        """Сохраняет накопленные данные звонка в logs/<call_id>.json и возвращает путь."""
        self.ended_at = time.time()
        if self.outcome == "in_progress":
            # Заказ не создан. Если был содержательный диалог (клиент что-то
            # обсуждал, а не просто сразу пропала связь) — считаем это отказом,
            # иначе — обрывом. Эвристика для демо, без отдельного сигнала от LLM.
            self.outcome = "declined" if len(self.transcript) >= 4 else "hangup"

        LOGS_DIR.mkdir(exist_ok=True)
        payload = {
            "call_id": self.call_id,
            "language": self.language,
            "llm_provider": self.llm_provider,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_secs": round(self.ended_at - self.started_at, 2),
            "outcome": self.outcome,
            "transcript": [asdict(e) for e in self.transcript],
            "function_calls": [asdict(e) for e in self.function_calls],
        }

        path = LOGS_DIR / f"{self.call_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Звонок {self.call_id} завершён: outcome={self.outcome}, "
            f"длительность={payload['duration_secs']}с, лог={path}"
        )
        return path
