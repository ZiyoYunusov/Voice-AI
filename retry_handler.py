"""Ретраит вызов LLM при rate-limit (HTTP 429) с экспоненциальной задержкой.

Ставится в пайплайн прямо перед LLM-сервисом. Ошибки LLM всплывают вверх по
пайплайну как ErrorFrame (см. FrameProcessor.push_error в pipecat) — процессор
перед LLM первым видит такой апстрим-фрейм. Pipecat сам классифицирует причину
ошибки в frame.category (ErrorCategory.RATE_LIMIT для 429 у любого провайдера),
поэтому логика одинаковая для Gemini и Claude.

При обнаружении rate-limit: озвучивает клиенту фразу вроде «секунду, уточняю»
через tts.queue_frame(), ждёт экспоненциальную задержку и повторно запускает
LLM через pipeline_worker.queue_frame(LLMRunFrame()) — это тот же механизм,
которым bot.py стартует разговор.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, LLMRunFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.utils.errors import ErrorCategory


class RateLimitRetryHandler(FrameProcessor):
    """Слушает ErrorFrame(RATE_LIMIT) от следующего в пайплайне LLM и ретраит его."""

    def __init__(
        self,
        tts: TTSService,
        retry_phrase: str,
        max_retries: int = 3,
        base_delay_secs: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tts = tts
        self._retry_phrase = retry_phrase
        self._max_retries = max_retries
        self._base_delay_secs = base_delay_secs
        self._attempt = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Новая реплика клиента — счётчик ретраев на предыдущий ответ больше не актуален.
            self._attempt = 0

        if (
            direction == FrameDirection.UPSTREAM
            and isinstance(frame, ErrorFrame)
            and frame.category == ErrorCategory.RATE_LIMIT
        ):
            if self._attempt < self._max_retries and self.pipeline_worker is not None:
                self._attempt += 1
                delay = self._base_delay_secs * (2 ** (self._attempt - 1))
                logger.warning(
                    f"LLM rate limit (429), попытка {self._attempt}/{self._max_retries}, "
                    f"повтор через {delay}с: {frame.error}"
                )
                await self._tts.queue_frame(TTSSpeakFrame(self._retry_phrase))
                await asyncio.sleep(delay)
                await self.pipeline_worker.queue_frame(LLMRunFrame())
                return  # не пробрасываем эту ошибку дальше вверх по пайплайну

            logger.error(
                f"LLM rate limit (429): превышено число попыток ({self._max_retries}), "
                "пробрасываю ошибку дальше"
            )

        await self.push_frame(frame, direction)
