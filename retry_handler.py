"""Ретраит вызов LLM при rate-limit (HTTP 429) с экспоненциальной задержкой.

Ставится в пайплайн прямо перед LLM-сервисом (между user-aggregator'ом и LLM).
Ошибки LLM всплывают вверх по пайплайну как ErrorFrame (см.
FrameProcessor.push_error в pipecat) — этот процессор первым видит такой
апстрим-фрейм, поскольку стоит перед LLM. Pipecat сам классифицирует причину
ошибки в frame.category (ErrorCategory.RATE_LIMIT для 429 у любого
провайдера), поэтому логика одинаковая для Gemini и Claude.

При обнаружении rate-limit: озвучивает клиенту фразу вроде «секунду, уточняю»
через tts.queue_frame(), ждёт экспоненциальную задержку и повторно запускает
LLM через pipeline_worker.queue_frame(LLMRunFrame()) — тот же механизм,
которым bot.py стартует разговор.

Счётчик попыток должен сбрасываться на каждую новую реплику клиента, а не
только при успехе (иначе одна неудачная серия ретраев "съест" лимит для всех
следующих реплик). TranscriptionFrame для этого не годится — user-aggregator
поглощает его и не пробрасывает дальше по пайплайну. Вместо этого ловим
LLMContextFrame — именно его user-aggregator пускает вниз по пайплайну перед
каждым запуском LLM, что при "органическом" ходе разговора, что при нашем
собственном ретрае. Различаем эти два случая простым флагом: перед тем как
самим инициировать ретрай, выставляем _retry_in_flight, и на LLMContextFrame,
вызванном этим ретраем, флаг снимаем, не трогая счётчик; любой другой
LLMContextFrame — это новая реплика клиента, счётчик сбрасывается.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, LLMContextFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.utils.errors import ErrorCategory


class RateLimitRetryHandler(FrameProcessor):
    """Слушает ErrorFrame(RATE_LIMIT) от следующего в пайплайне LLM и ретраит его."""

    def __init__(
        self,
        tts: TTSService,
        retry_phrase: str,
        giveup_phrase: str,
        max_retries: int = 3,
        base_delay_secs: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tts = tts
        self._retry_phrase = retry_phrase
        self._giveup_phrase = giveup_phrase
        self._max_retries = max_retries
        self._base_delay_secs = base_delay_secs
        self._attempt = 0
        self._retry_in_flight = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            if self._retry_in_flight:
                self._retry_in_flight = False
            else:
                # Новый запуск LLM, не наш собственный ретрай — новая реплика клиента.
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
                if self._attempt == 1:
                    # Озвучиваем только на первой попытке — иначе на серии
                    # ретраев клиент слышит одну и ту же фразу несколько раз подряд.
                    await self._tts.queue_frame(TTSSpeakFrame(self._retry_phrase))
                await asyncio.sleep(delay)
                self._retry_in_flight = True
                await self.pipeline_worker.queue_frame(LLMRunFrame())
                return  # не пробрасываем эту ошибку дальше вверх по пайплайну

            logger.error(
                f"LLM rate limit (429): превышено число попыток ({self._max_retries}), "
                "сообщаю клиенту и завершаю попытки для этой реплики"
            )
            self._attempt = 0
            await self._tts.queue_frame(TTSSpeakFrame(self._giveup_phrase))
            return  # тоже не пробрасываем — клиент уже получил внятный ответ

        await self.push_frame(frame, direction)
