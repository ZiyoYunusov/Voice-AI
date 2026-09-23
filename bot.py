"""Голосовой бот приёма заказов еды: transport -> STT -> LLM(+tools) -> TTS -> transport.

Запуск: python bot.py -t webrtc, затем открыть http://localhost:7860/client в браузере.
Язык и голос берутся из BOT_LANGUAGE (ru|uz) в .env, LLM-провайдер — из LLM_PROVIDER.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

if sys.platform == "win32":
    # Дев-раннер Pipecat печатает баннер с юникодными рамками. На консоли с
    # кодировкой по умолчанию (cp1251/cp866) это падает с UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from api_client import APIClient
from call_logger import CallLogger
from llm_factory import create_llm_service
from retry_handler import RateLimitRetryHandler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, LLMRunFrame, TranscriptionFrame, TTSTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from tools import check_delivery_time, create_order, get_menu

load_dotenv(override=True)

PROMPTS_DIR = Path(__file__).parent / "prompts"

LANGUAGE_CONFIG = {
    "ru": {
        "azure_language": Language.RU_RU,
        "voice_env": "AZURE_VOICE_RU",
        "voice_default": "ru-RU-SvetlanaNeural",
        "prompt_file": "system_ru.md",
        "greeting": "Здравствуйте! Это служба доставки еды. Что бы вы хотели заказать?",
        "retry_phrase": "Секунду, уточняю.",
        "giveup_phrase": "Извините, у меня сейчас технические неполадки. "
        "Пожалуйста, перезвоните через пару минут.",
    },
    "uz": {
        "azure_language": Language.UZ_UZ,
        "voice_env": "AZURE_VOICE_UZ",
        "voice_default": "uz-UZ-MadinaNeural",
        "prompt_file": "system_uz.md",
        "greeting": "Assalomu alaykum! Ovqat yetkazib berish xizmati. Nima buyurtma qilmoqchisiz?",
        "retry_phrase": "Bir soniya, aniqlab olay.",
        "giveup_phrase": "Kechirasiz, hozir texnik nosozlik yuz berdi. "
        "Iltimos, bir necha daqiqadan so'ng qayta qo'ng'iroq qiling.",
    },
}


def get_language_config() -> dict:
    language = os.getenv("BOT_LANGUAGE", "ru").strip().lower()
    if language not in LANGUAGE_CONFIG:
        logger.warning(f"Неизвестный BOT_LANGUAGE={language!r}, использую 'ru'")
        language = "ru"
    return {"code": language, **LANGUAGE_CONFIG[language]}


class TranscriptRecorderProcessor(FrameProcessor):
    """Записывает финальные реплики пользователя и бота в CallLogger."""

    def __init__(self, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._call_logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._call_logger.add_transcript_entry("user", frame.text)
        elif isinstance(frame, TTSTextFrame):
            self._call_logger.add_transcript_entry("assistant", frame.text)

        await self.push_frame(frame, direction)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime. Only "webrtc" is wired up for this demo; add a
# "twilio" entry here later to support phone calls without touching run_bot().
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    lang = get_language_config()
    logger.info(f"Starting bot: язык={lang['code']}, LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'gemini')}")

    system_instruction = (PROMPTS_DIR / lang["prompt_file"]).read_text(encoding="utf-8")

    stt = AzureSTTService(
        api_key=os.environ["AZURE_SPEECH_API_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"],
        settings=AzureSTTService.Settings(language=lang["azure_language"]),
    )

    tts = AzureTTSService(
        api_key=os.environ["AZURE_SPEECH_API_KEY"],
        region=os.environ["AZURE_SPEECH_REGION"],
        voice=os.getenv(lang["voice_env"], lang["voice_default"]),
        settings=AzureTTSService.Settings(language=lang["azure_language"]),
    )

    llm = create_llm_service(system_instruction)

    api_client = APIClient()
    call_logger = CallLogger(language=lang["code"], llm_provider=os.getenv("LLM_PROVIDER", "gemini"))

    context = LLMContext(tools=[get_menu, check_delivery_time, create_order])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    retry_handler = RateLimitRetryHandler(
        tts=tts, retry_phrase=lang["retry_phrase"], giveup_phrase=lang["giveup_phrase"]
    )
    # Два отдельных экземпляра: TranscriptionFrame не доходит дальше user_aggregator,
    # а TTSTextFrame рождается только после tts — один процессор не может стоять
    # в обеих точках пайплайна одновременно, поэтому пишем в общий call_logger из двух мест.
    user_transcript_recorder = TranscriptRecorderProcessor(call_logger=call_logger)
    assistant_transcript_recorder = TranscriptRecorderProcessor(call_logger=call_logger)

    pipeline = Pipeline(
        [
            transport.input(),  # Вход от клиента (аудио)
            stt,  # Речь -> текст
            user_transcript_recorder,  # Пишет финальные реплики клиента в лог
            user_aggregator,  # Собирает контекст диалога, VAD/перебивание
            retry_handler,  # Ретраит LLM при 429 перед тем как ошибка уйдёт выше
            llm,  # Function calling: get_menu / check_delivery_time / create_order
            tts,  # Текст -> речь
            transport.output(),  # Выход клиенту (аудио)
            assistant_transcript_recorder,  # Пишет финальные реплики бота в лог (TTS-текст)
            assistant_aggregator,  # Дописывает ответ ассистента и tool-контекст
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        app_resources={"api_client": api_client, "call_logger": call_logger},
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Начни разговор с этого приветствия дословно: "
                    f"\"{lang['greeting']}\""
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        call_logger.finalize()
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
