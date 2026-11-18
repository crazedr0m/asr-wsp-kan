"""
Точка входа Telegram-бота (Active Learning / Human-in-the-Loop).

Реализация на ``asyncio`` + ``aiohttp`` (long polling Telegram Bot API),
без внешних зависимостей кроме ``aiohttp``. Токен берётся из переменной
окружения ``KAN_ASR_BOT_TOKEN`` (или из ``configs/bot.yaml``).

Сценарий:
1. Пользователь отправляет голосовое сообщение → оно конвертируется в WAV
   и сохраняется в ``data/real/wavs/``.
2. Бот просит прислать правильный текст ответом.
3. Текстовый ответ → ``handlers.confirm_sample`` → пара попадает в metadata.csv.
4. При накоплении ``FINETUNE_THRESHOLD`` подтверждённых пар в фоне
   запускается дообучение (``run_finetune``), polling не блокируется.
"""

import asyncio
import os

from ..utils.logging import get_logger, setup_logging
from . import handlers
from .finetune_worker import run_finetune

logger = get_logger("bot.app")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# Порог подтверждённых сэмплов для запуска дообучения
FINETUNE_THRESHOLD = 20


class TelegramBot:
    """Минимальный long-polling клиент Telegram Bot API."""

    def __init__(self, token: str, finetune_threshold: int = FINETUNE_THRESHOLD):
        self.token = token
        self.offset = 0
        self._session = None
        self.finetune_threshold = finetune_threshold
        # user_id -> filename последнего необработанного голосового сэмпла
        self.pending: dict[int, str] = {}
        self._finetune_task = None

    async def _api(self, method: str, **params):
        url = TELEGRAM_API.format(token=self.token, method=method)
        async with self._session.post(url, json=params) as resp:
            return await resp.json()

    async def _download_file(self, file_id: str) -> bytes:
        info = await self._api("getFile", file_id=file_id)
        file_path = info["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        async with self._session.get(url) as resp:
            return await resp.read()

    async def _maybe_start_finetune(self) -> None:
        """Запускает дообучение в фоне при достижении порога."""
        if self._finetune_task is not None and not self._finetune_task.done():
            return
        confirmed = handlers.count_confirmed_samples()
        if confirmed < self.finetune_threshold:
            return

        logger.info(
            "Порог достигнут (%d >= %d) — запуск дообучения в фоне",
            confirmed,
            self.finetune_threshold,
        )
        self._finetune_task = asyncio.create_task(asyncio.to_thread(run_finetune))

    async def _handle_voice(self, message: dict) -> None:
        voice = message["voice"]
        user_id = message.get("from", {}).get("id", 0)
        message_id = message.get("message_id", 0)
        chat_id = message.get("chat", {}).get("id", user_id)

        ogg_bytes = await self._download_file(voice["file_id"])
        try:
            result = handlers.handle_voice(ogg_bytes, user_id, message_id)
        except RuntimeError as e:
            logger.error("Ошибка обработки голосового: %s", e)
            await self._api("sendMessage", chat_id=chat_id, text=f"Ошибка: {e}")
            return

        self.pending[user_id] = result["filename"]
        await self._api(
            "sendMessage",
            chat_id=chat_id,
            text=(
                f"Сэмпл сохранён: {result['filename']}\n"
                "Отправьте правильный текст ответом, чтобы подтвердить пару."
            ),
        )

    async def _handle_text(self, message: dict) -> None:
        user_id = message.get("from", {}).get("id", 0)
        chat_id = message.get("chat", {}).get("id", user_id)
        text = (message.get("text") or "").strip()

        if text.startswith("/start"):
            await self._api(
                "sendMessage",
                chat_id=chat_id,
                text=(
                    "Отправьте голосовое сообщение — я сохраню его и попрошу "
                    "подтвердить текст ответом."
                ),
            )
            return

        filename = self.pending.get(user_id)
        if not filename:
            await self._api(
                "sendMessage",
                chat_id=chat_id,
                text="Нет ожидающего сэмпла. Сначала отправьте голосовое сообщение.",
            )
            return

        if not text:
            await self._api("sendMessage", chat_id=chat_id, text="Текст пустой, попробуйте снова.")
            return

        handlers.confirm_sample(filename, text)
        self.pending.pop(user_id, None)
        await self._api(
            "sendMessage",
            chat_id=chat_id,
            text=f"Подтверждено: {filename} → «{text}»",
        )
        await self._maybe_start_finetune()

    async def _handle_update(self, update: dict) -> None:
        message = update.get("message") or {}
        if message.get("voice"):
            await self._handle_voice(message)
        elif message.get("text"):
            await self._handle_text(message)

    async def run(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            self._session = session
            logger.info("Бот запущен (long polling).")
            while True:
                try:
                    data = await self._api("getUpdates", offset=self.offset, timeout=30)
                    for update in data.get("result", []):
                        self.offset = update["update_id"] + 1
                        await self._handle_update(update)
                except Exception as e:
                    logger.error("Ошибка polling: %s", e)
                    await asyncio.sleep(3)


def main(token: str = None, finetune_threshold: int = FINETUNE_THRESHOLD) -> None:
    """Запускает бота. Токен: аргумент → env ``KAN_ASR_BOT_TOKEN``."""
    setup_logging()
    if token is None:
        token = os.environ.get("KAN_ASR_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Не задан токен бота. Установите KAN_ASR_BOT_TOKEN или передайте token.")
    bot = TelegramBot(token, finetune_threshold=finetune_threshold)
    asyncio.run(bot.run())


__all__ = ["TelegramBot", "main", "run_finetune"]
