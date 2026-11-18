"""
Обработчики Telegram-бота (Human-in-the-Loop сбор данных).

Бот принимает голосовые сообщения (.ogg), конвертирует их в WAV 16 кГц/моно
(формат, который читает :class:`~kan_asr.data.dataset.ASRDataset`), сохраняет
в ``data/real/wavs/`` и предлагает пользователю подтвердить или исправить
распознанный текст. Подтверждённые пары (аудио, текст) попадают в
``data/real/metadata.csv``.
"""

import csv
import os
import shutil
import subprocess
import tempfile

from ..utils.logging import get_logger
from ..utils.paths import resolver

logger = get_logger("bot.handlers")

METADATA_HEADER = ["filename", "text"]

# Параметры конвертации под ASRDataset (scipy.io.wavfile + 16 кГц)
TARGET_SR = 16000
TARGET_CHANNELS = 1


def _ensure_real_dir() -> str:
    real_dir = str(resolver.real)
    os.makedirs(os.path.join(real_dir, "wavs"), exist_ok=True)
    return real_dir


def _append_metadata(real_dir: str, filename: str, text: str) -> None:
    meta_path = os.path.join(real_dir, "metadata.csv")
    write_header = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        if write_header:
            writer.writerow(METADATA_HEADER)
        writer.writerow([filename, text])


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_ogg_to_wav(ogg_path: str, wav_path: str) -> bool:
    """
    Конвертирует OGG/Opus (голосовые Telegram) в WAV PCM 16 кГц/моно.

    Возвращает ``True`` при успехе. Требует системный ``ffmpeg``.
    """
    if not _ffmpeg_available():
        logger.error("ffmpeg не найден в PATH — конвертация ogg→wav невозможна")
        return False

    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        ogg_path,
        "-ar",
        str(TARGET_SR),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg завершился с ошибкой: %s", e.stderr.decode(errors="ignore"))
        return False
    return True


def save_voice_sample(ogg_bytes: bytes, user_id: int, message_id: int) -> str:
    """
    Сохраняет голосовой сэмпл как WAV в ``data/real/wavs/``.

    OGG-байты пишутся во временный файл, конвертируются в WAV 16 кГц/моно,
    временный файл удаляется. Возвращает относительный путь (для metadata.csv).
    """
    real_dir = _ensure_real_dir()
    base = f"user{user_id}_{message_id}"
    wav_path = os.path.join(real_dir, "wavs", f"{base}.wav")

    fd, tmp_ogg = tempfile.mkstemp(suffix=".ogg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(ogg_bytes)
        ok = convert_ogg_to_wav(tmp_ogg, wav_path)
    finally:
        if os.path.exists(tmp_ogg):
            os.remove(tmp_ogg)

    if not ok:
        raise RuntimeError("Не удалось конвертировать голосовое сообщение в WAV (проверьте ffmpeg)")

    logger.info("Сохранён голосовой сэмпл: %s", wav_path)
    return f"wavs/{base}.wav"


def confirm_sample(filename: str, text: str) -> None:
    """Фиксирует подтверждённую пользователем пару (аудио, текст)."""
    real_dir = _ensure_real_dir()
    _append_metadata(real_dir, filename, text)
    logger.info("Подтверждён сэмпл %s -> %r", filename, text)


def count_confirmed_samples() -> int:
    """Возвращает число подтверждённых пар в ``data/real/metadata.csv``."""
    meta_path = os.path.join(str(resolver.real), "metadata.csv")
    if not os.path.exists(meta_path):
        return 0
    with open(meta_path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("filename")]
    return len(lines)


def handle_voice(
    ogg_bytes: bytes, user_id: int, message_id: int, recognized_text: str = ""
) -> dict:
    """
    Полный цикл обработки голосового сообщения.

    Возвращает словарь с сохранённым путём и предварительным текстом,
    который бот показывает пользователю для подтверждения/правки.
    """
    filename = save_voice_sample(ogg_bytes, user_id, message_id)
    return {"filename": filename, "text": recognized_text}
