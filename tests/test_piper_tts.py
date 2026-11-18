"""
Тест Piper TTS (перенесено из корневого test_piper_tts.py).

Пропускается, если библиотека ``piper`` или модель не установлены.
"""

import os
import sys

import pytest

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.utils.paths import resolver

SAMPLE_RATE = 16000


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _piper_available(), reason="piper-tts не установлен")
def test_piper_voice():
    """Проверка работы Piper TTS модели (если модель доступна)."""
    from piper import voice

    model_path = str(resolver.piper / "ru_RU-irina-medium.onnx")
    config_path = model_path + ".json"

    if not os.path.exists(model_path):
        pytest.skip(f"Модель Piper не найдена: {model_path}")

    piper_voice = voice.PiperVoice.load(model_path, config_path)

    phrase = "привет, это тестовое аудио"
    audio_chunk = next(piper_voice.synthesize(phrase))
    audio_int16 = audio_chunk.audio_int16_array
    assert len(audio_int16) > 0, "Piper вернул пустое аудио"
    assert audio_chunk.sample_rate > 0


if __name__ == "__main__":
    if not _piper_available():
        print("[SKIP] piper-tts не установлен")
        sys.exit(0)
    test_piper_voice()
    print("[PASS] test_piper_tts")
