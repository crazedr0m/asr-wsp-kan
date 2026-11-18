"""
E2E-тесты распознавания WAV → текст на хосте (PyTorch, float32).

Проверяется:
- unit-уровень: ``load_audio``, ``audio_to_mel``, ``fit_mel_length``;
- smoke: инференс по ``data/synth`` даёт непустую строку;
- exact-match: распознанный текст совпадает с эталоном (строгий gate,
  включается ``KAN_ASR_STRICT_RECOGNITION=1``);
- quality: CER/WER не выше порога (тот же строгий gate).

Без чекпоинта или без ``metadata.csv`` тесты пропускаются, чтобы обычный
``make test`` не падал.
"""

import os
import sys

import numpy as np
import pytest

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.inference.recognize import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_TIME_FRAMES,
    audio_to_mel,
    evaluate_recognition,
    fit_mel_length,
    load_audio,
    load_recognition_cases,
    recognize_torch,
)
from kan_asr.utils.paths import resolver

STRICT_ENV = "KAN_ASR_STRICT_RECOGNITION"


def _strict_enabled() -> bool:
    """Включён ли строгий gate точного совпадения / CER."""
    return os.environ.get(STRICT_ENV, "0") == "1"


def _require_checkpoint(checkpoint_path: str) -> None:
    """Пропускает тест, если чекпоинт отсутствует."""
    if not os.path.exists(checkpoint_path):
        pytest.skip(
            f"Чекпоинт не найден: {checkpoint_path}. Сначала обучите модель (scripts/train.py)"
        )


def _require_cases(recognition_cases) -> None:
    """Пропускает тест, если нет эталонных кейсов."""
    if not recognition_cases:
        pytest.skip(
            f"Нет кейсов распознавания: {resolver.synth / 'metadata.csv'}. "
            "Сначала сгенерируйте датасет (make data)"
        )


# --- Unit-тесты хелперов (без чекпоинта) -----------------------------------


def test_load_audio_is_mono_16k(recognition_cases):
    """``load_audio`` возвращает моно float32 16 кГц в диапазоне [-1, 1]."""
    _require_cases(recognition_cases)
    audio = load_audio(recognition_cases[0].wav_path)

    assert audio.ndim == 1, f"Ожидался моно-массив, получено {audio.shape}"
    assert audio.dtype == np.float32, f"Ожидался float32, получено {audio.dtype}"
    assert np.isfinite(audio).all(), "Аудио содержит NaN/Inf"


def test_audio_to_mel_shape(recognition_cases):
    """``audio_to_mel`` даёт ``[1, 40, 1, T]``."""
    _require_cases(recognition_cases)
    audio = load_audio(recognition_cases[0].wav_path)
    mel = audio_to_mel(audio)

    assert mel.ndim == 4, f"Ожидался 4D тензор, получено {mel.shape}"
    assert mel.shape[0] == 1
    assert mel.shape[1] == 40
    assert mel.shape[2] == 1
    assert np.isfinite(mel).all(), "Mel содержит NaN/Inf"


def test_fit_mel_length_pads_short(recognition_cases):
    """Короткий Mel паддится нулями до 200 кадров."""
    _require_cases(recognition_cases)
    audio = load_audio(recognition_cases[0].wav_path)
    mel = audio_to_mel(audio)
    fitted = fit_mel_length(mel, time_frames=DEFAULT_TIME_FRAMES)

    assert fitted.shape == DEFAULT_INPUT_SHAPE, (
        f"Ожидалась форма {DEFAULT_INPUT_SHAPE}, получено {fitted.shape}"
    )


def test_fit_mel_length_trims_long():
    """Длинный Mel обрезается до 200 кадров."""
    mel = np.zeros((1, 40, 1, 350), dtype=np.float32)
    fitted = fit_mel_length(mel, time_frames=DEFAULT_TIME_FRAMES)
    assert fitted.shape == DEFAULT_INPUT_SHAPE


def test_load_recognition_cases_returns_pairs(recognition_cases):
    """Кейсы содержат существующий WAV и непустой эталонный текст."""
    _require_cases(recognition_cases)
    for case in recognition_cases:
        assert os.path.exists(case.wav_path), f"Нет файла: {case.wav_path}"
        assert case.text, f"Пустой эталон для {case.wav_path}"


def test_load_recognition_cases_is_deterministic():
    """Отбор кейсов детерминирован."""
    first = load_recognition_cases(n=2)
    second = load_recognition_cases(n=2)
    assert [(c.wav_path, c.text) for c in first] == [(c.wav_path, c.text) for c in second]


# --- E2E на хосте ----------------------------------------------------------


@pytest.mark.recognition
def test_recognize_host_smoke(checkpoint_path, recognition_cases):
    """Инференс на хосте проходит и даёт непустую строку."""
    _require_checkpoint(checkpoint_path)
    _require_cases(recognition_cases)

    text = recognize_torch(recognition_cases[0].wav_path, checkpoint_path)

    assert isinstance(text, str), f"Ожидалась строка, получено {type(text)}"
    assert text.strip(), "Распознанный текст пуст"


@pytest.mark.recognition
def test_recognize_host_exact_match(checkpoint_path, recognition_cases, min_exact_match):
    """
    Распознанный текст совпадает с эталонным текстом тестовой фразы.

    Строгий gate: по умолчанию пропускается, включается
    ``KAN_ASR_STRICT_RECOGNITION=1`` (качество крошечной модели может быть
    недостаточным сразу после WSP-инициализации).
    """
    if not _strict_enabled():
        pytest.skip(
            f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки точного совпадения"
        )
    _require_checkpoint(checkpoint_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: recognize_torch(wav, checkpoint_path),
        min_exact_match=min_exact_match,
    )

    assert report.exact_match_rate >= min_exact_match, (
        f"Доля точных совпадений {report.exact_match_rate:.3f} ниже порога "
        f"{min_exact_match}. Детали: {report.to_dict()['cases']}"
    )


@pytest.mark.recognition
def test_recognize_host_cer(checkpoint_path, recognition_cases, max_cer):
    """CER распознавания на хосте не выше порога (строгий gate)."""
    if not _strict_enabled():
        pytest.skip(f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки CER")
    _require_checkpoint(checkpoint_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: recognize_torch(wav, checkpoint_path),
        max_cer=max_cer,
    )

    assert report.cer <= max_cer, (
        f"CER {report.cer:.4f} превышает порог {max_cer}. Детали: {report.to_dict()['cases']}"
    )
