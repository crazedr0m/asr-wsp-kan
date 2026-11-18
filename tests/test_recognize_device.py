"""
E2E-тесты распознавания WAV → текст на реальной плате Luckfox (RV1103 NPU).

Проверяется:
- smoke: плата принимает Mel и возвращает непустой текст, latency > 0;
- exact-match: распознанный текст совпадает с эталоном (строгий gate);
- latency: замер не превышает разумного потолка (информационно).

Все тесты помечены ``hardware`` и пропускаются, если нет ``rknn-toolkit2``,
``adb``, подключённой платы или ``.rknn``. Запуск::

    pytest -m hardware tests/test_recognize_device.py
    make recognize-board
"""

import os
import sys

import pytest

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.export.verify_rknn import _adb_devices
from kan_asr.inference.recognize import (
    DEFAULT_TARGET,
    evaluate_recognition,
    recognize_on_device,
)
from kan_asr.utils.paths import resolver

STRICT_ENV = "KAN_ASR_STRICT_RECOGNITION"
# Потолок latency одного инференса на плате, мс (информационная проверка)
DEFAULT_MAX_LATENCY_MS = float(os.environ.get("KAN_ASR_MAX_LATENCY_MS", "2000"))


def _strict_enabled() -> bool:
    """Включён ли строгий gate точного совпадения / CER."""
    return os.environ.get(STRICT_ENV, "0") == "1"


def _require_board(rknn_path: str) -> None:
    """Пропускает тест, если нет toolkit, adb, платы или ``.rknn``."""
    pytest.importorskip(
        "rknn",
        reason="rknn-toolkit2 не установлен (см. docs/setup.md)",
    )
    if not os.path.exists(rknn_path):
        pytest.skip(
            f"RKNN-модель не найдена: {rknn_path}. "
            "Сначала выполните make convert (scripts/convert_rknn.py)"
        )
    try:
        devices = _adb_devices()
    except RuntimeError:
        pytest.skip("adb недоступен (установите android-tools-adb)")
    if not devices:
        pytest.skip("Плата Luckfox не подключена по USB/ADB")


def _require_cases(recognition_cases) -> None:
    """Пропускает тест, если нет эталонных кейсов."""
    if not recognition_cases:
        pytest.skip(
            f"Нет кейсов распознавания: {resolver.synth / 'metadata.csv'}. "
            "Сначала сгенерируйте датасет (make data)"
        )


def _recognize_board(wav_path: str, rknn_path: str):
    """Обёртка над ``recognize_on_device`` с путями из фикстур."""
    return recognize_on_device(
        wav_path,
        rknn_model=rknn_path,
        target=DEFAULT_TARGET,
    )


# --- E2E на реальной плате -------------------------------------------------


@pytest.mark.hardware
@pytest.mark.recognition
def test_recognize_device_smoke(rknn_path, recognition_cases):
    """Плата распознаёт WAV, возвращает непустой текст и latency > 0."""
    _require_board(rknn_path)
    _require_cases(recognition_cases)

    result = _recognize_board(recognition_cases[0].wav_path, rknn_path)

    assert isinstance(result.text, str), f"Ожидалась строка, получено {type(result.text)}"
    assert result.text.strip(), "Плата вернула пустой текст"
    assert result.latency_ms > 0.0, "Некорректная latency (< 0)"
    assert result.target == DEFAULT_TARGET


@pytest.mark.hardware
@pytest.mark.recognition
def test_recognize_device_latency(rknn_path, recognition_cases):
    """Latency одного инференса на плате не выше потолка (информационно)."""
    _require_board(rknn_path)
    _require_cases(recognition_cases)

    result = _recognize_board(recognition_cases[0].wav_path, rknn_path)

    assert result.latency_ms <= DEFAULT_MAX_LATENCY_MS, (
        f"Latency {result.latency_ms:.1f} мс превышает потолок "
        f"{DEFAULT_MAX_LATENCY_MS} мс"
    )


@pytest.mark.hardware
@pytest.mark.recognition
def test_recognize_device_exact_match(
    rknn_path, recognition_cases, min_exact_match
):
    """
    Точное совпадение распознанного текста с эталоном на реальной плате.

    Строгий gate: включается ``KAN_ASR_STRICT_RECOGNITION=1``.
    """
    if not _strict_enabled():
        pytest.skip(
            f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки "
            "точного совпадения на плате"
        )
    _require_board(rknn_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: _recognize_board(wav, rknn_path).text,
        min_exact_match=min_exact_match,
    )

    assert report.exact_match_rate >= min_exact_match, (
        f"Доля точных совпадений {report.exact_match_rate:.3f} ниже порога "
        f"{min_exact_match}. Детали: {report.to_dict()['cases']}"
    )


@pytest.mark.hardware
@pytest.mark.recognition
def test_recognize_device_cer(rknn_path, recognition_cases, max_cer):
    """CER распознавания на плате не выше порога (строгий gate)."""
    if not _strict_enabled():
        pytest.skip(
            f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки CER"
        )
    _require_board(rknn_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: _recognize_board(wav, rknn_path).text,
        max_cer=max_cer,
    )

    assert report.cer <= max_cer, (
        f"CER {report.cer:.4f} превышает порог {max_cer}. "
        f"Детали: {report.to_dict()['cases']}"
    )
