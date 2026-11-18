"""
E2E-тесты распознавания WAV → текст в симуляторе NPU (RKNN INT8, target=None).

Проверяется:
- smoke: INT8-модель в эмуляторе NPU даёт непустую строку;
- согласованность с хостом: CER(текст хоста, текст симулятора) не выше порога
  (квантизация не должна ломать распознавание);
- quality: exact-match / CER против эталона (строгий gate).

Тесты пропускаются, если нет ``rknn-toolkit2``, ONNX, калибровочного датасета
или чекпоинта хоста, чтобы обычный ``make test`` не падал.
"""

import os
import sys

import pytest

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.inference.recognize import (
    evaluate_recognition,
    recognize_rknn_simulator,
    recognize_torch,
)
from kan_asr.training.metrics import cer
from kan_asr.utils.paths import resolver

STRICT_ENV = "KAN_ASR_STRICT_RECOGNITION"
# Насколько текст симулятора может отличаться от текста хоста (CER между ними)
DEFAULT_SIM_HOST_MAX_CER = float(os.environ.get("KAN_ASR_SIM_HOST_MAX_CER", "0.5"))


def _strict_enabled() -> bool:
    """Включён ли строгий gate точного совпадения / CER."""
    return os.environ.get(STRICT_ENV, "0") == "1"


def _require_rknn() -> None:
    """Пропускает тест, если ``rknn-toolkit2`` не установлен."""
    pytest.importorskip(
        "rknn",
        reason="rknn-toolkit2 не установлен (см. docs/setup.md)",
    )


def _require_artifacts(onnx_path: str, calib_path: str) -> None:
    """Пропускает тест, если нет ONNX-модели или калибровочного датасета."""
    if not os.path.exists(onnx_path):
        pytest.skip(
            f"ONNX-модель не найдена: {onnx_path}. "
            "Сначала выполните make export (scripts/export_onnx.py)"
        )
    if not os.path.exists(calib_path):
        pytest.skip(f"Калибровочный датасет не найден: {calib_path}. Сначала выполните make calib")


def _require_cases(recognition_cases) -> None:
    """Пропускает тест, если нет эталонных кейсов."""
    if not recognition_cases:
        pytest.skip(
            f"Нет кейсов распознавания: {resolver.synth / 'metadata.csv'}. "
            "Сначала сгенерируйте датасет (make data)"
        )


def _recognize_sim(wav_path: str, onnx_path: str, calib_path: str) -> str:
    """Обёртка над ``recognize_rknn_simulator`` с путями из фикстур."""
    return recognize_rknn_simulator(
        wav_path,
        onnx_model=onnx_path,
        calib_dataset=calib_path,
    )


# --- E2E в симуляторе NPU --------------------------------------------------


@pytest.mark.recognition
def test_recognize_simulator_smoke(onnx_path, calib_path, recognition_cases):
    """INT8-модель в эмуляторе NPU распознаёт WAV и даёт непустую строку."""
    _require_rknn()
    _require_artifacts(onnx_path, calib_path)
    _require_cases(recognition_cases)

    text = _recognize_sim(recognition_cases[0].wav_path, onnx_path, calib_path)

    assert isinstance(text, str), f"Ожидалась строка, получено {type(text)}"
    assert text.strip(), "Симулятор NPU вернул пустой текст"


@pytest.mark.recognition
def test_recognize_simulator_matches_host(
    onnx_path, calib_path, checkpoint_path, recognition_cases
):
    """
    Распознавание в симуляторе согласовано с хостом.

    INT8-квантизация не должна сильно менять выход: CER между текстом хоста
    и текстом симулятора не выше ``KAN_ASR_SIM_HOST_MAX_CER`` (по умолчанию 0.5).
    """
    _require_rknn()
    _require_artifacts(onnx_path, calib_path)
    _require_cases(recognition_cases)
    if not os.path.exists(checkpoint_path):
        pytest.skip(
            f"Чекпоинт хоста не найден: {checkpoint_path}. Нужен для сравнения с симулятором"
        )

    host_texts: list[str] = []
    sim_texts: list[str] = []
    for case in recognition_cases:
        host_texts.append(recognize_torch(case.wav_path, checkpoint_path))
        sim_texts.append(_recognize_sim(case.wav_path, onnx_path, calib_path))

    delta_cer = cer(host_texts, sim_texts)
    assert delta_cer <= DEFAULT_SIM_HOST_MAX_CER, (
        f"CER(хост, симулятор) = {delta_cer:.4f} выше порога "
        f"{DEFAULT_SIM_HOST_MAX_CER}.\n"
        f"Хост:      {host_texts}\n"
        f"Симулятор: {sim_texts}"
    )


@pytest.mark.recognition
def test_recognize_simulator_exact_match(onnx_path, calib_path, recognition_cases, min_exact_match):
    """Точное совпадение распознанного текста с эталоном в симуляторе (строгий gate)."""
    if not _strict_enabled():
        pytest.skip(
            f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки "
            "точного совпадения в симуляторе"
        )
    _require_rknn()
    _require_artifacts(onnx_path, calib_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: _recognize_sim(wav, onnx_path, calib_path),
        min_exact_match=min_exact_match,
    )

    assert report.exact_match_rate >= min_exact_match, (
        f"Доля точных совпадений {report.exact_match_rate:.3f} ниже порога "
        f"{min_exact_match}. Детали: {report.to_dict()['cases']}"
    )


@pytest.mark.recognition
def test_recognize_simulator_cer(onnx_path, calib_path, recognition_cases, max_cer):
    """CER распознавания в симуляторе не выше порога (строгий gate)."""
    if not _strict_enabled():
        pytest.skip(f"Строгий gate отключён; включите {STRICT_ENV}=1 для проверки CER")
    _require_rknn()
    _require_artifacts(onnx_path, calib_path)
    _require_cases(recognition_cases)

    report = evaluate_recognition(
        recognition_cases,
        lambda wav: _recognize_sim(wav, onnx_path, calib_path),
        max_cer=max_cer,
    )

    assert report.cer <= max_cer, (
        f"CER {report.cer:.4f} превышает порог {max_cer}. Детали: {report.to_dict()['cases']}"
    )
