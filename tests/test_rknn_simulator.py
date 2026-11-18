"""
Тесты верификации RKNN-модели в симуляторе NPU на рабочем ПК.

Тесты требуют ``rknn-toolkit2`` и калибровочный датасет. Если чего-то нет —
тест пропускается (``pytest.skip``), чтобы обычный ``make test`` не падал.

Проверяется:
- сборка ``.rknn`` INT8 из ONNX;
- ``init_runtime(target=None)`` — эмулятор NPU на CPU;
- ``inference`` в эмуляторе;
- сравнение с ONNX Runtime по косинусному сходству;
- размер ``.rknn`` <= 4 МБ;
- ``eval_memory`` под лимит 64 МБ (если отчёт доступен).
"""

import os
import sys

import pytest

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.utils.paths import resolver

# Пропуск всего модуля, если rknn-toolkit2 не установлен
rknn = pytest.importorskip(
    "rknn", reason="rknn-toolkit2 не установлен (pip install -e '.[rknn]')"
)

from kan_asr.export.verify_rknn import (
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_MAX_SIZE_MB,
    DEFAULT_MIN_COSINE,
    _cosine_similarity,
    _parse_peak_memory_mb,
    verify_rknn_simulator,
)


def _onnx_path() -> str:
    return str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")


def _calib_path() -> str:
    return str(resolver.calib / "calib_dataset.txt")


def _require_artifacts():
    """Пропускает тест, если нет ONNX-модели или калибровочного датасета."""
    if not os.path.exists(_onnx_path()):
        pytest.skip(
            f"ONNX-модель не найдена: {_onnx_path()}. "
            "Сначала выполните scripts/export_onnx.py"
        )
    if not os.path.exists(_calib_path()):
        pytest.skip(
            f"Калибровочный датасет не найден: {_calib_path()}. "
            "Сначала обучите модель (scripts/train.py)"
        )


def test_cosine_similarity_identical():
    """Косинусное сходство одинаковых массивов равно 1."""
    import numpy as np

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    """Косинусное сходство ортогональных массивов равно 0."""
    import numpy as np

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_parse_peak_memory_mb():
    """Парсер пиковой памяти извлекает максимум из отчёта eval_memory."""
    report = (
        "Memory Profile Summary:\n"
        "  Total Weight Memory: 1.5 MB\n"
        "  Total Activation Memory: 12.25 MB\n"
        "  Peak Memory: 14 MB\n"
    )
    assert _parse_peak_memory_mb(report) == pytest.approx(14.0)


def test_parse_peak_memory_mb_empty():
    """Пустой отчёт даёт None."""
    assert _parse_peak_memory_mb(None) is None
    assert _parse_peak_memory_mb("") is None


def test_parse_peak_memory_mb_dict():
    """На устройстве eval_memory возвращает dict со значениями в байтах."""
    report = {
        "Weight Memory": 276224,
        "Internal Tensor Memory": 196608,
        "Other Memory": 55552,
        "Total Memory": 528384,
    }
    assert _parse_peak_memory_mb(report) == pytest.approx(528384 / (1024 * 1024))


def test_verify_rknn_simulator():
    """
    Полная проверка RKNN в симуляторе NPU.

    Собирает ``.rknn`` INT8 из ONNX, поднимает эмулятор, прогоняет вход,
    сравнивает с ONNX Runtime и проверяет лимиты размера и памяти.
    """
    _require_artifacts()

    result = verify_rknn_simulator(
        onnx_model=_onnx_path(),
        calib_dataset=_calib_path(),
        min_cosine=DEFAULT_MIN_COSINE,
        max_size_mb=DEFAULT_MAX_SIZE_MB,
        max_memory_mb=DEFAULT_MAX_MEMORY_MB,
        rebuild=True,
        verbose=False,
    )

    assert result.size_ok, (
        f"Размер .rknn {result.rknn_size_mb:.3f} МБ превышает "
        f"лимит {DEFAULT_MAX_SIZE_MB} МБ"
    )
    assert result.cosine_similarity >= DEFAULT_MIN_COSINE, (
        f"Косинусное сходство {result.cosine_similarity:.6f} ниже порога "
        f"{DEFAULT_MIN_COSINE}"
    )
    assert result.output_shape == (1, 35, 1, 50), (
        f"Неожиданная форма выхода: {result.output_shape}"
    )
    if result.memory_ok is not None:
        assert result.memory_ok, (
            f"Пиковая память {result.peak_memory_mb:.2f} МБ превышает "
            f"лимит {DEFAULT_MAX_MEMORY_MB} МБ"
        )
    assert result.passed, "Проваленные проверки: " + "; ".join(result.failures)
