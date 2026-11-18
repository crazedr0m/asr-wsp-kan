"""
Тесты бенчмарка RKNN на реальной плате (RV1103 NPU).

Не требуют ``rknn-toolkit2`` на этапе импорта: модуль ``verify_rknn``
импортирует тулчейн лениво. Интеграционный тест самого замера помечается
``skip``, если ``adb``/плата/``rknn-toolkit2`` недоступны.

Проверяется:
- статистика latency (mean/min/max/p50/p90);
- парсинг вывода ``adb devices``;
- выбор устройства (``_ensure_device``);
- сериализация ``PerformanceResult`` в JSON;
- опционально — реальный замер на подключённой плате.
"""

import json
import os
import sys

import pytest

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.export import verify_rknn as vr
from kan_asr.export.verify_rknn import (
    PerformanceResult,
    _adb_devices,
    _ensure_device,
    _latency_stats,
    save_perf_report,
)


def test_latency_stats_basic():
    """Статистика корректно считает mean/min/max/перцентили."""
    stats = _latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(50.0)
    assert stats["p50"] == pytest.approx(30.0)
    assert stats["p90"] == pytest.approx(46.0)


def test_latency_stats_single_sample():
    """Один замер: все метрики равны ему."""
    stats = _latency_stats([12.5])
    assert stats["mean"] == pytest.approx(12.5)
    assert stats["min"] == pytest.approx(12.5)
    assert stats["max"] == pytest.approx(12.5)
    assert stats["p50"] == pytest.approx(12.5)
    assert stats["p90"] == pytest.approx(12.5)


def test_latency_stats_empty_raises():
    """Пустой список замеров — ошибка."""
    with pytest.raises(ValueError):
        _latency_stats([])


def test_adb_devices_parsing(monkeypatch):
    """Парсер оставляет только устройства в состоянии ``device``."""
    output = (
        "List of devices attached\n"
        "67c890ed4c2fcc27\tdevice\n"
        "emulator-5554\tdevice\n"
        "1234567890\toffline\n"
        "abcdef\tunauthorized\n"
        "\n"
    )
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(vr.subprocess, "check_output", lambda *a, **k: output)
    assert _adb_devices() == ["67c890ed4c2fcc27", "emulator-5554"]  # pragma: allowlist secret


def test_adb_devices_empty(monkeypatch):
    """Нет подключённых устройств — пустой список."""
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(
        vr.subprocess,
        "check_output",
        lambda *a, **k: "List of devices attached\n\n",
    )
    assert _adb_devices() == []


def test_adb_devices_missing_binary(monkeypatch):
    """Отсутствие adb даёт понятную ошибку."""
    monkeypatch.setattr(vr.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="adb"):
        _adb_devices()


def test_ensure_device_explicit_match(monkeypatch):
    """Явно заданный device_id возвращается, если он подключён."""
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(vr, "_adb_devices", lambda adb=None: ["serial-1"])
    assert _ensure_device("serial-1") == "serial-1"


def test_ensure_device_explicit_missing(monkeypatch):
    """Неизвестный device_id — ошибка."""
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(vr, "_adb_devices", lambda adb=None: ["serial-1"])
    with pytest.raises(RuntimeError, match="serial-9"):
        _ensure_device("serial-9")


def test_ensure_device_single_auto(monkeypatch):
    """Без device_id выбирается единственное устройство."""
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(vr, "_adb_devices", lambda adb=None: ["only-one"])
    assert _ensure_device(None) == "only-one"


def test_ensure_device_multiple_requires_explicit(monkeypatch):
    """Несколько устройств без device_id — ошибка."""
    monkeypatch.setattr(vr, "_adb_binary", lambda: "/usr/bin/adb")
    monkeypatch.setattr(vr, "_adb_devices", lambda adb=None: ["a", "b"])
    with pytest.raises(RuntimeError, match="несколько"):
        _ensure_device(None)


def _make_result(**overrides) -> PerformanceResult:
    base = {
        "target": "rv1103",
        "device_id": "67c890ed4c2fcc27",
        "iterations": 20,
        "warmup": 3,
        "latency_ms_mean": 45.0,
        "latency_ms_min": 40.0,
        "latency_ms_max": 60.0,
        "latency_ms_p50": 44.0,
        "latency_ms_p90": 55.0,
        "fps": 22.22,
        "rtf": 0.0225,
        "audio_seconds": 2.0,
    }
    base.update(overrides)
    return PerformanceResult(**base)


def test_performance_result_to_dict():
    """Сериализация PerformanceResult содержит ключевые метрики."""
    payload = _make_result().to_dict()
    assert payload["target"] == "rv1103"
    assert payload["latency_ms"]["mean"] == pytest.approx(45.0)
    assert payload["fps"] == pytest.approx(22.22)
    assert payload["rtf"] == pytest.approx(0.0225)
    assert payload["passed"] is True


def test_performance_result_failures():
    """Проваленные пороги попадают в failures и ломают passed."""
    result = _make_result(
        max_latency_ms=30.0, failures=["Средняя latency 45.00 мс превышает порог 30.0 мс"]
    )
    assert result.passed is False
    assert "ПРОИЗВОДИТЕЛЬНОСТЬ" not in result.summary()
    assert result.summary().startswith("Плата:")


def test_save_perf_report(tmp_path):
    """JSON-отчёт пишется на диск и читается обратно."""
    path = tmp_path / "perf_report.json"
    saved = save_perf_report(_make_result(), str(path))
    assert saved == str(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["target"] == "rv1103"
    assert payload["iterations"] == 20


def test_benchmark_on_device_skips_without_board():
    """Интеграционный замер пропускается, если нет adb/платы/toolkit."""
    pytest.importorskip("rknn", reason="rknn-toolkit2 не установлен")
    try:
        devices = _adb_devices()
    except RuntimeError:
        pytest.skip("adb недоступен")
    if not devices:
        pytest.skip("Плата Luckfox не подключена по USB/ADB")

    result = vr.benchmark_on_device(warmup=1, iterations=3, verbose=False)
    assert result.iterations >= 1
    assert result.latency_ms_mean > 0.0
    assert result.fps > 0.0
    assert result.rtf > 0.0
