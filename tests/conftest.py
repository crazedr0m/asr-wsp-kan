"""
Общие фикстуры и метки для тестов распознавания WAV → текст.

Регистрирует метки ``recognition`` (E2E-распознавание) и ``hardware``
(тесты, требующие подключённой платы Luckfox), а также предоставляет
фикстуры путей к артефактам и набор кейсов распознавания.
"""

import os
import sys

import pytest

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kan_asr.inference.recognize import load_recognition_cases
from kan_asr.utils.paths import resolver

# Сколько кейсов брать по умолчанию в E2E-тестах
DEFAULT_N_CASES = int(os.environ.get("KAN_ASR_RECOGNITION_CASES", "3"))


def pytest_configure(config):
    """Регистрирует пользовательские метки, чтобы не было предупреждений."""
    config.addinivalue_line("markers", "recognition: E2E-тесты распознавания WAV → текст")
    config.addinivalue_line(
        "markers", "hardware: тесты, требующие подключённой платы Luckfox (ADB)"
    )


@pytest.fixture(scope="session")
def checkpoint_path() -> str:
    """Путь к чекпоинту хоста (``artifacts/checkpoints/best.pt``)."""
    return str(resolver.checkpoints / "best.pt")


@pytest.fixture(scope="session")
def onnx_path() -> str:
    """Путь к ONNX-модели (``artifacts/onnx/tiny_kan_asr_rv1103.onnx``)."""
    return str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")


@pytest.fixture(scope="session")
def rknn_path() -> str:
    """Путь к ``.rknn`` (``artifacts/rknn/tiny_kan_asr_rv1103.rknn``)."""
    return str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")


@pytest.fixture(scope="session")
def calib_path() -> str:
    """Путь к калибровочному датасету (``data/calib/calib_dataset.txt``)."""
    return str(resolver.calib / "calib_dataset.txt")


@pytest.fixture(scope="session")
def recognition_cases():
    """Детерминированный набор кейсов распознавания из ``data/synth``."""
    return load_recognition_cases(n=DEFAULT_N_CASES)


@pytest.fixture(scope="session")
def max_cer() -> float:
    """Порог CER из окружения (``KAN_ASR_RECOGNITION_MAX_CER``)."""
    return float(os.environ.get("KAN_ASR_RECOGNITION_MAX_CER", "1.0"))


@pytest.fixture(scope="session")
def min_exact_match() -> float:
    """Порог доли точных совпадений (``KAN_ASR_MIN_EXACT_MATCH``)."""
    return float(os.environ.get("KAN_ASR_MIN_EXACT_MATCH", "0.0"))
