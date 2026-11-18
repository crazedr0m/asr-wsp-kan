"""
Инференс KAN-ASR: распознавание речи WAV → текст.

Пакет объединяет единый пайплайн распознавания (Mel → модель → CTC-декод)
для трёх уровней: хост (PyTorch), симулятор NPU и реальная плата Luckfox.
"""

from .recognize import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_MAX_CER,
    DEFAULT_MIN_EXACT_MATCH,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TARGET,
    DEFAULT_TIME_FRAMES,
    DeviceRecognitionResult,
    RecognitionCase,
    RecognitionReport,
    audio_to_mel,
    evaluate_recognition,
    fit_mel_length,
    load_audio,
    load_recognition_cases,
    recognize_on_device,
    recognize_rknn_simulator,
    recognize_torch,
    save_recognition_report,
)

__all__ = [
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_MAX_CER",
    "DEFAULT_MIN_EXACT_MATCH",
    "DEFAULT_N_MELS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TARGET",
    "DEFAULT_TIME_FRAMES",
    "DeviceRecognitionResult",
    "RecognitionCase",
    "RecognitionReport",
    "audio_to_mel",
    "evaluate_recognition",
    "fit_mel_length",
    "load_audio",
    "load_recognition_cases",
    "recognize_on_device",
    "recognize_rknn_simulator",
    "recognize_torch",
    "save_recognition_report",
]
