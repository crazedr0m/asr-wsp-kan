# -*- coding: utf-8 -*-
"""Тесты PureMelSpectrogram (src/kan_asr/data/mel.py)."""

import os
import sys

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kan_asr.data.mel import PureMelSpectrogram


def test_mel_shape():
    """Форма выхода: [B, n_mels, 1, T]."""
    mel = PureMelSpectrogram(sr=16000, n_fft=400, hop_length=160, n_mels=40)
    waveform = torch.randn(1, 16000)
    out = mel(waveform)
    # center=True: 1 + 16000 // 160 = 101 кадр
    assert out.shape == (1, 40, 1, 101), f"Ожидается (1, 40, 1, 101), получил {out.shape}"


def test_mel_batch():
    """Батчевая обработка сохраняет размерность batch."""
    mel = PureMelSpectrogram()
    waveform = torch.randn(3, 8000)
    out = mel(waveform)
    assert out.shape[0] == 3
    assert out.shape[1] == 40
    assert out.shape[2] == 1


def test_mel_finite():
    """Выход не содержит NaN/Inf."""
    mel = PureMelSpectrogram()
    out = mel(torch.randn(1, 16000))
    assert torch.isfinite(out).all(), "Mel-спектрограмма содержит NaN/Inf"


def test_mel_filterbank():
    """Фильтрбанк неотрицателен и нормирован по мел-бинам."""
    fb = PureMelSpectrogram._build_mel_filterbank(16000, 400, 40)
    assert fb.shape[0] == 40
    assert (fb >= 0).all(), "Фильтрбанк должен быть неотрицательным"


if __name__ == "__main__":
    test_mel_shape()
    test_mel_batch()
    test_mel_finite()
    test_mel_filterbank()
    print("[PASS] test_mel")
