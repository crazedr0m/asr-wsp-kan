"""
Аугментации аудио на чистом NumPy/SciPy.

Перенесено из KAN_ASR_Pipeline_Bundle/generate_dataset.py (легаси-каталог).
"""

import random

import numpy as np
import scipy.signal as signal

SAMPLE_RATE = 16000


def add_augmentations(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Аугментации на чистом NumPy/SciPy: сдвиг скорости, шум, случайная громкость."""
    # 1. Изменение скорости (0.9x - 1.1x)
    speed_factor = random.uniform(0.9, 1.1)
    if abs(speed_factor - 1.0) > 0.02:
        new_len = int(len(audio) / speed_factor)
        audio = signal.resample(audio, new_len)

    # 2. Случайная громкость
    gain = random.uniform(0.5, 1.2)
    audio = audio * gain

    # 3. Белый или розовый фоновый шум
    if random.random() < 0.8:
        noise_level = random.uniform(0.002, 0.025)
        noise = np.random.normal(0, noise_level, size=audio.shape)
        audio = audio + noise

    # 4. Имитация легкого клиппинга / кодека
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)
