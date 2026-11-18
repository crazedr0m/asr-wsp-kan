"""
Генерация синтетического датасета через локальный Piper TTS.

Перенесено из KAN_ASR_Pipeline_Bundle/generate_dataset.py (легаси-каталог).
При отсутствии модели Piper используется тестовый сигнал-заглушка, чтобы
пайплайн работал «из коробки».
"""

import csv
import os
import random

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

from .augment import SAMPLE_RATE, add_augmentations

SAMPLE_PHRASES = [
    "включи свет в гостиной",
    "выключи музыку на кухне",
    "какая сегодня погода",
    "поставь таймер на десять минут",
    "открой дверь в прихожей",
    "проверка работы распознавания речи",
    "привет лакфокс как дела",
    "переключи канал на первый",
    "сделай звук погромче",
    "выключи все приборы в доме",
]


def synthesize_with_piper(text: str, out_wav: str, model_onnx: str) -> bool:
    """Генерация аудио через Python-библиотеку piper-tts."""
    if not os.path.exists(model_onnx):
        return False

    # Путь к config файлу
    config_path = model_onnx.replace(".onnx", ".onnx.json")
    if not os.path.exists(config_path):
        config_path = model_onnx + ".json"

    try:
        from piper import voice

        # Используем статический метод PiperVoice.load()
        piper_voice = voice.PiperVoice.load(model_onnx, config_path)
        # synthesize возвращает генератор AudioChunk
        audio_chunk = next(piper_voice.synthesize(text))
        # Получаем аудио из AudioChunk
        audio_int16 = audio_chunk.audio_int16_array
        sr = audio_chunk.sample_rate
        # Записываем WAV файл с правильным sample rate
        wav.write(out_wav, sr, audio_int16)
        return os.path.exists(out_wav)
    except Exception as e:
        print(f"[WARN] Piper TTS ошибка: {e}")
        return False


def generate_synthetic_dataset(
    output_dir: str = "data/synth",
    num_samples: int = 500,
    model_onnx: str = "artifacts/piper/ru_RU-irina-medium.onnx",
):
    """Генерирует датасет ``wavs/`` + ``metadata.csv`` (формат ``filename|text``)."""
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.csv")

    use_piper = os.path.exists(model_onnx)
    if not use_piper:
        print("[INFO] Piper TTS модель не найдена. Генерирую тестовые сигналы-заглушки для проверки пайплайна.")
    else:
        print(f"[INFO] Piper TTS модель найдена: {model_onnx}")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "text"])

        for i in range(num_samples):
            text = random.choice(SAMPLE_PHRASES)
            fname = f"utt_{i:06d}.wav"
            fpath = os.path.join(wav_dir, fname)

            if use_piper and synthesize_with_piper(text, fpath, model_onnx):
                sr, data = wav.read(fpath)
                if data.dtype == np.int16:
                    audio = data.astype(np.float32) / 32768.0
                else:
                    audio = data.astype(np.float32)
                if sr != SAMPLE_RATE:
                    audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
            else:
                # Тестовый синтетический сигнал (модулированный тон), чтобы код работал из коробки
                dur = random.uniform(1.0, 2.5)
                t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
                freq = random.uniform(150.0, 450.0)
                audio = 0.3 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))

            aug_audio = add_augmentations(audio, SAMPLE_RATE)
            wav.write(fpath, SAMPLE_RATE, (aug_audio * 32767.0).astype(np.int16))
            writer.writerow([f"wavs/{fname}", text])

    print(f"[DONE] Сгенерировано {num_samples} записей в {output_dir}/")
