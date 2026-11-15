# Инструкция по установке зависимостей для KAN-ASR проекта

## Требования к системе

- **ОС:** Linux (проверено на Ubuntu 22.04+)
- **Python:** 3.10+
- **GPU (опционально):** NVIDIA с CUDA 12.x (для обучения и CUDA-ядра)
- **FFmpeg:** для конвертации аудио (`ogg -> wav`)
- **NPU-тулчейн (опционально):** `rknn-toolkit2` для конвертации в RKNN (Этап 5)

## Установка

### Шаг 1: Клонирование и venv

```bash
git clone <repo-url> voice-recognition
cd voice-recognition

python3 -m venv venv
source venv/bin/activate
```

### Шаг 2: PyTorch (с CUDA или CPU)

```bash
# Вариант с CUDA 12.x (рекомендуется для обучения)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Вариант CPU-only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Шаг 3: Runtime-зависимости и пакет

```bash
# Базовые зависимости (numpy, scipy, onnx, pyyaml, aiohttp, ...)
pip install -r requirements.txt

# Установка пакета kan_asr в editable-режиме
# (регистрирует CLI-команды: train, wsp-init, export-onnx, convert-rknn, run-bot, generate-dataset)
pip install -e .
```

### Шаг 4: Опциональные extras

Тяжёлые и платформо-зависимые зависимости вынесены в extras, чтобы базовая
установка оставалась лёгкой:

```bash
# rknn-toolkit2 — только для конвертации ONNX -> RKNN (Этап 5)
pip install -e ".[rknn]"

# piper-tts — только для генерации синтетического датасета
pip install -e ".[tts]"

# Всё сразу
pip install -e ".[all]"
```

Альтернативно — через [`requirements-optional.txt`](../requirements-optional.txt):

```bash
pip install -r requirements-optional.txt
```

### Шаг 5: Проверка установки

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import kan_asr; print('kan_asr OK')"
pytest tests/ -v
```

## Генерация синтетического датасета

Если `piper-tts` установлен (extras `tts`):

```bash
python scripts/generate_dataset.py
```

Если установка Piper TTS сложна, можно использовать тестовые сигналы-заглушки —
модуль [`src/kan_asr/data/generate.py`](../src/kan_asr/data/generate.py) умеет
работать без Piper и генерирует синтетические аудио-сигналы. Это позволяет
быстро получить датасет для отладки контура, а Piper добавить позже.

## Ожидаемое время установки

- Создание venv: ~1 минута
- Установка PyTorch: ~5-10 минут
- Установка остальных зависимостей: ~3-5 минут
- Скачивание Piper TTS: ~10-15 минут (зависит от скорости интернета)

**Итого:** ~20-30 минут

---

## Следующие шаги после установки

1. Проверка установки зависимостей (`pytest tests/ -v`)
2. Генерация синтетического датасета
3. Инициализация весов из Whisper (WSP): `python scripts/wsp_init.py --whisper-state <path>`
4. Обучение: `python scripts/train.py`
5. Экспорт: `python scripts/export_onnx.py` -> `python scripts/convert_rknn.py`

См. также: [Архитектура](architecture.md), [Weight-Space Programming](wsp.md).
