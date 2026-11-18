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
# rknn-toolkit2 — конвертация ONNX -> RKNN, симулятор NPU и замер на плате (Этап 5)
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

> **Версия тулчейна.** `rknn-toolkit2` зафиксирован на `2.5.0` (колесо `cp312`
> x86_64, проверено на Python 3.12). Версия тулчейна на ПК **должна совпадать**
> с runtime на плате (`librknnmrt.so` + `rknn_server`), иначе `init_runtime`
> вернёт ошибку. Runtime 2.5.0 для Luckfox лежит в
> [`artifacts/rknn_runtime/`](../artifacts/rknn_runtime/).

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

## Проверка RKNN в симуляторе NPU на ПК

После конвертации `ONNX -> RKNN INT8` точность квантованной модели можно
проверить **без физической платы** — в программном симуляторе NPU, встроенном
в `rknn-toolkit2` (работает на CPU x86_64 под Linux).

```bash
# Ручной запуск: сборка .rknn, эмулятор NPU, сравнение с ONNX Runtime
python scripts/verify_rknn.py --accuracy-analysis

# Или через make
make verify-rknn
```

Скрипт [`scripts/verify_rknn.py`](../scripts/verify_rknn.py):

1. собирает `.rknn` INT8 из ONNX (или берёт готовый);
2. поднимает эмулятор NPU: `rknn.init_runtime(target=None)`;
3. прогоняет один и тот же вход через ONNX Runtime (float32) и эмулятор (INT8);
4. считает косинусное сходство и max abs diff;
5. проверяет размер `.rknn` (≤ 4 МБ) и пиковую память (`eval_memory`, ≤ 64 МБ);
6. опционально запускает `rknn.accuracy_analysis` (отчёт по слоям).

Ключевые параметры: `--min-cosine` (порог сходства), `--max-size-mb`,
`--max-memory-mb`, `--accuracy-analysis`, `--rebuild`.

Тот же сценарий покрыт тестом [`tests/test_rknn_simulator.py`](../tests/test_rknn_simulator.py).
Если `rknn-toolkit2` не установлен, тест **пропускается** (`pytest.importorskip`),
поэтому обычный `make test` не падает.

---

## Замер производительности на реальной плате (USB/ADB)

Если плата Luckfox Pico Plus подключена к хосту по USB, тот же скрипт умеет
замерить **реальную** latency/FPS/RTF на NPU RV1103 (не в симуляторе).

Предварительно на плате должен быть развёрнут runtime 2.5.0
(`librknnmrt.so` + `rknn_server` из [`artifacts/rknn_runtime/`](../artifacts/rknn_runtime/)).
Скрипт сам поднимает `rknn_server` через `start-stop-daemon --background`
(abstract-сокет `@transfer_proxy`, чтобы не конфликтовать с adbd на TCP 5555).

```bash
# Список подключённых устройств
python scripts/verify_rknn.py --list-devices
# или
make adb-devices

# Полный цикл: симулятор + замер на плате (warmup + N итераций + eval_memory)
python scripts/verify_rknn.py --target rv1103 --warmup 3 --iterations 20 \
    --max-latency-ms 200 --min-fps 5
# или
make bench-rknn
```

Скрипт:

1. проверяет `adb devices` и наличие `rknn_server` на плате;
2. при необходимости поднимает `rknn_server`;
3. прогревает NPU (`--warmup` прогонов, не учитываются);
4. замеряет latency по `--iterations` прогонам (mean/min/max/p50/p90);
5. считает FPS и RTF (`latency / --audio-seconds`);
6. читает пиковую память (`eval_memory`, только на устройстве);
7. сохраняет JSON-отчёт в `artifacts/rknn/perf_report.json`.

Ключевые параметры: `--target` (например `rv1103`), `--device-id`, `--warmup`,
`--iterations`, `--audio-seconds`, `--max-latency-ms`, `--min-fps`,
`--perf-report`, `--no-save-report`, `--no-auto-start-server`, `--list-devices`.

Коды возврата: `0` — все проверки пройдены, `1` — провалены пороги,
`2` — ошибка окружения (нет `adb`/платы/`rknn-toolkit2`).

Пример результата на Luckfox Pico Plus (RV1103, окно 2 с):

```
Latency, мс:     mean=64.0 min=61.6 max=64.8 p50=64.1 p90=64.7
FPS:             15.6
RTF:             0.0320 (окно 2.00 с)
Пиковая память:  0.50 МБ
Cosine similarity: 0.988930
```

Тесты бенчмарка — [`tests/test_rknn_benchmark.py`](../tests/test_rknn_benchmark.py)
(интеграционный замер пропускается, если плата не подключена).

---

## Распознавание WAV → текст (сквозная проверка)

Отдельный контур проверяет не задержку, а **качество распознавания**: подаёт
реальные WAV из `data/synth` и сравнивает распознанный текст с эталонным из
`metadata.csv`. Главная метрика — доля **точных совпадений**
(`recognized_text == reference_text`), дополнительно считаются CER и WER.

Единый пайплайн [`src/kan_asr/inference/recognize.py`](../src/kan_asr/inference/recognize.py):
`WAV → load_audio → audio_to_mel → fit_mel_length → модель → greedy_ctc_decode`.

```bash
# Хост (PyTorch, float32)
python scripts/recognize.py --device host
# или: make recognize-host

# Симулятор NPU (RKNN INT8, target=None)
python scripts/recognize.py --device simulator
# или: make recognize-sim

# Реальная плата Luckfox (USB/ADB)
python scripts/recognize.py --device rv1103
# или: make recognize-board

# Строгий gate: падать, если нет точных совпадений / CER выше порога
KAN_ASR_STRICT_RECOGNITION=1 python scripts/recognize.py --device host \
    --min-exact-match 0.5 --max-cer 0.3
```

Ключевые параметры: `--device {host,simulator,rv1103}`, `--wav`, `--cases`,
`--offset`, `--checkpoint`, `--onnx`, `--rknn`, `--target`, `--device-id`,
`--min-exact-match`, `--max-cer`, `--report`. Коды возврата: `0` — пороги
пройдены, `1` — провалены, `2` — ошибка окружения (нет чекпоинта/`rknn`/платы).

Пороги и число кейсов читаются из окружения:

| Переменная | Смысл | По умолчанию |
|---|---|---|
| `KAN_ASR_STRICT_RECOGNITION` | включить строгие проверки exact-match/CER | `0` |
| `KAN_ASR_MIN_EXACT_MATCH` | минимальная доля точных совпадений | `0.0` |
| `KAN_ASR_RECOGNITION_MAX_CER` | максимально допустимый CER | `1.0` |
| `KAN_ASR_RECOGNITION_CASES` | сколько кейсов брать в тестах | `3` |
| `KAN_ASR_SIM_HOST_MAX_CER` | допуск CER(хост, симулятор) | `0.5` |

Тот же хелпер `evaluate_recognition` переиспользуется этапами пайплайна:
после обучения сохраняется `artifacts/reports/train_recognition.json`, после
дообучения — `finetune_recognition.json` (с дельтой), после конвертации INT8 —
`convert_recognition.json`.

Тесты по уровням:

- [`tests/test_recognize_host.py`](../tests/test_recognize_host.py) — хост;
- [`tests/test_recognize_simulator.py`](../tests/test_recognize_simulator.py) — симулятор NPU
  (+ согласованность с хостом);
- [`tests/test_recognize_device.py`](../tests/test_recognize_device.py) — реальная плата (метка `hardware`).

Без чекпоинта/`rknn`/платы тесты **пропускаются**, поэтому обычный `make test`
не падает. Точечный запуск: `make test-recognition` (метка `recognition`),
`make test-hardware` (метка `hardware`).

## Следующие шаги после установки

1. Проверка установки зависимостей (`pytest tests/ -v`)
2. Генерация синтетического датасета
3. Инициализация весов из Whisper (WSP): `python scripts/wsp_init.py --whisper-state <path>`
4. Обучение: `python scripts/train.py`
5. Экспорт: `python scripts/export_onnx.py` -> `python scripts/convert_rknn.py`
6. Проверка в симуляторе NPU: `python scripts/verify_rknn.py`
7. Замер на плате (USB/ADB): `python scripts/verify_rknn.py --target rv1103`
8. Распознавание WAV → текст: `python scripts/recognize.py --device host`
   (затем `--device simulator` и `--device rv1103`)

См. также: [Архитектура](architecture.md), [Weight-Space Programming](wsp.md).
