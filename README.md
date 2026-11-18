# KAN-ASR: Weight-Space Programming для компактного распознавания речи

## 🎯 Цель проекта

Разработка компактной акустической модели распознавания речи на базе сетей Колмогорова–Арнольда (KAN) без использования B-сплайнов, с портированием на Rockchip RV1103 NPU (Luckfox Pico Plus).

**Ключевая особенность:** Weight-Space Programming (WSP) — инициализация весов модели из предобученного Whisper-tiny через SVD-проекцию и функциональную аппроксимацию, а не обучение с нуля.

## 📁 Структура проекта

```
voice-recognition/
├── src/kan_asr/          # Пакет с кодом
│   ├── models/           # Архитектуры (FastKAN, TinyKAN_ASR, WhisperTeacher)
│   ├── wsp/              # Weight-Space Programming (extract, project, initialize, hyper_adapter)
│   ├── data/             # Данные и признаки (vocab, mel, augment, generate, dataset)
│   ├── training/         # Обучение и метрики (ctc, alignment, metrics, train)
│   ├── export/           # Экспорт и квантование (export_onnx, convert_rknn)
│   ├── bot/              # Telegram-бот (app, handlers, finetune_worker)
│   └── utils/            # Утилиты (paths, logging, config)
├── csrc/                 # CUDA-расширения
├── edge/                 # C++ раннер для Luckfox
├── scripts/              # CLI-точки входа
├── data/                 # Данные (synth, calib, samples, raw, real)
├── artifacts/            # Артефакты (piper, onnx, rknn, checkpoints)
├── docs/                 # Документация
├── tests/                # Тесты
├── configs/              # Конфигурация (YAML)
├── archive/              # Архивы (.zip)
├── .gitignore            # Игнорирование для git
├── pyproject.toml        # Метаданные пакета
├── requirements.txt      # Runtime-зависимости
└── Makefile              # Цели сборки
```

## 🚀 Быстрый старт

### Установка

```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в editable-режиме
pip install -e .
```

### Weight-Space Programming (WSP)

```bash
# Инициализация модели весами Whisper-tiny
python scripts/wsp_init.py

# Обучение с инициализированными весами
python scripts/train.py

# Экспорт в ONNX
python scripts/export_onnx.py

# Конвертация в RKNN
python scripts/convert_rknn.py
```

### Telegram-бот (Active Learning)

```bash
python scripts/run_bot.py
```

## 🧪 Разработка

Установка инструментов разработки и pre-commit:

```bash
pip install -r requirements-dev.txt
pip install -e .
./venv/bin/pre-commit install --install-hooks
```

Основные цели Makefile для контроля качества:

```bash
make lint        # Ruff: проверка стиля и ошибок
make format      # Ruff: автоформатирование и автофиксы
make typecheck   # mypy: статическая типизация
make test-cov    # pytest + покрытие
make security    # bandit + pip-audit + detect-secrets
make complexity  # xenon + vulture + interrogate
make check       # всё сразу (lint + typecheck + test-cov + security + complexity)
```

Контур построен по принципу «мягкого старта»: существующий техдолг зафиксирован
как baseline, а новые изменения проверяются строго. Подробности — в
[docs/quality.md](docs/quality.md).

## 📖 Документация

- [Техническое задание](docs/KAN_ASR_Docs_Markdown/TZ_KAN_ASR_Luckfox.md)
- [План разработки](docs/KAN_ASR_Docs_Markdown/Plan_Razrabotki_KAN_ASR.md)
- [Архитектура проекта](docs/architecture.md)
- [Weight-Space Programming (WSP)](docs/wsp.md)
- [CKA-выравнивание](docs/alignment.md)
- [Комплект запуска (pipeline bundle)](docs/pipeline_bundle.md)
- [Настройка окружения](docs/setup.md)
- [Контур качества кода](docs/quality.md)
- [CUDA-ядро](csrc/README.md)
- [Edge-инференс на Luckfox](edge/README.md)

## 🛠️ Лицензия

MIT
