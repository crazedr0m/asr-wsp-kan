# Makefile для KAN-ASR
#
# По умолчанию используется интерпретатор из venv (./venv/bin/python).
# Переопределить: make train PYTHON=python3

PYTHON ?= ./venv/bin/python
PIP    ?= ./venv/bin/pip

.PHONY: help data wsp train export rknn bot test clean install

help:
	@echo "Доступные цели:"
	@echo "  data      — генерация синтетического датасета"
	@echo "  wsp       — Weight-Space Programming (инициализация весов Whisper)"
	@echo "  train     — обучение модели"
	@echo "  export    — экспорт в ONNX"
	@echo "  rknn      — конвертация в RKNN"
	@echo "  bot       — запуск Telegram-бота"
	@echo "  test      — запуск тестов"
	@echo "  clean     — удаление временных файлов"
	@echo "  install   — установка зависимостей"

data:
	@echo "Генерация синтетического датасета..."
	$(PYTHON) scripts/generate_dataset.py

wsp:
	@echo "Weight-Space Programming (инициализация весов Whisper-tiny)..."
	$(PYTHON) scripts/wsp_init.py

train:
	@echo "Обучение модели..."
	$(PYTHON) scripts/train.py

export:
	@echo "Экспорт в ONNX..."
	$(PYTHON) scripts/export_onnx.py

rknn:
	@echo "Конвертация в RKNN..."
	$(PYTHON) scripts/convert_rknn.py

bot:
	@echo "Запуск Telegram-бота..."
	$(PYTHON) scripts/run_bot.py

test:
	@echo "Запуск тестов..."
	$(PYTHON) -m pytest tests/ -v

clean:
	@echo "Очистка временных файлов..."
	rm -rf data/synth/wavs/*.wav data/calib/*.npy artifacts/checkpoints/*.pt artifacts/onnx/*.onnx artifacts/rknn/*.rknn

install:
	@echo "Установка зависимостей..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
