# Makefile для KAN-ASR
#
# По умолчанию используется интерпретатор из venv (./venv/bin/python).
# Переопределить: make train PYTHON=python3

PYTHON ?= ./venv/bin/python
PIP    ?= ./venv/bin/pip
RUFF   ?= ./venv/bin/ruff
MYPY   ?= ./venv/bin/mypy
BANDIT ?= ./venv/bin/bandit
XENON  ?= ./venv/bin/xenon
VULTURE ?= ./venv/bin/vulture
INTERROGATE ?= ./venv/bin/interrogate
PIP_AUDIT ?= ./venv/bin/pip-audit
PRECOMMIT ?= ./venv/bin/pre-commit

.PHONY: help data wsp train export rknn verify-rknn bench-rknn adb-devices bot \
	recognize-host recognize-sim recognize-board test test-recognition test-hardware clean install \
	lint format typecheck security complexity test-cov pre-commit check ci

help:
	@echo "Доступные цели:"
	@echo "  data      — генерация синтетического датасета"
	@echo "  wsp       — Weight-Space Programming (инициализация весов Whisper)"
	@echo "  train     — обучение модели"
	@echo "  export    — экспорт в ONNX"
	@echo "  rknn      — конвертация в RKNN"
	@echo "  verify-rknn — проверка RKNN в симуляторе NPU"
	@echo "  bench-rknn — замер latency/FPS/RTF на плате (USB/ADB)"
	@echo "  adb-devices — список подключённых по ADB устройств"
	@echo "  bot       — запуск Telegram-бота"
	@echo "  recognize-host  — распознавание WAV→текст на хосте (PyTorch)"
	@echo "  recognize-sim   — распознавание WAV→текст в симуляторе NPU"
	@echo "  recognize-board — распознавание WAV→текст на плате (USB/ADB)"
	@echo "  test      — запуск тестов"
	@echo "  test-recognition — E2E-тесты распознавания (метка recognition)"
	@echo "  test-hardware — тесты на подключённой плате (метка hardware)"
	@echo ""
	@echo "Качество кода:"
	@echo "  lint      — Ruff: проверка стиля и ошибок"
	@echo "  format    — Ruff: автоформатирование и автофиксы"
	@echo "  typecheck — mypy: статическая типизация"
	@echo "  security  — bandit + pip-audit + detect-secrets"
	@echo "  complexity — radon/xenon + vulture + interrogate"
	@echo "  test-cov  — тесты с покрытием (pytest-cov)"
	@echo "  pre-commit — прогон всех pre-commit хуков по всем файлам"
	@echo "  check     — всё сразу: lint + typecheck + test-cov + security + complexity"
	@echo "  ci        — то же, что check (для CI)"
	@echo ""
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

verify-rknn:
	@echo "Проверка RKNN в симуляторе NPU..."
	$(PYTHON) scripts/verify_rknn.py

bench-rknn:
	@echo "Замер latency/FPS/RTF на плате (target=rv1103, USB/ADB)..."
	$(PYTHON) scripts/verify_rknn.py --target rv1103

adb-devices:
	@echo "Список устройств ADB..."
	$(PYTHON) scripts/verify_rknn.py --list-devices

bot:
	@echo "Запуск Telegram-бота..."
	$(PYTHON) scripts/run_bot.py

recognize-host:
	@echo "Распознавание WAV→текст на хосте (PyTorch)..."
	$(PYTHON) scripts/recognize.py --device host

recognize-sim:
	@echo "Распознавание WAV→текст в симуляторе NPU..."
	$(PYTHON) scripts/recognize.py --device simulator

recognize-board:
	@echo "Распознавание WAV→текст на плате (USB/ADB)..."
	$(PYTHON) scripts/recognize.py --device rv1103

test:
	@echo "Запуск тестов..."
	$(PYTHON) -m pytest tests/ -v

test-recognition:
	@echo "E2E-тесты распознавания WAV→текст..."
	$(PYTHON) -m pytest tests/ -v -m recognition

test-hardware:
	@echo "Тесты на подключённой плате Luckfox..."
	$(PYTHON) -m pytest tests/ -v -m hardware

# --- Качество кода ---------------------------------------------------------

lint:
	@echo "Ruff: проверка стиля и ошибок..."
	$(RUFF) check .
	$(RUFF) format --check .

format:
	@echo "Ruff: автоформатирование и автофиксы..."
	$(RUFF) check --fix .
	$(RUFF) format .

typecheck:
	@echo "mypy: статическая типизация..."
	$(MYPY) src scripts

security:
	@echo "bandit: статический анализ безопасности (порог: medium)..."
	$(BANDIT) -c pyproject.toml -r src scripts --severity-level medium
	@echo "pip-audit: уязвимости в зависимостях..."
	-$(PIP_AUDIT) -r requirements.txt
	@echo "detect-secrets: поиск секретов..."
	$(PRECOMMIT) run detect-secrets --all-files

complexity:
	@echo "xenon: пороги цикломатической сложности (baseline: E/C/B)..."
	-$(XENON) --max-absolute E --max-modules C --max-average B src
	@echo "vulture: поиск мёртвого кода..."
	-$(VULTURE) src scripts --min-confidence 80
	@echo "interrogate: покрытие docstring'ами..."
	$(INTERROGATE) -c pyproject.toml src

test-cov:
	@echo "Тесты с покрытием..."
	$(PYTHON) -m pytest tests/ -v

pre-commit:
	@echo "Прогон pre-commit по всем файлам..."
	$(PRECOMMIT) run --all-files

check: lint typecheck test-cov security complexity
	@echo "Все проверки качества пройдены."

ci: check

clean:
	@echo "Очистка временных файлов..."
	rm -rf data/synth/wavs/*.wav data/calib/*.npy artifacts/checkpoints/*.pt artifacts/onnx/*.onnx artifacts/rknn/*.rknn

install:
	@echo "Установка зависимостей..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
