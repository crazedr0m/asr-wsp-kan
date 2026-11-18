# Контур качества кода KAN-ASR

Документ описывает инструменты контроля качества, их конфигурацию, текущий
**baseline** (зафиксированные исключения) и стратегию **ratchet** (постепенное
ужесточение). Контур построен по принципу «мягкого старта»: ничего не блокирует
разработку сразу, но каждый новый коммит не должен ухудшать показатели.

## 1. Обзор инструментов

| Область | Инструмент | Где настроен | Блокирует? |
|---|---|---|---|
| Линт и стиль | **Ruff** (`check`) | `[tool.ruff]` в `pyproject.toml` | Да |
| Форматирование | **Ruff** (`format`) | `[tool.ruff.format]` | Да |
| Типизация | **mypy** | `[tool.mypy]` | Да |
| Тесты | **pytest** | `[tool.pytest.ini_options]` | Да |
| Покрытие | **pytest-cov** | `[tool.coverage.*]` | Да (порог) |
| Безопасность | **bandit** | `[tool.bandit]` | Да (порог medium) |
| Уязвимости зависимостей | **pip-audit** | CLI | Нет (информационно) |
| Секреты | **detect-secrets** | `.secrets.baseline` | Да |
| Сложность | **xenon** | CLI (пороги в `Makefile`) | Нет (информационно) |
| Мёртвый код | **vulture** | CLI | Нет (информационно) |
| Docstrings | **interrogate** | `[tool.interrogate]` | Да (порог 60%) |
| Гигиена коммитов | **pre-commit** | `.pre-commit-config.yaml` | Да (локально) |
| Conventional commits | **commitizen** | `[tool.commitizen]` | Нет (опционально) |

## 2. Локальный цикл

```bash
make lint        # Ruff: проверка стиля и ошибок
make format      # Ruff: автоформатирование и автофиксы
make typecheck   # mypy
make test-cov    # pytest + покрытие
make security    # bandit + pip-audit + detect-secrets
make complexity  # xenon + vulture + interrogate
make check       # всё сразу (lint + typecheck + test-cov + security + complexity)
make ci          # алиас check
```

Pre-commit устанавливается один раз:

```bash
./venv/bin/pre-commit install --install-hooks
./venv/bin/pre-commit install --hook-type commit-msg   # для commitizen
```

Хуки на `pre-commit` (быстрые): базовые проверки файлов, `ruff-check`, `ruff-format`,
`detect-secrets`. Хуки на `pre-push` (медленные): `mypy`, `bandit`.

## 3. CI (GitHub Actions)

Workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) запускается на
`push`/`pull_request` в `main`/`master`/`develop` и вручную (`workflow_dispatch`).

| Job | Блокирует? | Что делает |
|---|---|---|
| `lint` | Да | `ruff check` + `ruff format --check` |
| `typecheck` | Да | `mypy src scripts` |
| `test` | Да | pytest на Python 3.10 / 3.11 / 3.12, загрузка `coverage.xml` |
| `security` | **Нет** (`continue-on-error`) | bandit + pip-audit |
| `complexity` | **Нет** (`continue-on-error`) | xenon + vulture + interrogate |

Зависимости обновляются автоматически через
[`.github/dependabot.yml`](../.github/dependabot.yml) (pip + github-actions,
еженедельно, минорные/патчевые — сгруппированы в один PR).

## 4. Baseline (зафиксированные исключения)

Baseline — это осознанно принятые исключения для существующего кода. Они **не
отключают** проверки, а фиксируют текущее состояние, чтобы новые изменения
проверялись строго.

### 4.1 Ruff

- **Глобальные `ignore`**: `RUF001/002/003` (кириллица в тексте — осознанно),
  `N806` (имена тензорных размерностей `N, M, K, B, T, L`), `N812`
  (`import torch.nn.functional as F`), `RUF013` (дублирует mypy `no_implicit_optional`).
- **`per-file-ignores`**: тесты (`S101`, `T201`, `PT018`, `PT011`), CLI-скрипты
  (`T201`), `csrc` (`S101`, `T201`), модули-интерфейсы (`T201`), генераторы
  синтетики (`S311` — `random` для данных), `tiny_kan_asr.py` (`N801` — имена
  классов зафиксированы в чекпоинтах), `verify_rknn.py`/`handlers.py`
  (`S603` — `subprocess` с фиксированными аргументами).

### 4.2 mypy

Базовый режим намеренно мягкий. Ужесточение — через `[[tool.mypy.overrides]]`:

- **Строгие модули** (`disallow_untyped_defs`, `warn_return_any`):
  `kan_asr.data.mel`, `kan_asr.data.vocab`, `kan_asr.training.ctc`,
  `kan_asr.training.metrics`, `kan_asr.models.kan_layers`.
- **Сторонние библиотеки без стабов** (`ignore_missing_imports`):
  `scipy.*`, `rknn.*`, `onnxruntime`, `onnx.*`, `piper.*`.
- **Техдолг** (`disable_error_code`): `train`, `bot.app`, `recognize`, `generate`,
  `alignment`, `cuda_fastkan`, `verify_rknn`, `export_onnx`, `logging`, `paths`,
  `config`, `whisper_teacher`, `tiny_kan_asr` — отключены `assignment`,
  `no-any-return`, `attr-defined`, `arg-type`, `var-annotated`, `misc`,
  `call-overload`.

### 4.3 Покрытие тестами

- `--cov-fail-under=40` (фактическое покрытие с `branch=true` — ~41%).
- Исключения из отчёта (`[tool.coverage.report].exclude_also`): `if TYPE_CHECKING`,
  `raise NotImplementedError`, `if __name__ == "__main__"`, `@abstractmethod`,
  `...` и т.п.

### 4.4 Безопасность

- `[tool.bandit].skips`: `B101` (assert), `B311` (`random` для синтетики),
  `B614` (`torch.load` собственных чекпоинтов).
- Порог в CI/Makefile: `--severity-level medium` (низкоуровневые `B404`/`B603`
  для `subprocess` с фиксированными аргументами — информационные).
- `.secrets.baseline` — пустой (`results: {}`); ложное срабатывание на ADB-серийник
  в тесте помечено `# pragma: allowlist secret`.

### 4.5 Сложность и мейнтейнабилити

- **xenon**: `--max-absolute E --max-modules C --max-average B` (текущий максимум —
  ранг `E` у `train()` и `verify_rknn_simulator()`).
- **interrogate**: `fail-under=60` (фактически ~81.6%).
- **vulture**: `--min-confidence 80` (информационно).

## 5. Ratchet (постепенное ужесточение)

Правило: **новый код не должен ухудшать baseline**. Когда техдолг в модуле
устранён — исключение убирается, и модуль переходит в строгий режим.

Порядок ужесточения:

1. **mypy**: убрать модуль из блока «Техдолг» → исправить ошибки → при
   необходимости добавить в «Строгие модули».
2. **Покрытие**: поднимать `--cov-fail-under` по мере роста тестов (40 → 45 → 50 …).
3. **xenon**: понижать пороги (`E/C/B` → `D/C/B` → `C/B/A`) после рефакторинга
   сложных функций (`train`, `verify_rknn_simulator`, `benchmark_on_device`).
4. **interrogate**: поднимать `fail-under` (60 → 70 → 80 …).
5. **CI**: снимать `continue-on-error` с job'ов `security` и `complexity`, когда
   они стабильно зелёные.

## 6. Бэклог

- Консолидировать дублирующийся `greedy_ctc_decode` (`data/vocab.py` и
  `training/ctc.py`).
- Рассмотреть `pyright` в дополнение к mypy.
- Сделать `commitizen`/conventional commits обязательным gate (сейчас опционально).
- Добавить C++/CUDA-линтеры (`clang-format`/`clang-tidy`) для `csrc/` и `edge/`.
