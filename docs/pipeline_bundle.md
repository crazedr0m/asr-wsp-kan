# Комплект запуска KAN-ASR: GPU (обучение) и Luckfox Pico Plus (RV1103 NPU)

> Перенесено из `KAN_ASR_Pipeline_Bundle/README.md` (легаси-каталог удалён).
> В новой структуре функциональность разнесена по пакету `kan_asr` и CLI-скриптам.

## Соответствие «бандл -> новая структура»

| Бандл (легаси) | Новая структура |
|---|---|
| `generate_dataset.py` | [`src/kan_asr/data/generate.py`](../src/kan_asr/data/generate.py) + [`scripts/generate_dataset.py`](../scripts/generate_dataset.py) |
| `train_and_export.py` | [`src/kan_asr/training/train.py`](../src/kan_asr/training/train.py) + [`src/kan_asr/export/export_onnx.py`](../src/kan_asr/export/export_onnx.py) |
| `convert_to_rknn.py` | [`src/kan_asr/export/convert_rknn.py`](../src/kan_asr/export/convert_rknn.py) + [`scripts/convert_rknn.py`](../scripts/convert_rknn.py) |
| `verify_rknn.py` (симулятор NPU) | [`src/kan_asr/export/verify_rknn.py`](../src/kan_asr/export/verify_rknn.py) + [`scripts/verify_rknn.py`](../scripts/verify_rknn.py) |
| `luckfox_asr_inference.cpp` | [`edge/luckfox_asr_inference.cpp`](../edge/luckfox_asr_inference.cpp) |

## Порядок запуска на ПК (RTX 3060 / GTX 1050 Ti)

```bash
pip install -r requirements.txt && pip install -e .

# 1. Генерация синтетического датасета (Piper TTS + аугментации)
python scripts/generate_dataset.py

# 2. Инициализация весов из Whisper (WSP) и обучение
python scripts/wsp_init.py --whisper-state whisper_tiny.pt
python scripts/train.py --wsp-checkpoint artifacts/checkpoints/tiny_kan_wsp_initialized.pt

# 3. Экспорт ONNX -> RKNN INT8
python scripts/export_onnx.py
python scripts/convert_rknn.py

# 4. Проверка точности INT8 в симуляторе NPU (без платы)
python scripts/verify_rknn.py --accuracy-analysis
```

## Проверка в симуляторе NPU

Симулятор NPU встроен в `rknn-toolkit2` и работает на CPU x86_64 под Linux.
[`scripts/verify_rknn.py`](../scripts/verify_rknn.py) собирает `.rknn`, поднимает
эмулятор (`init_runtime(target=None)`), прогоняет вход через INT8-модель и
сравнивает выход с ONNX Runtime (float32) по косинусному сходству. Дополнительно
проверяются лимиты платы: размер `.rknn` ≤ 4 МБ и пиковая память ≤ 64 МБ
(`eval_memory`). Подробнее — в [`docs/setup.md`](setup.md).

> Требуется `rknn-toolkit2==2.5.0` (колесо `cp312` x86_64). Версия тулчейна
> должна совпадать с runtime на плате (`librknnmrt.so` + `rknn_server`).

## Замер производительности на реальной плате (USB/ADB)

Если Luckfox Pico Plus подключена к хосту по USB, тот же скрипт умеет мерить
реальную latency/FPS на NPU через `init_runtime(target='rv1103', device_id=...)`:

```bash
# 1. Убедиться, что плата видна по ADB
python scripts/verify_rknn.py --list-devices   # или: make adb-devices

# 2. Замер: 3 прогрева + 20 итераций, пороги 200 мс и 5 FPS
python scripts/verify_rknn.py --target rv1103 --warmup 3 --iterations 20 \
    --max-latency-ms 200 --min-fps 5
# или: make bench-rknn
```

Скрипт сам поднимает `rknn_server` на плате (если он не запущен), выполняет
прогрев, замеряет latency по итерациям, считает FPS и RTF (при окне 2 с),
снимает пиковую память (`eval_memory`) и сохраняет отчёт в
`artifacts/rknn/perf_report.json`. Коды возврата: `0` — пороги пройдены,
`1` — не пройдены, `2` — плата/ADB недоступны.

## Сборка C++ инференса для Luckfox Pico Plus

Компилируется кросс-компилятором `arm-rockchip830-linux-uclibcgnueabihf-g++` из официального
Luckfox SDK с линковкой `-lrknnmrt` (или `-lrknnrt` в зависимости от версии SDK).
См. [`edge/README.md`](../edge/README.md).
