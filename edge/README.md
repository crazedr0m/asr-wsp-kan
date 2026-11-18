# Нативный инференс KAN-ASR для Luckfox Pico Plus (RV1103 NPU)

Минимальный C++ раннер, который загружает INT8 `.rknn`-модель, прогоняет
Mel-спектрограмму через NPU (0.5 TOPS) и декодирует выход greedy-CTC.

## Состав
- `luckfox_asr_inference.cpp` — точка входа: `rknn_init` → `rknn_inputs_set` →
  `rknn_run` → `rknn_outputs_get` → greedy CTC decode.
- `CMakeLists.txt` — кросс-сборка под `arm-rockchip830-linux-uclibcgnueabihf`.

## Форматы тензоров
- Вход: `[1, 40, 1, 200]` (2 секунды Log-Mel, шаг 10 мс), `RKNN_TENSOR_FLOAT32`, NCHW.
- Выход: `[1, 35, 1, 50]` (логиты CTC), NPU сам деквантует INT8 → float32.

Словарь (`VOCAB`) должен **строго совпадать** с
[`kan_asr.data.vocab`](../../src/kan_asr/data/vocab.py).

## Сборка
Требуется официальный Luckfox SDK с кросс-компилятором
`arm-rockchip830-linux-uclibcgnueabihf-g++` и RKNN-библиотекой
(`librknnmrt` или `librknnrt` в зависимости от версии SDK).

```bash
cmake -B build \
  -DCMAKE_C_COMPILER=arm-rockchip830-linux-uclibcgnueabihf-gcc \
  -DCMAKE_CXX_COMPILER=arm-rockchip830-linux-uclibcgnueabihf-g++ \
  -DRKNN_SDK_ROOT=/path/to/luckfox/rknn
cmake --build build
```

## Запуск на устройстве
```bash
./luckfox_kan_asr tiny_kan_asr_rv1103.rknn
```

## Проверка точности INT8 до прошивки

Перед запуском на плате точность квантованной модели проверяется в симуляторе
NPU на ПК (без физического устройства):

```bash
python scripts/verify_rknn.py --accuracy-analysis
```

Скрипт собирает `.rknn`, поднимает эмулятор (`init_runtime(target=None)`),
сравнивает выход INT8 с ONNX Runtime (float32) по косинусному сходству и
проверяет лимиты платы: размер `.rknn` ≤ 4 МБ и пиковая память ≤ 64 МБ.
См. [`docs/setup.md`](../docs/setup.md).

## Распознавание WAV → текст на подключённой плате (USB/ADB)

Тот же контур, что и на хосте/в симуляторе, но с инференсом на NPU RV1103.
Mel считается на хосте (C++-Mel на устройстве ещё не реализован) и подаётся
в INT8-модель по USB/ADB:

```bash
python scripts/recognize.py --device rv1103
# или: make recognize-board
```

Скрипт сам поднимает `rknn_server`, считает exact-match/CER против эталона из
`data/synth/metadata.csv` и пишет отчёт `artifacts/reports/recognition_report.json`.
Тесты уровня платы — [`tests/test_recognize_device.py`](../tests/test_recognize_device.py)
(метка `hardware`, `make test-hardware`); пропускаются без `rknn`/`adb`/платы.

## Замер производительности на подключённой плате (USB/ADB)

Когда Luckfox Pico Plus подключена к хосту по USB, можно измерить реальную
latency/FPS на NPU, не прошивая устройство:

```bash
python scripts/verify_rknn.py --list-devices          # проверить ADB
python scripts/verify_rknn.py --target rv1103 \
    --warmup 3 --iterations 20 --max-latency-ms 200 --min-fps 5
# или: make bench-rknn
```

Скрипт поднимает `rknn_server` на плате, выполняет прогрев и N итераций
инференса, считает latency/FPS/RTF и пиковую память, сохраняет отчёт в
`artifacts/rknn/perf_report.json`. Требуется совпадение версий тулчейна на ПК
и runtime на плате (`rknn-toolkit2==2.5.0` ↔ `librknnmrt.so`/`rknn_server` 2.5.0).

## Что осталось реализовать
- Заполнение `mel_input` реальными данными с микрофона (ALSA → FFT → Mel-фильтры).
  Сейчас буфер инициализирован нулями (заглушка `TODO` в коде).
