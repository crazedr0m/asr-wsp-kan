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

## Что осталось реализовать
- Заполнение `mel_input` реальными данными с микрофона (ALSA → FFT → Mel-фильтры).
  Сейчас буфер инициализирован нулями (заглушка `TODO` в коде).
