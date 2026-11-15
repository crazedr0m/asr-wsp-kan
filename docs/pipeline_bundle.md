# Комплект запуска KAN-ASR: GPU (обучение) и Luckfox Pico Plus (RV1103 NPU)

> Перенесено из `KAN_ASR_Pipeline_Bundle/README.md` (легаси-каталог удалён).
> В новой структуре функциональность разнесена по пакету `kan_asr` и CLI-скриптам.

## Соответствие «бандл -> новая структура»

| Бандл (легаси) | Новая структура |
|---|---|
| `generate_dataset.py` | [`src/kan_asr/data/generate.py`](../src/kan_asr/data/generate.py) + [`scripts/generate_dataset.py`](../scripts/generate_dataset.py) |
| `train_and_export.py` | [`src/kan_asr/training/train.py`](../src/kan_asr/training/train.py) + [`src/kan_asr/export/export_onnx.py`](../src/kan_asr/export/export_onnx.py) |
| `convert_to_rknn.py` | [`src/kan_asr/export/convert_rknn.py`](../src/kan_asr/export/convert_rknn.py) + [`scripts/convert_rknn.py`](../scripts/convert_rknn.py) |
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
```

## Сборка C++ инференса для Luckfox Pico Plus

Компилируется кросс-компилятором `arm-rockchip830-linux-uclibcgnueabihf-g++` из официального
Luckfox SDK с линковкой `-lrknnmrt` (или `-lrknnrt` в зависимости от версии SDK).
См. [`edge/README.md`](../edge/README.md).
