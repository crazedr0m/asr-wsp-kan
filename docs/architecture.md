# Архитектура KAN-ASR

Обзор системы распознавания речи на базе KAN (Kolmogorov–Arnold Networks)
для встраиваемой платформы Luckfox Pico Plus (RV1103 NPU).

## 1. Цель

Перенести знания из предобученного **Whisper-tiny** в компактную модель
`TinyKAN_ASR` (~1.2M параметров) методом **Weight-Space Programming (WSP)**,
дообучить её (CTC + CKA-выравнивание), экспортировать в ONNX, сконвертировать
в RKNN INT8 и запустить на NPU RV1103.

## 2. Поток данных (WSP-контур)

```mermaid
graph LR
    A[Whisper-tiny weights] --> B[wsp/extract.py SVD]
    B --> C[wsp/project.py аппроксимация MLP в FastKAN]
    C --> D[wsp/initialize.py TinyKAN_ASR]
    D --> E[training/train.py CTC + CKA]
    E --> F[export/export_onnx.py]
    F --> G[export/convert_rknn.py INT8]
    G --> H[edge/luckfox_asr_inference.cpp NPU]
```

## 3. Структура пакета

| Модуль | Назначение |
|---|---|
| [`models/kan_layers.py`](../src/kan_asr/models/kan_layers.py) | `PureFastKANLayer`, `PureChebyKANLayer`, `NPUFriendlyMoEKAN` |
| [`models/tiny_kan_asr.py`](../src/kan_asr/models/tiny_kan_asr.py) | `TinyKAN_ASR`, `FastKANConv1x1`, `DepthwiseSeparableConv1D_as_2D` — **источник истины архитектуры** |
| [`models/whisper_teacher.py`](../src/kan_asr/models/whisper_teacher.py) | `WhisperTinyTeacher` — автономный энкодер-учитель |
| [`models/cuda_fastkan.py`](../src/kan_asr/models/cuda_fastkan.py) | `CudaFastKANConv1x1` — обёртка над CUDA-ядром (JIT) |
| [`wsp/`](../src/kan_asr/wsp/) | Ядро WSP: `extract`, `project`, `initialize`, `hyper_adapter` |
| [`data/`](../src/kan_asr/data/) | `vocab`, `mel`, `augment`, `generate`, `dataset` |
| [`training/`](../src/kan_asr/training/) | `ctc`, `alignment`, `metrics`, `train` |
| [`export/`](../src/kan_asr/export/) | `export_onnx`, `convert_rknn` |
| [`bot/`](../src/kan_asr/bot/) | Telegram-бот (Active Learning): `app`, `handlers`, `finetune_worker` |
| [`utils/`](../src/kan_asr/utils/) | `paths` (резолвер), `logging` |

## 4. Архитектура TinyKAN_ASR

```
Вход:  mel [B, 40, 1, T]
  │
  ├─ encoder: 3× DepthwiseSeparableConv1D_as_2D
  │     40→64 (stride 2) → 64→128 (stride 2) → 128→128 (stride 1)
  │
  ├─ kan1: FastKANConv1x1(128→128) + BatchNorm2d  (residual)
  ├─ kan2: FastKANConv1x1(128→128) + BatchNorm2d  (residual)
  │
  └─ head: Conv2d 1x1 (128 → vocab_size)
Выход: CTC-логиты [B, vocab_size, 1, T_out]
```

`FastKANConv1x1` реализует базис SiLU + RBF-Гауссианы:

```
y = base_conv(silu(x)) + rbf_conv(rbf(x))
rbf(x) = exp(-((x - grid) / denom)^2)
```

Атрибуты `base_conv`, `rbf_conv`, `grid`, `denom` используются WSP-кодом
для аналитической записи весов.

## 5. Кастомное CUDA-ядро

[`csrc/`](../csrc/) содержит fused-ядро RBF-базиса + SiLU (forward/backward),
JIT-компилируемое через `torch.utils.cpp_extension.load`:

* [`csrc/fastkan_kernel.cu`](../csrc/fastkan_kernel.cu) — ядра (шаблоны float/Half/BFloat16)
* [`csrc/fastkan_kernel.h`](../csrc/fastkan_kernel.h) — объявления
* [`csrc/bindings.cpp`](../csrc/bindings.cpp) — C++-обёртки + pybind11
* [`csrc/setup.py`](../csrc/setup.py) — сборка `CUDAExtension`

[`CudaFastKANConv1x1`](../src/kan_asr/models/cuda_fastkan.py) автоматически
откатывается на стандартный PyTorch-граф при экспорте в ONNX или работе на CPU.

## 6. Edge-инференс

[`edge/luckfox_asr_inference.cpp`](../edge/luckfox_asr_inference.cpp) — нативный
раннер RKNN для RV1103 (вход `[1,40,1,200]`, выход `[1,35,1,50]`, greedy CTC-декод).
Сборка описана в [`edge/CMakeLists.txt`](../edge/CMakeLists.txt) и
[`edge/README.md`](../edge/README.md).

## 7. Конфигурация

Все гиперпараметры вынесены в [`configs/`](../configs/):
`model.yaml`, `wsp.yaml`, `train.yaml`, `bot.yaml`.

## 8. Связанные документы

* [Weight-Space Programming](wsp.md)
* [План миграции структуры](../plans/project_structure_migration.md)
* [Аудит миграции](../plans/migration_audit.md)
