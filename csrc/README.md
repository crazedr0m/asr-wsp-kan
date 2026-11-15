# Кастомное CUDA-ядро для FastKANConv1x1

> Перенесено из `FastKAN_CUDA_Kernel/README_CUDA.md` (легаси-каталог удалён).
> Python-обёртка: [`src/kan_asr/models/cuda_fastkan.py`](../src/kan_asr/models/cuda_fastkan.py).
> Исходники ядра: этот каталог (`fastkan_kernel.cu`, `fastkan_kernel.h`, `bindings.cpp`).

## В чём оптимизация?

В стандартном PyTorch вычисление базиса RBF (`(x.unsqueeze(2) - grid) / denom` -> `exp(-diff^2)`)
порождает 4 отдельных запуска CUDA-ядер (kernel launch overhead) и записывает промежуточный
5D-тензор `[B, C, Grid, 1, T]` в глобальную видеопамять (VRAM). На **GTX 1050 Ti** с её узкой
шиной памяти (112 ГБ/с) это становится главным бутылочным горлышком.

Наше ядро `fastkan_kernel.cu` объединяет (**fuses**):

1. Вычисление `SiLU(x)` для базовой ветки.
2. Расчёт всех `num_grids` гауссиан в регистрах одного потока GPU.
3. Прямой (Forward) и обратный (Backward) проходы с переиспользованием посчитанного RBF-кэша.

А матричное умножение (`Conv2d 1x1`) отдаётся штатному высокооптимизированному **cuBLAS / cuDNN**.

## Совместимость с RKNN (Luckfox Pico Plus)

Класс `CudaFastKANConv1x1` проверяет флаг `torch.onnx.is_in_onnx_export()`. При обучении на GPU
работает C++/CUDA ядро, а при вызове `torch.onnx.export(...)` автоматически собирается стандартный
ONNX-граф для `rknn-toolkit2`.

## Сборка и запуск

Убедитесь, что установлен NVIDIA CUDA Toolkit (`nvcc --version`), совпадающий с версией CUDA
в вашем PyTorch.

```bash
# Вариант 1 (рекомендуется): JIT-компиляция при первом импорте
#   kan_asr.models.cuda_fastkan — ничего делать не нужно, ядро соберётся само.
#   Проверка + бенчмарк:
python csrc/test_and_benchmark.py

# Вариант 2: явная сборка расширения в систему
cd csrc && python setup.py install
```

Каталог с исходниками можно переопределить переменной окружения `KAN_ASR_CSRC`.
