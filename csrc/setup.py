"""Сборка кастомного CUDA-расширения FastKAN.

Перенесено из FastKAN_CUDA_Kernel/setup.py (легаси-каталог).

Использование:
    # JIT-компиляция при первом импорте kan_asr.models.cuda_fastkan (рекомендуется)
    # либо явная сборка в систему:
    cd csrc && python setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastkan_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="fastkan_cuda_ext",
            sources=["fastkan_kernel.cu", "bindings.cpp"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
