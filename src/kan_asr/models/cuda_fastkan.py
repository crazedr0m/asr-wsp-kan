"""
Кастомное CUDA-ядро для FastKANConv1x1 (fused RBF basis + SiLU).

Перенесено из FastKAN_CUDA_Kernel/cuda_fastkan.py (легаси-каталог).
Исходники ядра теперь лежат в ``csrc/`` в корне проекта.

Класс :class:`CudaFastKANConv1x1` — drop-in замена для
:class:`kan_asr.models.tiny_kan_asr.FastKANConv1x1`. При экспорте в ONNX
(``torch.onnx.is_in_onnx_export()``) или работе на CPU автоматически
переключается на стандартные операции PyTorch для совместимости с RKNN.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch.utils.cpp_extension доступен только при наличии torch с CUDA-тулчейном
    from torch.utils.cpp_extension import load as _load_ext
except Exception:  # pragma: no cover - окружение без cpp_extension
    _load_ext = None


def _find_csrc_dir() -> str:
    """Ищет каталог ``csrc/`` с исходниками CUDA-ядра.

    Порядок поиска:
    1. Переменная окружения ``KAN_ASR_CSRC``.
    2. ``<project_root>/csrc`` (на 3 уровня выше этого файла: models → kan_asr → src → root).
    3. ``<cwd>/csrc``.
    """
    env_dir = os.environ.get("KAN_ASR_CSRC")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    here = os.path.dirname(os.path.abspath(__file__))
    # src/kan_asr/models/cuda_fastkan.py -> ../../../csrc
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidate = os.path.join(project_root, "csrc")
    if os.path.isdir(candidate):
        return candidate

    cwd_candidate = os.path.join(os.getcwd(), "csrc")
    if os.path.isdir(cwd_candidate):
        return cwd_candidate

    return candidate  # вернём ожидаемый путь, даже если его нет


_CSRC_DIR = _find_csrc_dir()
_cuda_ext = None


def get_cuda_ext():
    """JIT-компилирует и кэширует CUDA-расширение при первом вызове."""
    global _cuda_ext
    if _cuda_ext is None:
        if _load_ext is None:
            raise RuntimeError(
                "torch.utils.cpp_extension недоступен: невозможно собрать CUDA-ядро."
            )
        sources = [
            os.path.join(_CSRC_DIR, "fastkan_kernel.cu"),
            os.path.join(_CSRC_DIR, "bindings.cpp"),
        ]
        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            raise FileNotFoundError(
                f"Не найдены исходники CUDA-ядра: {missing}. "
                f"Ожидался каталог csrc/ (текущий: {_CSRC_DIR})."
            )
        _cuda_ext = _load_ext(
            name="fastkan_cuda_ext",
            sources=sources,
            extra_include_paths=[_CSRC_DIR],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _cuda_ext


class FusedFastKANBasisFunction(torch.autograd.Function):
    """Autograd-обёртка над fused CUDA-ядром (forward + backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, grid: torch.Tensor, inv_denom: float):
        ext = get_cuda_ext()
        rbf_out, silu_out = ext.forward(x.contiguous(), grid.contiguous(), inv_denom)
        ctx.save_for_backward(x, grid, rbf_out)
        ctx.inv_denom = inv_denom
        return rbf_out, silu_out

    @staticmethod
    def backward(ctx, grad_rbf: torch.Tensor, grad_silu: torch.Tensor):
        x, grid, rbf_out = ctx.saved_tensors
        ext = get_cuda_ext()
        grad_x = ext.backward(
            x.contiguous(),
            grid.contiguous(),
            rbf_out.contiguous(),
            grad_rbf.contiguous(),
            grad_silu.contiguous(),
            ctx.inv_denom,
        )
        return grad_x, None, None


class CudaFastKANConv1x1(nn.Module):
    """
    Drop-in замена для FastKANConv1x1 с кастомным CUDA-ядром.

    Поддерживает FP32 (для GTX 1050 Ti), FP16 и BFloat16 (для RTX 3060).
    При экспорте в ONNX (``torch.onnx.is_in_onnx_export()``) автоматически
    переключается на стандартные операции PyTorch для совместимости с RKNN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_grids: int = 6,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_grids = num_grids

        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)
        self.denom = (grid_max - grid_min) / (num_grids - 1)
        self.inv_denom = 1.0 / self.denom

        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.rbf_conv = nn.Conv2d(in_channels * num_grids, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # При экспорте в ONNX или работе на CPU используем стандартный граф
        if torch.onnx.is_in_onnx_export() or not x.is_cuda:
            B, C, H, T = x.shape
            base_out = self.base_conv(F.silu(x))
            grid_5d = self.grid.view(1, 1, self.num_grids, 1, 1)
            diff = (x.unsqueeze(2) - grid_5d) * self.inv_denom
            rbf = torch.exp(-diff * diff).reshape(B, C * self.num_grids, H, T)
            return base_out + self.rbf_conv(rbf)

        # На GPU вызываем наше Fused CUDA-ядро + аппаратный cuBLAS Conv2d 1x1
        rbf_flat, silu_x = FusedFastKANBasisFunction.apply(x, self.grid, self.inv_denom)
        return self.base_conv(silu_x) + self.rbf_conv(rbf_flat)
