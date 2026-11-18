#!/usr/bin/env python3
"""Проверка точности и бенчмарк кастомного CUDA-ядра FastKAN.

Перенесено из ``FastKAN_CUDA_Kernel/test_and_benchmark.py`` (легаси-каталог удалён).
Импортирует обёртку из пакета ``kan_asr`` вместо локального ``cuda_fastkan``.

Запуск (из корня проекта, на машине с CUDA):
    python csrc/test_and_benchmark.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    )

from kan_asr.models.cuda_fastkan import CudaFastKANConv1x1


class ReferenceFastKANConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, num_grids=6, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids).view(1, 1, num_grids, 1, 1)
        self.register_buffer("grid", grid)
        self.denom = (grid_max - grid_min) / (num_grids - 1)
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.rbf_conv = nn.Conv2d(in_channels * num_grids, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, T = x.shape
        base_out = self.base_conv(F.silu(x))
        diff = (x.unsqueeze(2) - self.grid) / self.denom
        rbf = torch.exp(-diff * diff).reshape(B, C * self.num_grids, H, T)
        return base_out + self.rbf_conv(rbf)


def run_test():
    if not torch.cuda.is_available():
        print("CUDA недоступна! Запустите на машине с GTX 1050 Ti или RTX 3060.")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    B, C, T, G = 32, 128, 250, 6
    x_ref = torch.randn(B, C, 1, T, device=device, requires_grad=True)
    x_cuda = x_ref.detach().clone().requires_grad_(True)

    ref_layer = ReferenceFastKANConv1x1(C, C, num_grids=G).to(device)
    cuda_layer = CudaFastKANConv1x1(C, C, num_grids=G).to(device)

    # Копируем идентичные веса
    cuda_layer.load_state_dict(ref_layer.state_dict())

    # 1. Проверка точности Forward и Backward
    out_ref = ref_layer(x_ref)
    out_cuda = cuda_layer(x_cuda)

    loss_ref = out_ref.sum()
    loss_cuda = out_cuda.sum()
    loss_ref.backward()
    loss_cuda.backward()

    fwd_err = (out_ref - out_cuda).abs().max().item()
    bwd_err = (x_ref.grad - x_cuda.grad).abs().max().item()
    print(f"Max Forward Error:  {fwd_err:.2e}")
    print(f"Max Backward Error: {bwd_err:.2e}")
    assert fwd_err < 1e-4, "Ошибка расхождения forward-градиентов!"
    assert bwd_err < 1e-3, "Ошибка расхождения backward-градиентов!"

    # 2. Замер скорости и потребления VRAM
    iters = 200
    for _ in range(20):  # прогрев + JIT компиляция
        y = cuda_layer(x_cuda)
        y.sum().backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = ref_layer(x_ref)
        y.sum().backward()
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - t0) * 1000 / iters
    ref_mem = torch.cuda.max_memory_allocated() / (1024**2)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = cuda_layer(x_cuda)
        y.sum().backward()
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - t0) * 1000 / iters
    cuda_mem = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"\n[PyTorch Reference] Time: {ref_time:.3f} ms/iter | Peak VRAM: {ref_mem:.1f} MB")
    print(f"[Custom CUDA Kernel] Time: {cuda_time:.3f} ms/iter | Peak VRAM: {cuda_mem:.1f} MB")
    print(f"Ускорение: {ref_time / cuda_time:.2f}x | Экономия VRAM: {ref_mem - cuda_mem:.1f} MB")


if __name__ == "__main__":
    run_test()
