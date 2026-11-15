"""
Базовые слои KAN.

Этот файл содержит реализации PureFastKANLayer, PureChebyKANLayer, NPUFriendlyMoEKAN.

Порт из ``kan_asr_models.py`` (легаси-файл корня) — реализации без сплайнов,
совместимые с ONNX/RKNN и NPU RV1103.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureFastKANLayer(nn.Module):
    """
    Базовый слой FastKAN (радиально-базисные функции / Гауссианы).

    Реализует базис SiLU + RBF-Гауссианы одним большим GEMM (MatMul),
    что удобно для квантования в INT8 на NPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        # Фиксированная сетка центров (буфер, не обучается, зашивается в константы NPU)
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)

        # Ширина Гауссианы
        self.denom = (grid_max - grid_min) / (num_grids - 1)

        # Веса для RBF-базиса и базовой линейной ветки (SiLU/Linear)
        self.rbf_weight = nn.Parameter(torch.randn(out_features, in_features * num_grids) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Time, In] -> сохраняем форму для решейпа
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # [N, In]

        # 1. Базовая ветка (residual-подобная, стабилизирует градиенты)
        # На NPU SiLU легко квантуется в INT8
        base_out = F.linear(F.silu(x_flat), self.base_weight)

        # 2. RBF ветка (разворачиваем в плоский вектор для одного MatMul)
        # x_flat: [N, In, 1], grid: [1, 1, Grid]
        diff = (x_flat.unsqueeze(-1) - self.grid) / self.denom
        rbf = torch.exp(-diff * diff)  # [N, In, Grid]

        # Сплющиваем последние два измерения и делаем один большой GEMM (MatMul)
        rbf_flat = rbf.reshape(-1, self.in_features * self.num_grids)
        rbf_out = F.linear(rbf_flat, self.rbf_weight)

        out = base_out + rbf_out
        return out.reshape(*orig_shape[:-1], self.out_features)


class PureChebyKANLayer(nn.Module):
    """
    Базовый слой ChebyKAN (полиномы Чебышёва).

    Альтернативная реализация для экспериментов. Полиномы вычисляются
    развёрнутым циклом (без динамических списков) для совместимости с ONNX/RKNN.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Веса для степеней полинома
        self.cheby_weights = nn.Parameter(
            torch.randn(out_features, in_features * (degree + 1)) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = torch.tanh(x.reshape(-1, self.in_features))  # Строго [-1, 1]

        # Вычисляем полиномы Чебышёва без динамических списков (развернутый цикл для ONNX/RKNN)
        T0 = torch.ones_like(x)
        T1 = x
        cheby_list = [T0, T1]

        for _ in range(2, self.degree + 1):
            Tn = 2 * x * cheby_list[-1] - cheby_list[-2]
            cheby_list.append(Tn)

        # Конкатенация по последнему измерению -> [N, In * (degree + 1)]
        cheby_basis = torch.stack(cheby_list, dim=-1).reshape(
            -1, self.in_features * (self.degree + 1)
        )

        out = F.linear(cheby_basis, self.cheby_weights)
        return out.reshape(*orig_shape[:-1], self.out_features)


class NPUFriendlyMoEKAN(nn.Module):
    """
    NPU-совместимый Soft-MoE на базе FastKAN (без ветвлений и TopK).

    Лёгкий роутер предсказывает веса экспертов для каждого временного фрейма;
    все эксперты вычисляются параллельно, затем берётся взвешенная сумма.
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        # Легкий роутер: предсказывает веса экспертов для каждого временного фрейма
        self.router = nn.Linear(in_features, num_experts)

        # Эксперты на базе FastKAN
        self.experts = nn.ModuleList(
            [
                PureFastKANLayer(in_features, out_features, num_grids=6)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Time, Features]
        # Веса экспертов: [Batch, Time, Num_Experts]
        gates = F.softmax(self.router(x), dim=-1)

        # Вычисляем выходы всех экспертов (при 2-3 экспертах это быстрее, чем ветвление на NPU)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # [B, T, Out, E]

        # Взвешенная сумма: [B, T, Out, E] * [B, T, 1, E] -> sum по E
        out = torch.sum(expert_outputs * gates.unsqueeze(-2), dim=-1)
        return out
