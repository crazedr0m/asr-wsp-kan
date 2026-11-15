"""
Weight-Space Hyper-Adapter для дообучения через Telegram-бота.

Модуль содержит класс WeightSpaceHyperAdapter для генерации низкоранговых поправок.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightSpaceHyperAdapter(nn.Module):
    """
    Компактная гиперсеть для дообучения на малом количестве данных из Telegram-бота.
    Вместо прямого изменения 1.2M весов TinyKAN_ASR (что вызывает катастрофическое забывание),
    гиперсеть генерирует низкоранговые поправки (Delta-W) в пространстве весов KAN.
    """
    
    def __init__(self, target_model: "TinyKAN_ASR", latent_dim: int = 32, rank: int = 8):  # type: ignore
        super().__init__()
        self.rank = rank
        self.latent_code = nn.Parameter(torch.randn(1, latent_dim) * 0.02)

        # Генератор поправок для rbf_conv двух слоев KAN
        c_out = target_model.kan1.out_channels if hasattr(target_model, 'kan1') else 64
        c_in_grid = target_model.kan1.in_channels * target_model.kan1.num_grids if hasattr(target_model, 'kan1') else 64 * 6

        self.hyper_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * (c_out * rank + rank * c_in_grid))
        )
        self.c_out = c_out
        self.c_in_grid = c_in_grid

    def get_delta_weights(self):
        raw = self.hyper_mlp(self.latent_code).squeeze(0)
        split_size = self.c_out * self.rank + self.rank * self.c_in_grid
        part1, part2 = raw[:split_size], raw[split_size:]

        def build_delta(p):
            A = p[:self.c_out * self.rank].view(self.c_out, self.rank)
            B = p[self.c_out * self.rank:].view(self.rank, self.c_in_grid)
            return (A @ B).view(self.c_out, self.c_in_grid, 1, 1) * 0.05

        return build_delta(part1), build_delta(part2)
