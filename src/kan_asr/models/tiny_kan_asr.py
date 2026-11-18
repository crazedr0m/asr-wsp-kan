"""
Каноничная реализация TinyKAN_ASR.

Этот файл является единственным источником истины для архитектуры модели.
Все остальные файлы (WSP, training, export, tests) должны импортировать
классы из этого модуля.

Архитектура (совместима с WSP-контуром и экспортом на RV1103 NPU):
- Вход: mel-спектрограмма [B, 40, 1, T]
- Encoder: 3x DepthwiseSeparableConv1D_as_2D (40->64->128->128)
- KAN-блоки: 2x FastKANConv1x1 (128->128) с residual + BatchNorm
- Head: Conv2d 1x1 (128 -> vocab_size)
- Выход: [B, vocab_size, 1, T_out]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_layers import NPUFriendlyMoEKAN, PureChebyKANLayer, PureFastKANLayer  # noqa: F401


class FastKANConv1x1(nn.Module):
    """
    FastKAN-слой в виде 1x1 свёртки (совместим с NPU RV1103).

    Реализует базис SiLU + RBF-Гауссианы:
        y = base_conv(silu(x)) + rbf_conv(rbf(x))

    Атрибуты ``base_conv``, ``rbf_conv``, ``grid``, ``denom``, ``num_grids``,
    ``in_channels``, ``out_channels`` используются WSP-кодом
    (``wsp/project.py``) для аналитической записи весов.
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
        self.grid_min = grid_min
        self.grid_max = grid_max

        grid = torch.linspace(grid_min, grid_max, num_grids).view(1, 1, num_grids, 1, 1)
        self.register_buffer("grid", grid)
        self.denom = (grid_max - grid_min) / (num_grids - 1)

        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.rbf_conv = nn.Conv2d(in_channels * num_grids, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Вход: B x C x H x T
        B, C, H, T = x.shape
        base_out = self.base_conv(F.silu(x))
        diff = (x.unsqueeze(2) - self.grid) / self.denom
        rbf = torch.exp(-diff * diff)
        rbf_flat = rbf.reshape(B, C * self.num_grids, H, T)
        return base_out + self.rbf_conv(rbf_flat)


class DepthwiseSeparableConv1D_as_2D(nn.Module):
    """
    Глубинно-разделяемая свёртка 1D, реализованная как 2D.

    Для сжатия временной оси на NPU. Атрибуты ``dw`` и ``pw`` используются
    WSP-кодом (``wsp/initialize.py``) для записи спектральных фильтров.
    """

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            groups=in_c,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class TinyKAN_ASR(nn.Module):
    """
    Полная архитектура TinyKAN_ASR (~1.2M параметров).

    Вход:  mel-спектрограмма [B, 40, 1, T]
    Выход: CTC-логиты [B, vocab_size, 1, T_out]
    """

    def __init__(self, n_mels: int = 40, vocab_size: int = 35):
        super().__init__()
        self.n_mels = n_mels
        self.vocab_size = vocab_size

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv1D_as_2D(n_mels, 64, kernel_size=5, stride=2),
            DepthwiseSeparableConv1D_as_2D(64, 128, kernel_size=5, stride=2),
            DepthwiseSeparableConv1D_as_2D(128, 128, kernel_size=5, stride=1),
        )
        self.kan1 = FastKANConv1x1(128, 128, num_grids=6)
        self.bn_kan1 = nn.BatchNorm2d(128)
        self.kan2 = FastKANConv1x1(128, 128, num_grids=6)
        self.bn_kan2 = nn.BatchNorm2d(128)
        self.head = nn.Conv2d(128, vocab_size, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_all_features: bool = False,
    ):
        """
        x: [B, 40, 1, T] (mel-спектрограмма)

        return_features=True      -> (logits, h2)          — обратная совместимость
        return_all_features=True  -> (logits, h1, h2)      — для CKA-выравнивания
        """
        x = self.encoder(x)
        h1 = x + self.bn_kan1(self.kan1(x))
        h2 = h1 + self.bn_kan2(self.kan2(h1))
        logits = self.head(h2)

        if return_all_features:
            return logits, h1, h2
        if return_features:
            return logits, h2

        return logits
