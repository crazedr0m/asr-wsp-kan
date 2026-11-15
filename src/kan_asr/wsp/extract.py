"""
Извлечение спектральных фильтров из Whisper.

Модуль содержит метод extract_conv_subspace для SVD-проекции conv-фильтров.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperWeightSpaceProgrammer:
    """
    Выполняет перенос знаний из пространства весов Whisper-tiny/base в TinyKAN_ASR:
    1. Спектральный анализ входных сверток (conv1, conv2) через SVD и частотную интерполяцию.
    2. Аналитическую аппроксимацию GELU-MLP блоков Whisper в базис Гауссиан FastKAN 
       через решение задачи наименьших квадратов (torch.linalg.lstsq).
    3. Построение компактного дескриптора пространства весов (Weight-Space Signature)
       для работы с HyperNetwork / Weight-Space LoRA.
    """
    
    def __init__(self, whisper_state_dict: dict, device: str = "cuda"):
        self.sd = {k: v.to(device).float() for k, v in whisper_state_dict.items()}
        self.device = device

    @staticmethod
    def _find_key(sd: dict, suffix: str):
        for k in sd.keys():
            if k.endswith(suffix):
                return k
        return None

    def extract_conv_subspace(self, target_in_mels: int = 40, target_out_c: int = 64, kernel_size: int = 5):
        """
        Извлекает главные частотно-временные фильтры из encoder.conv1 Whisper [384, 80, 3],
        интерполирует частотную ось 80 -> 40, временную 3 -> 5 и проецирует через SVD.
        """
        k_w = self._find_key(self.sd, "encoder.conv1.weight")
        if k_w is None:
            raise KeyError("Не найден ключ encoder.conv1.weight в state_dict Whisper")

        w_whisper = self.sd[k_w]  # [384, 80, 3]
        out_w, in_mels_w, kw_w = w_whisper.shape

        # 1. Интерполяция частотной оси (80 -> 40) и временной оси (3 -> 5)
        w_interp = F.interpolate(
            w_whisper.unsqueeze(1),  # [384, 1, 80, 3]
            size=(target_in_mels, kernel_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # [384, 40, 5]

        # 2. SVD-разложение по выходным каналам для выделения 64 главных детекторов формант
        flat = w_interp.reshape(out_w, -1)  # [384, 200]
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
        
        # Берем топ-target_out_c сингулярных компонент, взвешенных корнем из сингулярных чисел
        top_filters = (Vh[:target_out_c] * torch.sqrt(S[:target_out_c]).unsqueeze(-1))
        top_filters = top_filters.reshape(target_out_c, target_in_mels, kernel_size)  # [64, 40, 5]

        # Разделяем на Depthwise [40, 1, 1, 5] и Pointwise [64, 40, 1, 1] через среднее по каналам
        dw_weight = top_filters.mean(dim=0, keepdim=True).transpose(0, 1).unsqueeze(2)  # [40, 1, 1, 5]
        pw_weight = top_filters.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)               # [64, 40, 1, 1]

        # Нормализация дисперсии под INT8 квантование (защита от выбросов на RV1103)
        dw_weight = dw_weight / (dw_weight.std() + 1e-6) * 0.25
        pw_weight = pw_weight / (pw_weight.std() + 1e-6) * 0.25
        return dw_weight, pw_weight
