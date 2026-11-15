"""
CKA-выравнивание представлений между учеником (TinyKAN_ASR) и учителем (Whisper-tiny).

Перенесено из Whisper_CKA_Alignment/cka_feature_alignment.py (легаси-каталог).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_cka_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Linear Centered Kernel Alignment (CKA) Loss = 1 - CKA(X, Y).
    Устойчив к разнице размерностей каналов (128 у TinyKAN vs 384 у Whisper).
    Ожидает тензоры формы [N_samples, Channels].
    """
    X = student_feat.float()
    Y = teacher_feat.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Используем формулировку через матрицы Грама по признакам [C_s, C_t]
    # для экономии памяти при длинных аудиопоследовательностях
    xt_y = X.t() @ Y
    xt_x = X.t() @ X
    yt_y = Y.t() @ Y

    hsic_xy = torch.sum(xt_y * xt_y)
    hsic_xx = torch.sum(xt_x * xt_x)
    hsic_yy = torch.sum(yt_y * yt_y)

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)
    return 1.0 - torch.clamp(cka, 0.0, 1.0)


class WhisperKANAlignmentLoss(nn.Module):
    """
    Комбинированный лосс выравнивания:
    1. Синхронизация временной оси (шаг 20 мс у Whisper -> 40 мс у TinyKAN через avg_pool1d).
    2. Linear CKA для сохранения геометрии фонемных кластеров.
    3. Cosine Projection Loss для пофреймового выравнивания.
    4. WSP Anchor Penalty для удержания весов в диапазоне, безопасном для INT8 NPU (RV1103).
    """

    def __init__(self, student_dim: int = 128, teacher_dim: int = 384, anchor_weight: float = 1e-3):
        super().__init__()
        self.proj_mid = nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=False)
        self.proj_top = nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=False)
        self.anchor_weight = anchor_weight
        self.initial_weights = {}

    def register_wsp_anchors(self, student_model: nn.Module):
        """Сохраняет стартовые веса FastKAN после Weight-Space Initialization."""
        self.initial_weights = {
            name: param.detach().clone()
            for name, param in student_model.named_parameters()
            if "kan" in name and param.requires_grad
        }

    def _align_time(self, teacher_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        # teacher_feat: [B, C_t, T_teacher] (шаг 20 мс)
        # Усредняем соседние фреймы (20 мс * 2 = 40 мс) и подгоняем длину под ученика
        pooled = F.avg_pool1d(teacher_feat, kernel_size=2, stride=2, ceil_mode=True)
        if pooled.size(-1) != target_len:
            pooled = F.interpolate(pooled, size=target_len, mode="linear", align_corners=False)
        return pooled

    def forward(
        self,
        student_mid: torch.Tensor,
        student_top: torch.Tensor,
        teacher_mid: torch.Tensor,
        teacher_top: torch.Tensor,
        student_model: nn.Module = None,
    ) -> dict:
        # student_*: [B, 128, T_s] (после squeeze(2))
        T_s = student_mid.size(-1)
        t_mid_aligned = self._align_time(teacher_mid, T_s)
        t_top_aligned = self._align_time(teacher_top, T_s)

        # 1. CKA Loss (выпрямляем [B, C, T] -> [B*T, C])
        s_mid_flat = student_mid.permute(0, 2, 1).reshape(-1, student_mid.size(1))
        t_mid_flat = t_mid_aligned.permute(0, 2, 1).reshape(-1, t_mid_aligned.size(1))
        cka_mid = linear_cka_loss(s_mid_flat, t_mid_flat)

        s_top_flat = student_top.permute(0, 2, 1).reshape(-1, student_top.size(1))
        t_top_flat = t_top_aligned.permute(0, 2, 1).reshape(-1, t_top_aligned.size(1))
        cka_top = linear_cka_loss(s_top_flat, t_top_flat)

        # 2. Cosine Similarity через обучаемый 1x1 проектор (отбрасывается после обучения!)
        pred_mid = self.proj_mid(student_mid)
        cos_mid = 1.0 - F.cosine_similarity(pred_mid, t_mid_aligned, dim=1).mean()

        pred_top = self.proj_top(student_top)
        cos_top = 1.0 - F.cosine_similarity(pred_top, t_top_aligned, dim=1).mean()

        # 3. Якорь к WSP-весам (защита от выбросов при квантовании INT8 на RV1103)
        anchor_loss = torch.tensor(0.0, device=student_mid.device)
        if student_model is not None and self.initial_weights:
            for name, param in student_model.named_parameters():
                if name in self.initial_weights:
                    anchor_loss = anchor_loss + F.mse_loss(param, self.initial_weights[name].to(param.device))

        total_align = 0.5 * (cka_mid + cka_top) + 0.5 * (cos_mid + cos_top) + self.anchor_weight * anchor_loss
        return {
            "total_align": total_align,
            "cka": 0.5 * (cka_mid + cka_top).detach(),
            "cosine": 0.5 * (cos_mid + cos_top).detach(),
            "anchor": anchor_loss.detach(),
        }
