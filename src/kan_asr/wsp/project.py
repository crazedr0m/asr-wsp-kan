"""
Функциональная проекция MLP Whisper в FastKAN.

Модуль содержит функцию project_mlp_to_fastkan для аппроксимации нелинейности
FFN-блока Whisper базисом Гауссиан FastKAN (SiLU + RBF).

Порт из ``whisper_wsp_initializer.py`` (метод ``project_mlp_to_fastkan``).
"""

import torch
import torch.nn.functional as F


def _find_key(sd: dict, suffix: str):
    for k in sd:
        if k.endswith(suffix):
            return k
    return None


def project_mlp_to_fastkan(
    whisper_sd: dict,
    kan_layer,
    block_idx: int,
    num_samples: int = 2048,
):
    """
    Weight-Space Functional Projection.

    Берет блок FFN (fc1 + GELU + fc2) из энкодера Whisper, сжимает его размерность
    через SVD до размерности ``kan_layer`` и решает задачу наименьших квадратов
    (Ridge), чтобы базис Гауссиан FastKAN точно повторял нелинейную передаточную
    функцию Whisper.

    Ожидает ``kan_layer`` типа :class:`~kan_asr.models.tiny_kan_asr.FastKANConv1x1`
    с атрибутами ``in_channels``, ``out_channels``, ``num_grids``, ``grid``,
    ``denom``, ``base_conv``, ``rbf_conv``.
    """
    k_fc1_w = _find_key(whisper_sd, f"encoder.layers.{block_idx}.fc1.weight")
    k_fc1_b = _find_key(whisper_sd, f"encoder.layers.{block_idx}.fc1.bias")
    k_fc2_w = _find_key(whisper_sd, f"encoder.layers.{block_idx}.fc2.weight")
    k_fc2_b = _find_key(whisper_sd, f"encoder.layers.{block_idx}.fc2.bias")

    if not all([k_fc1_w, k_fc2_w]):
        raise KeyError(f"Не найдены веса FFN для блока encoder.layers.{block_idx}")

    device = kan_layer.grid.device

    W1 = whisper_sd[k_fc1_w].to(device)  # [1536, 384]
    b1 = whisper_sd[k_fc1_b].to(device) if k_fc1_b else torch.zeros(W1.shape[0], device=device)
    W2 = whisper_sd[k_fc2_w].to(device)  # [384, 1536]
    b2 = whisper_sd[k_fc2_b].to(device) if k_fc2_b else torch.zeros(W2.shape[0], device=device)

    C_in = kan_layer.in_channels
    C_out = kan_layer.out_channels
    G = kan_layer.num_grids

    # 1. Проецируем пространство признаков Whisper (384) -> (C_in) через SVD главных компонент W1
    _, _, Vh1 = torch.linalg.svd(W1, full_matrices=False)
    P_in = Vh1[:C_in, :]  # [C_in, 384] - ортогональный проектор входа

    U2, _, _ = torch.linalg.svd(W2, full_matrices=False)
    P_out = U2[:, :C_out].T  # [C_out, 384] - ортогональный проектор выхода

    # 2. Генерируем синтетический поток активаций в диапазоне [-2.5, 2.5]
    x_probe = torch.randn(num_samples, C_in, device=device) * 0.9
    x_probe = torch.clamp(x_probe, -2.5, 2.5)

    # Пропускаем через спроецированный FFN учителя (Whisper)
    with torch.no_grad():
        x_whisper = x_probe @ P_in  # [N, 384]
        h_whisper = F.gelu(F.linear(x_whisper, W1, b1))  # [N, 1536]
        y_whisper = F.linear(h_whisper, W2, b2)  # [N, 384]
        y_target = y_whisper @ P_out.T  # [N, C_out]

    # 3. Вычисляем базисные функции ученика (SiLU + RBF Гауссианы)
    grid = kan_layer.grid.view(1, 1, G).to(device)
    denom = kan_layer.denom

    base_feat = F.silu(x_probe)  # [N, C_in]
    diff = (x_probe.unsqueeze(-1) - grid) / denom
    rbf_feat = torch.exp(-diff * diff).reshape(num_samples, C_in * G)  # [N, C_in * G]

    # Объединяем матрицу признаков Phi: [N, C_in + C_in * G + 1 (bias)]
    ones = torch.ones(num_samples, 1, device=device)
    Phi = torch.cat([base_feat, rbf_feat, ones], dim=-1)

    # 4. Решаем переопределенную СЛАУ: Phi @ W_kan ≈ y_target с L2-регуляризацией (Ridge)
    # Регуляризация критична для INT8 квантования на Luckfox RV1103!
    reg_lambda = 1e-2
    A = Phi.T @ Phi + reg_lambda * torch.eye(Phi.shape[1], device=device)
    B_mat = Phi.T @ y_target
    W_solved = torch.linalg.solve(A, B_mat)  # [C_in + C_in*G + 1, C_out]

    # Раскладываем найденные веса обратно в свертки FastKANConv1x1
    W_base = W_solved[:C_in, :].T.view(C_out, C_in, 1, 1)
    W_rbf = W_solved[C_in : C_in + C_in * G, :].T.view(C_out, C_in * G, 1, 1)
    b_rbf = W_solved[-1, :]

    with torch.no_grad():
        kan_layer.base_conv.weight.copy_(W_base)
        kan_layer.rbf_conv.weight.copy_(W_rbf)
        kan_layer.rbf_conv.bias.copy_(b_rbf)

    # Считаем ошибку аппроксимации (R^2)
    y_pred = Phi @ W_solved
    mse = F.mse_loss(y_pred, y_target).item()
    r2 = 1.0 - mse / (y_target.var().item() + 1e-8)
    return mse, r2
