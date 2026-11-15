"""
Инициализация ученика из весов Whisper.

Модуль содержит метод initialize_student для полной инициализации TinyKAN_ASR.
"""

import torch

from .extract import WhisperWeightSpaceProgrammer
from .project import project_mlp_to_fastkan


def extract_conv_subspace(
    whisper_sd: dict,
    device: str = "cuda",
    target_in_mels: int = 40,
    target_out_c: int = 64,
    kernel_size: int = 5,
):
    """
    Извлекает главные частотно-временные фильтры из encoder.conv1 Whisper.

    Тонкая обёртка над :class:`WhisperWeightSpaceProgrammer` (единый источник
    истины в ``wsp/extract.py``).
    """
    programmer = WhisperWeightSpaceProgrammer(whisper_sd, device=device)
    return programmer.extract_conv_subspace(
        target_in_mels=target_in_mels,
        target_out_c=target_out_c,
        kernel_size=kernel_size,
    )


def initialize_student(
    whisper_sd: dict,
    student: "TinyKAN_ASR",  # type: ignore - будет импортировано из models
    device: str = "cuda",
):
    """Полная инициализация TinyKAN_ASR из весов Whisper."""
    student = student.to(device)
    print("[WSP] 1. Извлечение спектральных фильтров из Whisper conv1...")

    # Извлекаем conv subspace
    dw_w, pw_w = extract_conv_subspace(whisper_sd, device)

    with torch.no_grad():
        if hasattr(student.encoder[0], "dw") and hasattr(student.encoder[0], "pw"):
            student.encoder[0].dw.weight.copy_(dw_w)
            student.encoder[0].pw.weight.copy_(pw_w)
        else:
            print("[WSP] Предупреждение: encoder[0] не имеет dw/pw атрибутов")

    print("[WSP] 2. Проекция Whisper Encoder Block 0 -> FastKAN Layer 1...")
    mse1, r2_1 = project_mlp_to_fastkan(whisper_sd, student.kan1, block_idx=0)
    print(f"      -> KAN1 инициализирован (MSE={mse1:.4f}, R2={r2_1:.4f})")

    # Берем последний доступный блок энкодера (для whisper-tiny это блок 3)
    has_block3 = any(k.endswith("encoder.layers.3.fc1.weight") for k in whisper_sd)
    last_block = 3 if has_block3 else 1
    print(f"[WSP] 3. Проекция Whisper Encoder Block {last_block} -> FastKAN Layer 2...")
    mse2, r2_2 = project_mlp_to_fastkan(whisper_sd, student.kan2, block_idx=last_block)
    print(f"      -> KAN2 инициализирован (MSE={mse2:.4f}, R2={r2_2:.4f})")

    return student
