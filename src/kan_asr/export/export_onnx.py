"""
Экспорт TinyKAN_ASR в ONNX со статическим входом для RV1103 NPU.

Порт из ``KAN_ASR_Pipeline_Bundle/train_and_export.py`` (секция экспорта).
"""

import os

import torch

from ..models.tiny_kan_asr import TinyKAN_ASR
from ..utils.paths import resolver


def export_onnx(
    checkpoint_path: str = None,
    onnx_path: str = None,
    input_shape: tuple = (1, 40, 1, 200),
    opset_version: int = 13,
    device: str = "cpu",
):
    """
    Загружает чекпоинт и экспортирует модель в ONNX.

    input_shape: статический вход [B, 40, 1, T] (2 сек аудио для RV1103).
    """
    if onnx_path is None:
        onnx_path = str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")

    model = TinyKAN_ASR()
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[EXPORT] Загружен чекпоинт: {checkpoint_path}")
    else:
        print("[EXPORT] Чекпоинт не найден, экспортируем случайно инициализированную модель")

    model.eval().to(device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dummy_input = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["mel_input"],
        output_names=["ctc_logits"],
    )
    print(f"[DONE] Модель экспортирована в {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    export_onnx()
