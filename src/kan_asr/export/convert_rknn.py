"""
Конвертация ONNX -> RKNN INT8 для NPU RV1103 (Luckfox Pico Plus).

Порт из ``KAN_ASR_Pipeline_Bundle/convert_to_rknn.py``.
"""

import os

from ..utils.paths import resolver


def build_rknn_model(rknn, onnx_model: str, rknn_model: str, calib_dataset: str):
    """
    Настраивает, квантует и экспортирует RKNN-модель в переданный инстанс ``RKNN``.

    Вынесено в отдельную функцию, чтобы её переиспользовали и конвертация
    (``convert_onnx_to_rknn``), и верификация в симуляторе NPU
    (``verify_rknn.verify_rknn_simulator``).
    """
    # Целевая платформа строго rv1103 (Luckfox Pico Plus)
    rknn.config(
        target_platform="rv1103",
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
    )

    print("--> Загрузка ONNX модели...")
    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        raise RuntimeError("Ошибка загрузки ONNX")

    print("--> Квантование в INT8 и компиляция под NPU RV1103...")
    ret = rknn.build(do_quantization=True, dataset=calib_dataset)
    if ret != 0:
        raise RuntimeError("Ошибка сборки RKNN")

    print("--> Экспорт .rknn файла...")
    os.makedirs(os.path.dirname(rknn_model) or ".", exist_ok=True)
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        raise RuntimeError("Ошибка экспорта RKNN")


def convert_onnx_to_rknn(
    onnx_model: str = None,
    rknn_model: str = None,
    calib_dataset: str = None,
):
    """
    Конвертирует ONNX-модель в RKNN с INT8-квантованием под rv1103.

    Требует установленного ``rknn-toolkit2`` (импортируется лениво).
    """
    try:
        from rknn.api import RKNN
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Не найден пакет 'rknn-toolkit2'. Установите его отдельно:\n"
            "    pip install -e \".[rknn]\"\n"
            "или\n"
            "    pip install -r requirements-optional.txt"
        ) from exc

    if onnx_model is None:
        onnx_model = str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")
    if rknn_model is None:
        rknn_model = str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")
    if calib_dataset is None:
        calib_dataset = str(resolver.calib / "calib_dataset.txt")

    rknn = RKNN(verbose=True)
    build_rknn_model(rknn, onnx_model, rknn_model, calib_dataset)
    rknn.release()
    print(f"[DONE] Готово: {rknn_model}")
    return rknn_model


if __name__ == "__main__":
    convert_onnx_to_rknn()
