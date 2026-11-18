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


def _evaluate_after_conversion(
    onnx_model: str,
    rknn_model: str,
    calib_dataset: str,
    n_cases: int = 3,
    report_path: str | None = None,
) -> None:
    """
    Проверяет распознавание INT8-модели в симуляторе NPU после конвертации.

    Считает exact-match/CER и сохраняет отчёт. Никогда не роняет конвертацию:
    сбой эмулятора или отсутствие данных логируются и подавляются.

    Args:
        onnx_model: путь к ONNX (симулятор собирает ``.rknn`` из него заново).
        rknn_model: путь к только что собранному ``.rknn``.
        calib_dataset: файл калибровочных ``.npy``.
        n_cases: сколько кейсов оценить.
        report_path: куда сохранить JSON (по умолчанию
            ``artifacts/reports/convert_recognition.json``).
    """
    try:
        from ..inference.recognize import (
            evaluate_recognition,
            load_recognition_cases,
            recognize_rknn_simulator,
            save_recognition_report,
        )
    except Exception as exc:
        print(f"[WARN] Eval распознавания недоступен: {exc}")
        return

    cases = load_recognition_cases(n=n_cases)
    if not cases:
        print("[WARN] Eval распознавания пропущен: нет кейсов (make data)")
        return

    try:
        report = evaluate_recognition(
            cases,
            lambda wav: recognize_rknn_simulator(
                wav,
                onnx_model=onnx_model,
                rknn_model=rknn_model,
                calib_dataset=calib_dataset,
            ),
            verbose=False,
        )
    except Exception as exc:
        print(f"[WARN] Eval распознавания INT8 не удался: {exc}")
        return

    if report_path is None:
        report_path = str(resolver.artifacts / "reports" / "convert_recognition.json")
    save_recognition_report(report, report_path)
    print(report.summary())
    print(f"[DONE] Отчёт распознавания INT8 сохранён: {report_path}")


def convert_onnx_to_rknn(
    onnx_model: str | None = None,
    rknn_model: str | None = None,
    calib_dataset: str | None = None,
    eval_after_convert: bool = True,
    eval_cases: int = 3,
):
    """
    Конвертирует ONNX-модель в RKNN с INT8-квантованием под rv1103.

    Требует установленного ``rknn-toolkit2`` (импортируется лениво).

    После сборки ``.rknn`` проверяет распознавание INT8 в симуляторе NPU
    (exact-match/CER) и сохраняет отчёт ``convert_recognition.json``.
    """
    try:
        from rknn.api import RKNN
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Не найден пакет 'rknn-toolkit2'. Установите его отдельно:\n"
            '    pip install -e ".[rknn]"\n'
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

    if eval_after_convert:
        _evaluate_after_conversion(
            onnx_model=onnx_model,
            rknn_model=rknn_model,
            calib_dataset=calib_dataset,
            n_cases=eval_cases,
        )

    return rknn_model


if __name__ == "__main__":
    convert_onnx_to_rknn()
