#!/usr/bin/env python3
"""
CLI для верификации RKNN-модели: симулятор NPU на ПК и замер на плате.

Использует: kan_asr.export.verify_rknn
"""

import argparse
import os
import sys

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    )

from kan_asr.export.verify_rknn import (
    DEFAULT_AUDIO_SECONDS,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_MAX_SIZE_MB,
    DEFAULT_MIN_COSINE,
    DEFAULT_WARMUP,
    _adb_devices,
    verify_rknn_simulator,
)


def _list_devices() -> int:
    """Печатает список устройств ADB и возвращает код выхода."""
    try:
        devices = _adb_devices()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if not devices:
        print("Устройства ADB не найдены. Подключите плату по USB.")
        return 1
    print("Устройства ADB:")
    for serial in devices:
        print(f"  - {serial}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Проверка RKNN-модели в симуляторе NPU (target=None) "
            "и замер на реальной плате (--target rv1103)"
        )
    )
    parser.add_argument("--onnx", type=str, default=None, help="Путь к ONNX")
    parser.add_argument("--rknn", type=str, default=None, help="Путь к .rknn")
    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Калибровочный файл (по умолчанию data/calib/calib_dataset.txt)",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=",".join(str(x) for x in DEFAULT_INPUT_SHAPE),
        help="Форма входа, например 1,40,1,200",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=DEFAULT_MIN_COSINE,
        help=f"Минимальное косинусное сходство (по умолчанию {DEFAULT_MIN_COSINE})",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_SIZE_MB,
        help=f"Лимит размера .rknn в МБ (по умолчанию {DEFAULT_MAX_SIZE_MB})",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=DEFAULT_MAX_MEMORY_MB,
        help=f"Лимит пиковой памяти в МБ (по умолчанию {DEFAULT_MAX_MEMORY_MB})",
    )
    parser.add_argument(
        "--accuracy-analysis",
        action="store_true",
        help="Запустить rknn.accuracy_analysis (отчёт по слоям)",
    )
    parser.add_argument(
        "--accuracy-output-dir",
        type=str,
        default=None,
        help="Каталог отчётов accuracy analysis (по умолчанию artifacts/rknn_accuracy)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересобрать .rknn из ONNX, даже если файл уже существует",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="Формат входных данных для rknn.inference в симуляторе",
    )

    # Замер на реальной плате
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Целевая платформа для замера на железе (например rv1103). "
            "Без флага выполняется только симулятор."
        ),
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Серийный номер платы (по умолчанию — единственное устройство)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Число прогревочных прогонов на плате (по умолчанию {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Число замеряемых прогонов на плате (по умолчанию {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=DEFAULT_AUDIO_SECONDS,
        help=f"Длительность аудиоокна для RTF (по умолчанию {DEFAULT_AUDIO_SECONDS})",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="Порог средней latency на плате (мс)",
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=None,
        help="Минимальный FPS на плате",
    )
    parser.add_argument(
        "--perf-report",
        type=str,
        default=None,
        help="Путь к JSON-отчёту бенчмарка (по умолчанию artifacts/rknn/perf_report.json)",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Не сохранять JSON-отчёт бенчмарка",
    )
    parser.add_argument(
        "--no-auto-start-server",
        action="store_true",
        help="Не поднимать rknn_server на плате автоматически",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Показать список устройств ADB и выйти",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать прогресс",
    )

    args = parser.parse_args()

    if args.list_devices:
        return _list_devices()

    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    try:
        result = verify_rknn_simulator(
            onnx_model=args.onnx,
            rknn_model=args.rknn,
            calib_dataset=args.calib_file,
            input_shape=input_shape,
            min_cosine=args.min_cosine,
            max_size_mb=args.max_size_mb,
            max_memory_mb=args.max_memory_mb,
            run_accuracy_analysis=args.accuracy_analysis,
            accuracy_output_dir=args.accuracy_output_dir,
            rebuild=args.rebuild,
            data_format=args.data_format,
            target=args.target,
            device_id=args.device_id,
            warmup=args.warmup,
            iterations=args.iterations,
            audio_seconds=args.audio_seconds,
            max_latency_ms=args.max_latency_ms,
            min_fps=args.min_fps,
            perf_report_path=args.perf_report,
            save_report=not args.no_save_report,
            auto_start_server=not args.no_auto_start_server,
            verbose=not args.quiet,
        )
    except ImportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    # Сводка уже напечатана внутри verify_rknn_simulator (verbose=True).
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
