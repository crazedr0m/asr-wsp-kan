#!/usr/bin/env python3
"""
CLI распознавания речи WAV → текст: хост, симулятор NPU и плата Luckfox.

Использует: kan_asr.inference.recognize

Примеры:
    python scripts/recognize.py --device host --cases 5
    python scripts/recognize.py --device host --wav data/samples/test_audio_001.wav
    python scripts/recognize.py --device simulator --cases 3
    python scripts/recognize.py --device rv1103 --cases 3 --min-exact-match 0.5
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

from kan_asr.inference.recognize import (
    DEFAULT_MAX_CER,
    DEFAULT_MIN_EXACT_MATCH,
    DEFAULT_TARGET,
    DEFAULT_TIME_FRAMES,
    RecognitionCase,
    evaluate_recognition,
    load_recognition_cases,
    recognize_on_device,
    recognize_rknn_simulator,
    recognize_torch,
    save_recognition_report,
)
from kan_asr.utils.paths import resolver


def _build_recognizer(args):
    """Возвращает функцию ``wav_path -> str`` для выбранного устройства."""
    if args.device == "host":
        checkpoint = args.checkpoint or str(resolver.checkpoints / "best.pt")
        return lambda wav: recognize_torch(
            wav, checkpoint_path=checkpoint, device=args.torch_device
        )
    if args.device == "simulator":
        return lambda wav: recognize_rknn_simulator(
            wav,
            onnx_model=args.onnx,
            rknn_model=args.rknn,
            calib_dataset=args.calib_file,
            time_frames=args.time_frames,
            data_format=args.data_format,
            verbose=not args.quiet,
        )
    if args.device == "rv1103":
        return lambda wav: (
            recognize_on_device(
                wav,
                rknn_model=args.rknn,
                target=args.target,
                device_id=args.device_id,
                time_frames=args.time_frames,
                data_format=args.data_format,
                input_dtype=args.input_dtype,
                auto_start_server=not args.no_auto_start_server,
                verbose=not args.quiet,
            ).text
        )
    raise ValueError(f"Неизвестное устройство: {args.device}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Распознавание речи WAV → текст на хосте (PyTorch), "
            "в симуляторе NPU (RKNN INT8) или на плате Luckfox (rv1103)"
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="host",
        choices=["host", "simulator", "rv1103"],
        help="Уровень инференса (по умолчанию host)",
    )
    parser.add_argument("--wav", type=str, default=None, help="Один WAV-файл")
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Число кейсов из metadata.csv (если не задан --wav)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Каталог с metadata.csv (по умолчанию data/synth)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Сколько первых кейсов пропустить (held-out хвост)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Чекпоинт для хоста (по умолчанию artifacts/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="Устройство PyTorch для хоста (cpu/cuda)",
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
        "--time-frames",
        type=int,
        default=DEFAULT_TIME_FRAMES,
        help=f"Длина окна Mel (по умолчанию {DEFAULT_TIME_FRAMES})",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default=None,
        choices=["nchw", "nhwc"],
        help="Формат входа rknn.inference (по умолчанию nchw для симулятора, nhwc для платы)",
    )
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="float32",
        choices=["float32", "uint8"],
        help="Тип входа на плате (по умолчанию float32)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Целевая платформа (по умолчанию {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Серийный номер платы (по умолчанию — единственное устройство)",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=DEFAULT_MAX_CER,
        help=f"Максимально допустимый CER (по умолчанию {DEFAULT_MAX_CER})",
    )
    parser.add_argument(
        "--min-exact-match",
        type=float,
        default=DEFAULT_MIN_EXACT_MATCH,
        help=(
            "Минимальная доля точных совпадений "
            f"(по умолчанию {DEFAULT_MIN_EXACT_MATCH} — только отчёт)"
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Путь к JSON-отчёту (по умолчанию artifacts/reports/recognition_report.json)",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Не сохранять JSON-отчёт",
    )
    parser.add_argument(
        "--no-auto-start-server",
        action="store_true",
        help="Не поднимать rknn_server на плате автоматически",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать прогресс",
    )

    args = parser.parse_args()

    # Формат данных по умолчанию зависит от уровня
    if args.data_format is None:
        args.data_format = "nhwc" if args.device == "rv1103" else "nchw"

    # Набор кейсов: один WAV или N из metadata.csv
    if args.wav:
        cases = [RecognitionCase(wav_path=args.wav, text="")]
    else:
        cases = load_recognition_cases(n=args.cases, data_dir=args.data_dir, offset=args.offset)
        if not cases:
            print(
                "[ERROR] Не найдено кейсов распознавания. Проверьте metadata.csv "
                "(data/synth) или укажите --wav.",
                file=sys.stderr,
            )
            return 2

    try:
        recognize_fn = _build_recognizer(args)
        report = evaluate_recognition(
            cases,
            recognize_fn,
            max_cer=args.max_cer,
            min_exact_match=args.min_exact_match,
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

    print(report.summary())

    if not args.no_save_report:
        saved = save_recognition_report(report, args.report)
        if not args.quiet:
            print(f"--> Отчёт сохранён: {saved}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
