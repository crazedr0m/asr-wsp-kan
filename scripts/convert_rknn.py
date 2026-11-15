#!/usr/bin/env python3
"""
CLI для конвертации в RKNN.

Использует: kan_asr.export.convert_rknn
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

from kan_asr.export.convert_rknn import convert_onnx_to_rknn


def main():
    parser = argparse.ArgumentParser(description="Конвертация в RKNN")
    parser.add_argument("--onnx", type=str, default=None, help="Путь к ONNX")
    parser.add_argument("--output", type=str, default=None, help="Путь к RKNN")
    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Калибровочный файл (по умолчанию data/calib/calib_dataset.txt)",
    )

    args = parser.parse_args()

    print(f"[RKNN] Конвертация {args.onnx or 'artifacts/onnx/tiny_kan_asr_rv1103.onnx'}...")

    convert_onnx_to_rknn(
        onnx_model=args.onnx,
        rknn_model=args.output,
        calib_dataset=args.calib_file,
    )


if __name__ == "__main__":
    main()
