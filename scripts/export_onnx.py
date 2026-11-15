#!/usr/bin/env python3
"""
CLI для экспорта в ONNX.

Использует: kan_asr.export.export_onnx
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

from kan_asr.export.export_onnx import export_onnx


def main():
    parser = argparse.ArgumentParser(description="Экспорт в ONNX")
    parser.add_argument("--model", type=str, default=None, help="Путь к чекпоинту (.pt)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь к ONNX (по умолчанию artifacts/onnx/tiny_kan_asr_rv1103.onnx)",
    )
    parser.add_argument("--input-shape", type=str, default="1,40,1,200", help="Input shape")

    args = parser.parse_args()

    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    print(f"[EXPORT] Экспорт модели (input_shape={input_shape})...")

    export_onnx(
        checkpoint_path=args.model,
        onnx_path=args.output,
        input_shape=input_shape,
    )


if __name__ == "__main__":
    main()
