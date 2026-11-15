#!/usr/bin/env python3
"""
CLI для генерации синтетического датасета.

Использует: kan_asr.data.generate
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

from kan_asr.data.generate import generate_synthetic_dataset
from kan_asr.utils.paths import resolver


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетического датасета")
    parser.add_argument(
        "--output-dir", type=str, default=str(resolver.synth), help="Путь к выходной директории"
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Количество сэмплов")
    parser.add_argument(
        "--piper-model",
        type=str,
        default=str(resolver.piper / "ru_RU-irina-medium.onnx"),
        help="Путь к Piper TTS модели",
    )

    args = parser.parse_args()

    print(f"[DATASET] Генерация {args.num_samples} сэмплов в {args.output_dir}...")
    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        model_onnx=args.piper_model,
    )


if __name__ == "__main__":
    main()
