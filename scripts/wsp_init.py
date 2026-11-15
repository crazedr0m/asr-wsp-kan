#!/usr/bin/env python3
"""
CLI для инициализации WSP (Weight-Space Programming).

Использует: kan_asr.wsp
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

import torch

from kan_asr.models.tiny_kan_asr import TinyKAN_ASR
from kan_asr.utils.paths import resolver
from kan_asr.wsp.initialize import initialize_student


def main():
    parser = argparse.ArgumentParser(description="WSP инициализация")
    parser.add_argument(
        "--whisper-state", type=str, required=True, help="Путь к state_dict Whisper"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(resolver.checkpoints / "tiny_kan_wsp_initialized.pt"),
        help="Путь к сохранённой модели",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device для работы")

    args = parser.parse_args()

    print("[WSP] Загрузка весов Whisper...")
    whisper_sd = torch.load(args.whisper_state, map_location=args.device)

    print("[WSP] Создание ученика...")
    student = TinyKAN_ASR(n_mels=40, vocab_size=35)

    print("[WSP] Инициализация из весов Whisper...")
    student = initialize_student(whisper_sd, student, device=args.device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(student.state_dict(), args.output)
    print(f"[WSP] Инициализация завершена. Модель сохранена: {args.output}")


if __name__ == "__main__":
    main()
