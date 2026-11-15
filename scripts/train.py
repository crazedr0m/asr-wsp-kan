#!/usr/bin/env python3
"""
CLI для обучения модели.

Использует: kan_asr.training.train
"""

import argparse
import os
import sys

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    )

from kan_asr.training.train import train


def main():
    parser = argparse.ArgumentParser(description="Обучение KAN-ASR")
    parser.add_argument("--data-dir", type=str, default=None, help="Путь к датасету")
    parser.add_argument("--epochs", type=int, default=5, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь для чекпоинта")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (по умолчанию auto)")

    # Weight-Space Initialization
    parser.add_argument(
        "--wsp-checkpoint",
        type=str,
        default=None,
        help="Путь к WSP-инициализации (выход scripts/wsp_init.py)",
    )

    # CKA-выравнивание
    parser.add_argument(
        "--align", action="store_true", help="Включить CKA-выравнивание с Whisper-tiny"
    )
    parser.add_argument(
        "--align-weight", type=float, default=0.3, help="Вес CKA-компоненты в лоссе"
    )
    parser.add_argument(
        "--whisper-state", type=str, default=None, help="Путь к state_dict Whisper-tiny"
    )
    parser.add_argument(
        "--anchor-weight", type=float, default=1e-3, help="Вес WSP-якоря (защита INT8)"
    )

    args = parser.parse_args()

    print(f"[TRAIN] Обучение на {args.data_dir or 'data/synth'}...")
    print(f"[TRAIN] Эпохи: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    if args.wsp_checkpoint:
        print(f"[TRAIN] WSP-инициализация: {args.wsp_checkpoint}")
    if args.align:
        print(f"[TRAIN] CKA-выравнивание: weight={args.align_weight}, teacher={args.whisper_state}")

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_path=args.checkpoint,
        device=args.device,
        wsp_checkpoint=args.wsp_checkpoint,
        use_alignment=args.align,
        align_weight=args.align_weight,
        whisper_state=args.whisper_state,
        anchor_weight=args.anchor_weight,
    )


if __name__ == "__main__":
    main()
