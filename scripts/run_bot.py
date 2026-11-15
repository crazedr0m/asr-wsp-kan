#!/usr/bin/env python3
"""
CLI для запуска Telegram-бота (Human-in-the-Loop / Active Learning).

Использует: kan_asr.bot.app
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


def main():
    parser = argparse.ArgumentParser(description="Запуск Telegram-бота")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Telegram bot token (по умолчанию env KAN_ASR_BOT_TOKEN)",
    )
    parser.add_argument(
        "--finetune-threshold",
        type=int,
        default=20,
        help="Число подтверждённых сэмплов для запуска дообучения (по умолчанию 20)",
    )

    args = parser.parse_args()

    from kan_asr.bot.app import main as run_bot

    token = args.token or os.environ.get("KAN_ASR_BOT_TOKEN", "")
    if token:
        print(f"[BOT] Запуск бота с token={token[:10]}...")
    run_bot(token=args.token, finetune_threshold=args.finetune_threshold)


if __name__ == "__main__":
    main()
