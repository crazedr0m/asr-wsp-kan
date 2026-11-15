"""
Фоновое дообучение модели на данных, собранных через Telegram-бота.

Стратегия: 70% новых подтверждённых сэмплов + 30% базового синтетического
датасета (защита от катастрофического забывания).

Дообучение выполняется через ``WeightSpaceHyperAdapter``: базовая модель
замораживается, обучаются только низкоранговые поправки Delta-W к слоям
``kan1``/``kan2``. Это дополнительно снижает риск забывания на малых данных.
"""

import os
import random
import shutil

from ..utils.logging import get_logger
from ..utils.paths import resolver

logger = get_logger("bot.finetune_worker")

NEW_DATA_RATIO = 0.7


def _read_meta(root: str):
    """Читает ``metadata.csv`` (формат ``filename|text``) -> список (rel, text)."""
    meta = os.path.join(root, "metadata.csv")
    if not os.path.exists(meta):
        return []
    rows = []
    with open(meta, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("filename"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def build_mixed_dataset(
    real_dir: str = None,
    synth_dir: str = None,
    out_dir: str = None,
    new_ratio: float = NEW_DATA_RATIO,
) -> str | None:
    """
    Собирает смешанный датасет (новые + базовые) в ``out_dir``.

    Возвращает путь к каталогу с ``metadata.csv`` и ``wavs/`` или ``None``,
    если данных для дообучения нет (тогда ``metadata.csv`` не создаётся).
    """
    if real_dir is None:
        real_dir = str(resolver.real)
    if synth_dir is None:
        synth_dir = str(resolver.synth)
    if out_dir is None:
        out_dir = str(resolver.data / "finetune")

    out_wavs = os.path.join(out_dir, "wavs")
    os.makedirs(out_wavs, exist_ok=True)

    new_rows = _read_meta(real_dir)
    base_rows = _read_meta(synth_dir)

    # 70% новых + 30% базовых (по количеству новых)
    n_new = len(new_rows)
    n_base = int(n_new * (1.0 - new_ratio) / max(new_ratio, 1e-6)) if n_new else 0
    sampled_base = random.sample(base_rows, min(n_base, len(base_rows))) if base_rows else []

    # Явно помечаем источник каждого сэмпла (устойчиво к совпадающим парам)
    combined = [(rel, text, real_dir) for rel, text in new_rows]
    combined += [(rel, text, synth_dir) for rel, text in sampled_base]

    if not combined:
        logger.warning("Нет данных для дообучения (real=%d, base=%d)", len(new_rows), len(base_rows))
        return None

    with open(os.path.join(out_dir, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("filename|text\n")
        for rel, text, src_root in combined:
            src = os.path.join(src_root, rel)
            dst_name = os.path.basename(rel)
            dst = os.path.join(out_wavs, dst_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
            f.write(f"wavs/{dst_name}|{text}\n")

    logger.info("Смешанный датасет: %d новых + %d базовых -> %s", len(new_rows), len(sampled_base), out_dir)
    return out_dir


def run_finetune(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
    use_hyper_adapter: bool = True,
    base_checkpoint: str = None,
) -> None:
    """
    Запускает дообучение на смешанном датасете (ленивый импорт train).

    При ``use_hyper_adapter=True`` обучаются только низкоранговые поправки
    Delta-W (``WeightSpaceHyperAdapter``), базовая модель заморожена.
    """
    from ..training.train import train

    data_dir = build_mixed_dataset()
    if data_dir is None or not os.path.exists(os.path.join(data_dir, "metadata.csv")):
        logger.warning("Дообучение пропущено: нет подготовленного датасета.")
        return
    if base_checkpoint is None:
        base_checkpoint = str(resolver.checkpoints / "best.pt")

    logger.info(
        "Старт дообучения: epochs=%d, batch=%d, lr=%s, hyper_adapter=%s",
        epochs,
        batch_size,
        lr,
        use_hyper_adapter,
    )
    train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_hyper_adapter=use_hyper_adapter,
        base_checkpoint=base_checkpoint,
    )
    logger.info("Дообучение завершено.")
