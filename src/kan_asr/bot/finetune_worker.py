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
    with open(meta, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("filename"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def build_mixed_dataset(
    real_dir: str | None = None,
    synth_dir: str | None = None,
    out_dir: str | None = None,
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
        logger.warning(
            "Нет данных для дообучения (real=%d, base=%d)", len(new_rows), len(base_rows)
        )
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

    logger.info(
        "Смешанный датасет: %d новых + %d базовых -> %s", len(new_rows), len(sampled_base), out_dir
    )
    return out_dir


def _measure_exact_match(
    checkpoint_path: str, data_dir: str, n_cases: int = 5
) -> tuple[float, float] | None:
    """
    Считает ``(exact_match_rate, cer)`` на кейсах из ``data_dir``.

    Возвращает ``None``, если кейсов нет или инференс не удался. Никогда не
    роняет дообучение.
    """
    try:
        from ..inference.recognize import (
            evaluate_recognition,
            load_recognition_cases,
            recognize_torch,
        )
    except Exception as exc:
        logger.warning("Eval распознавания недоступен: %s", exc)
        return None

    cases = load_recognition_cases(n=n_cases, data_dir=data_dir)
    if not cases:
        logger.warning("Eval распознавания пропущен: нет кейсов в %s", data_dir)
        return None

    try:
        report = evaluate_recognition(
            cases,
            lambda wav: recognize_torch(wav, checkpoint_path),
            verbose=False,
        )
    except Exception as exc:
        logger.warning("Eval распознавания не удался: %s", exc)
        return None
    return report.exact_match_rate, report.cer


def run_finetune(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
    use_hyper_adapter: bool = True,
    base_checkpoint: str | None = None,
    eval_after_finetune: bool = True,
    eval_cases: int = 5,
) -> None:
    """
    Запускает дообучение на смешанном датасете (ленивый импорт train).

    При ``use_hyper_adapter=True`` обучаются только низкоранговые поправки
    Delta-W (``WeightSpaceHyperAdapter``), базовая модель заморожена.
    После дообучения считает дельту точных совпадений относительно базового
    чекпоинта и сохраняет отчёт ``artifacts/reports/finetune_recognition.json``.
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
    checkpoint_path = str(resolver.checkpoints / "best.pt")

    before = None
    if eval_after_finetune and os.path.exists(base_checkpoint):
        before = _measure_exact_match(base_checkpoint, data_dir, n_cases=eval_cases)

    train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_hyper_adapter=use_hyper_adapter,
        base_checkpoint=base_checkpoint,
        eval_after_train=False,
    )

    if eval_after_finetune and os.path.exists(checkpoint_path):
        _report_finetune_delta(
            base_checkpoint=base_checkpoint,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            n_cases=eval_cases,
            before=before,
        )

    logger.info("Дообучение завершено.")


def _report_finetune_delta(
    base_checkpoint: str,
    checkpoint_path: str,
    data_dir: str,
    n_cases: int = 5,
    before: tuple[float, float] | None = None,
) -> None:
    """
    Считает дельту exact-match/CER после дообучения и сохраняет JSON-отчёт.

    Args:
        base_checkpoint: чекпоинт до дообучения (для сравнения).
        checkpoint_path: чекпоинт после дообучения.
        data_dir: каталог с кейсами.
        n_cases: сколько кейсов оценить.
        before: заранее измеренные ``(exact_match_rate, cer)`` базы (если есть).
    """
    after = _measure_exact_match(checkpoint_path, data_dir, n_cases=n_cases)
    if after is None:
        return

    after_rate, after_cer = after
    delta_rate = None
    delta_cer = None
    if before is not None:
        delta_rate = after_rate - before[0]
        delta_cer = after_cer - before[1]
        logger.info(
            "Дообучение: exact-match %.3f -> %.3f (delta %+.3f) | CER %.4f -> %.4f (delta %+.4f)",
            before[0],
            after_rate,
            delta_rate,
            before[1],
            after_cer,
            delta_cer,
        )
    else:
        logger.info("Дообучение: exact-match %.3f | CER %.4f", after_rate, after_cer)

    payload = {
        "base_checkpoint": base_checkpoint,
        "checkpoint": checkpoint_path,
        "n_cases": n_cases,
        "exact_match_rate": after_rate,
        "cer": after_cer,
        "exact_match_rate_before": before[0] if before is not None else None,
        "cer_before": before[1] if before is not None else None,
        "delta_exact_match_rate": delta_rate,
        "delta_cer": delta_cer,
    }
    report_path = str(resolver.artifacts / "reports" / "finetune_recognition.json")
    try:
        import json

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        logger.info("Отчёт дообучения сохранён: %s", report_path)
    except Exception as exc:
        logger.warning("Не удалось сохранить отчёт дообучения: %s", exc)
