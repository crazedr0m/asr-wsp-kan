"""
CTC-декодер и утилиты для KAN-ASR.

Перенесено из KAN_ASR_Pipeline_Bundle/train_and_export.py (легаси-каталог).
"""

import torch

from ..data.vocab import IDX2CHAR


def greedy_ctc_decode(logits: torch.Tensor) -> list[str]:
    """
    Greedy CTC-декодинг.

    logits: [Batch, Vocab, 1, Time]
    returns: список предсказанных строк
    """
    preds = torch.argmax(logits.squeeze(2), dim=1)  # [B, T]
    results = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                chars.append(IDX2CHAR.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


def ctc_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Приводит логиты модели [B, Vocab, 1, T] к log-вероятностям [T, B, Vocab],
    как того требует ``torch.nn.CTCLoss``.
    """
    return torch.log_softmax(logits.squeeze(2).permute(2, 0, 1), dim=-1)
