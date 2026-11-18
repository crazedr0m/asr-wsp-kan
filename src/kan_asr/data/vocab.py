"""
Словарь и утилиты для работы с текстом.

Модуль содержит VOCAB, CHAR2IDX, IDX2CHAR и функции text_to_ids, greedy_ctc_decode.
"""

VOCAB = ["<blank>", " "] + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}


def text_to_ids(text: str) -> list[int]:
    """Преобразование текста в список ID."""
    return [CHAR2IDX[c] for c in text.lower() if c in CHAR2IDX and c != "<blank>"]


def greedy_ctc_decode(logits: "torch.Tensor") -> list[str]:  # type: ignore
    """
    CTC декодинг (greedy).

    logits: [Batch, Vocab, 1, Time]
    returns: список предсказанных строк
    """
    import torch
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
