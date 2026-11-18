"""
Метрики качества распознавания: WER / CER.

Реализованы на чистом Python (без внешних зависимостей).
"""


def _edit_distance(ref: list, hyp: list) -> int:
    """Расстояние Левенштейна между двумя последовательностями."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(references: list[str], hypotheses: list[str]) -> float:
    """Character Error Rate (суммарный по корпусу)."""
    total_dist = 0
    total_len = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        total_dist += _edit_distance(list(ref), list(hyp))
        total_len += len(ref)
    return total_dist / max(1, total_len)


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate (суммарный по корпусу)."""
    total_dist = 0
    total_len = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_dist += _edit_distance(ref_words, hyp_words)
        total_len += len(ref_words)
    return total_dist / max(1, total_len)
