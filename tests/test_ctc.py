"""Тесты CTC-декодера и метрик (src/kan_asr/training/ctc.py, metrics.py)."""

import os
import sys

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kan_asr.data.vocab import VOCAB, greedy_ctc_decode
from kan_asr.training.ctc import ctc_log_probs
from kan_asr.training.metrics import cer, wer


def test_greedy_decode_shape():
    """greedy_ctc_decode возвращает список строк длиной batch_size."""
    logits = torch.randn(2, len(VOCAB), 1, 10)
    results = greedy_ctc_decode(logits)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_greedy_decode_collapse():
    """Повторяющиеся метки схлопываются, blank (0) игнорируется."""
    vocab = len(VOCAB)
    logits = torch.full((1, vocab, 1, 5), -10.0)
    # последовательность: blank, 'а'(2), 'а'(2), blank, 'б'(3)
    for t, idx in enumerate([0, 2, 2, 0, 3]):
        logits[0, idx, 0, t] = 10.0
    result = greedy_ctc_decode(logits)[0]
    assert result == "аб", f"Ожидается 'аб', получил '{result}'"


def test_ctc_log_probs_shape():
    """ctc_log_probs: [B, V, 1, T] -> [T, B, V]."""
    logits = torch.randn(2, len(VOCAB), 1, 7)
    lp = ctc_log_probs(logits)
    assert lp.shape == (7, 2, len(VOCAB)), f"Получил {lp.shape}"
    # log-softmax: exp суммируется в 1 по последней оси
    assert torch.allclose(lp.exp().sum(dim=-1), torch.ones(7, 2), atol=1e-5)


def test_cer_wer():
    """CER/WER на тривиальных примерах."""
    assert cer(["привет"], ["привет"]) == 0.0
    assert wer(["привет мир"], ["привет мир"]) == 0.0
    assert cer(["аб"], ["ав"]) == 0.5
    assert wer(["а б"], ["а в"]) == 0.5


if __name__ == "__main__":
    test_greedy_decode_shape()
    test_greedy_decode_collapse()
    test_ctc_log_probs_shape()
    test_cer_wer()
    print("[PASS] test_ctc")
