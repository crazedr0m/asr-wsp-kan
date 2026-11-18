"""
Smoke-тесты для KAN-ASR.

Проверяет базовую функциональность:
- Формы тензоров в моделях
- CTC-декодер
- Mel-спектрограф
- WSP-инициализацию
"""

import os
import sys

# Bootstrap: если пакет не установлен, добавляем src/ в path
try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


def test_mel_spectrogram():
    """Тест PureMelSpectrogram."""
    from kan_asr.data.mel import PureMelSpectrogram

    mel = PureMelSpectrogram(sr=16000, n_fft=400, hop_length=160, n_mels=40)
    waveform = torch.randn(1, 16000)
    mel_out = mel(waveform)

    # center=True (по умолчанию в torch.stft): 1 + 16000 // 160 = 101 кадр
    assert mel_out.shape == (1, 40, 1, 101), f"Ожидается (1, 40, 1, 101), получил {mel_out.shape}"
    print("[PASS] test_mel_spectrogram")


def test_vocab():
    """Тест словаря."""
    from kan_asr.data.vocab import CHAR2IDX, IDX2CHAR, VOCAB, text_to_ids

    assert len(VOCAB) > 0, "VOCAB должен быть непустым"
    assert len(CHAR2IDX) == len(VOCAB), "CHAR2IDX должен соответствовать VOCAB"
    assert len(IDX2CHAR) == len(VOCAB), "IDX2CHAR должен соответствовать VOCAB"

    text = "привет мир"
    ids = text_to_ids(text)
    assert isinstance(ids, list), "text_to_ids должен возвращать список"
    print("[PASS] test_vocab")


def test_ctc_decode():
    """Тест CTC декодера."""
    from kan_asr.data.vocab import greedy_ctc_decode

    # Создаём dummy logits
    batch_size = 1
    vocab_size = len(["<blank>", " ", *list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")])
    time_steps = 10

    logits = torch.randn(batch_size, vocab_size, 1, time_steps)
    results = greedy_ctc_decode(logits)

    assert isinstance(results, list), "greedy_ctc_decode должен возвращать список"
    assert len(results) == batch_size, "Должно быть столько же результатов, сколько batch_size"
    print("[PASS] test_ctc_decode")


def test_model_forward():
    """Тест forward модели."""
    from kan_asr.models.tiny_kan_asr import TinyKAN_ASR

    model = TinyKAN_ASR(n_mels=40, vocab_size=35)
    model.eval()

    # Mel-спектрограмма: (batch, n_mels, 1, time)
    batch_size = 1
    time_steps = 100
    mel_input = torch.randn(batch_size, 40, 1, time_steps)

    with torch.no_grad():
        logits = model(mel_input)

    assert logits.shape[0] == batch_size, "Batch dimension должна совпадать"
    assert logits.shape[1] == 35, "Vocab dimension должна быть 35"
    assert logits.shape[2] == 1, "Канальная dimension должна быть 1"
    print("[PASS] test_model_forward")


def test_wsp_extractor():
    """Тест WSP экстрактора (без реальных весов Whisper)."""
    from kan_asr.wsp.extract import WhisperWeightSpaceProgrammer

    # Создаём mock state_dict
    mock_sd = {
        "encoder.conv1.weight": torch.randn(384, 80, 3),
    }

    extractor = WhisperWeightSpaceProgrammer(mock_sd, device="cpu")

    # Проверяем, что экстрактор создан
    assert extractor is not None, "Экстрактор должен быть создан"
    print("[PASS] test_wsp_extractor")


def run_all_tests():
    """Запуск всех smoke-тестов."""
    print("=" * 50)
    print("Smoke-тесты KAN-ASR")
    print("=" * 50)

    tests = [
        test_mel_spectrogram,
        test_vocab,
        test_ctc_decode,
        test_model_forward,
        test_wsp_extractor,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Результаты: {passed} пройдено, {failed} провалено")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
