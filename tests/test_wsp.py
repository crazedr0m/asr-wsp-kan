"""Тесты Weight-Space Programming (src/kan_asr/wsp/)."""

import os
import sys

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kan_asr.models.tiny_kan_asr import FastKANConv1x1
from kan_asr.wsp.extract import WhisperWeightSpaceProgrammer
from kan_asr.wsp.hyper_adapter import WeightSpaceHyperAdapter
from kan_asr.wsp.project import project_mlp_to_fastkan


def _mock_whisper_sd():
    """Минимальный mock state_dict Whisper-tiny (n_state=384, ffn=1536)."""
    return {
        "encoder.conv1.weight": torch.randn(384, 80, 3),
        "encoder.layers.1.fc1.weight": torch.randn(1536, 384),
        "encoder.layers.1.fc1.bias": torch.randn(1536),
        "encoder.layers.1.fc2.weight": torch.randn(384, 1536),
        "encoder.layers.1.fc2.bias": torch.randn(384),
    }


def test_extract_conv_subspace():
    """SVD-проекция conv1: 80->40 мел-бинов, 3->5 по времени."""
    extractor = WhisperWeightSpaceProgrammer(_mock_whisper_sd(), device="cpu")
    dw, pw = extractor.extract_conv_subspace(target_in_mels=40, target_out_c=64, kernel_size=5)
    assert dw.shape == (40, 1, 1, 5), f"dw: {dw.shape}"
    assert pw.shape == (64, 40, 1, 1), f"pw: {pw.shape}"
    assert torch.isfinite(dw).all() and torch.isfinite(pw).all()


def test_project_mlp_to_fastkan():
    """Аппроксимация FFN Whisper базисом Гауссиан FastKAN."""
    sd = _mock_whisper_sd()
    kan = FastKANConv1x1(128, 128, num_grids=6)
    mse, r2 = project_mlp_to_fastkan(sd, kan, block_idx=1, num_samples=256)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    # Веса записаны в слой
    assert kan.base_conv.weight.shape == (128, 128, 1, 1)
    assert kan.rbf_conv.weight.shape == (128, 128 * 6, 1, 1)


def test_hyper_adapter():
    """WeightSpaceHyperAdapter генерирует низкоранговые Delta-W для kan1/kan2."""
    from kan_asr.models.tiny_kan_asr import TinyKAN_ASR

    model = TinyKAN_ASR(n_mels=40, vocab_size=35)
    adapter = WeightSpaceHyperAdapter(model, latent_dim=32, rank=8)
    delta1, delta2 = adapter.get_delta_weights()
    # Форма совпадает с rbf_conv.weight слоёв kan1/kan2: [C_out, C_in*G, 1, 1]
    assert delta1.shape == (128, 128 * 6, 1, 1), f"delta1: {delta1.shape}"
    assert delta2.shape == (128, 128 * 6, 1, 1), f"delta2: {delta2.shape}"
    assert torch.isfinite(delta1).all() and torch.isfinite(delta2).all()


if __name__ == "__main__":
    test_extract_conv_subspace()
    test_project_mlp_to_fastkan()
    test_hyper_adapter()
    print("[PASS] test_wsp")
