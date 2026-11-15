# -*- coding: utf-8 -*-
"""Тесты KAN-слоёв и архитектуры TinyKAN_ASR."""

import os
import sys

try:
    import kan_asr  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kan_asr.models.kan_layers import (
    NPUFriendlyMoEKAN,
    PureChebyKANLayer,
    PureFastKANLayer,
)
from kan_asr.models.tiny_kan_asr import (
    DepthwiseSeparableConv1D_as_2D,
    FastKANConv1x1,
    TinyKAN_ASR,
)


def test_pure_fastkan_layer():
    layer = PureFastKANLayer(8, 4, num_grids=6)
    out = layer(torch.randn(5, 8))
    assert out.shape == (5, 4)


def test_pure_chebykan_layer():
    layer = PureChebyKANLayer(8, 4, degree=4)
    out = layer(torch.randn(5, 8))
    assert out.shape == (5, 4)


def test_moe_kan():
    layer = NPUFriendlyMoEKAN(8, 4, num_experts=3)
    out = layer(torch.randn(5, 8))
    assert out.shape == (5, 4)


def test_fastkan_conv1x1():
    """FastKANConv1x1: [B, C, H, T] -> [B, C_out, H, T]."""
    layer = FastKANConv1x1(16, 32, num_grids=6)
    x = torch.randn(2, 16, 1, 20)
    out = layer(x)
    assert out.shape == (2, 32, 1, 20)
    # WSP-атрибуты должны присутствовать
    assert hasattr(layer, "grid")
    assert hasattr(layer, "denom")
    assert hasattr(layer, "base_conv")
    assert hasattr(layer, "rbf_conv")


def test_depthwise_separable():
    layer = DepthwiseSeparableConv1D_as_2D(40, 64, kernel_size=5, stride=2)
    x = torch.randn(2, 40, 1, 100)
    out = layer(x)
    assert out.shape == (2, 64, 1, 50)
    assert hasattr(layer, "dw")
    assert hasattr(layer, "pw")


def test_tiny_kan_asr_forward():
    model = TinyKAN_ASR(n_mels=40, vocab_size=35)
    model.eval()
    x = torch.randn(1, 40, 1, 100)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 35
    assert logits.shape[2] == 1


def test_tiny_kan_asr_features():
    """return_features=True возвращает промежуточные признаки для CKA."""
    model = TinyKAN_ASR(n_mels=40, vocab_size=35)
    model.eval()
    x = torch.randn(1, 40, 1, 100)
    with torch.no_grad():
        logits, feats = model(x, return_features=True)
    assert logits.shape[1] == 35
    assert feats.shape[1] == 128


if __name__ == "__main__":
    test_pure_fastkan_layer()
    test_pure_chebykan_layer()
    test_moe_kan()
    test_fastkan_conv1x1()
    test_depthwise_separable()
    test_tiny_kan_asr_forward()
    test_tiny_kan_asr_features()
    print("[PASS] test_kan_layers")
