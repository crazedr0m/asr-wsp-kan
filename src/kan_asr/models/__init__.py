"""KAN-ASR модели."""

from .cuda_fastkan import CudaFastKANConv1x1
from .kan_layers import (
    NPUFriendlyMoEKAN,
    PureChebyKANLayer,
    PureFastKANLayer,
)
from .tiny_kan_asr import (
    DepthwiseSeparableConv1D_as_2D,
    FastKANConv1x1,
    TinyKAN_ASR,
)
from .whisper_teacher import WhisperTinyTeacher

__all__ = [
    "CudaFastKANConv1x1",
    "DepthwiseSeparableConv1D_as_2D",
    "FastKANConv1x1",
    "NPUFriendlyMoEKAN",
    "PureChebyKANLayer",
    "PureFastKANLayer",
    "TinyKAN_ASR",
    "WhisperTinyTeacher",
]
