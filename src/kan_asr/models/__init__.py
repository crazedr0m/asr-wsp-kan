"""KAN-ASR модели."""

from .kan_layers import (
    PureFastKANLayer,
    PureChebyKANLayer,
    NPUFriendlyMoEKAN,
)
from .tiny_kan_asr import (
    FastKANConv1x1,
    DepthwiseSeparableConv1D_as_2D,
    TinyKAN_ASR,
)
from .cuda_fastkan import CudaFastKANConv1x1
from .whisper_teacher import WhisperTinyTeacher

__all__ = [
    "PureFastKANLayer",
    "PureChebyKANLayer",
    "NPUFriendlyMoEKAN",
    "FastKANConv1x1",
    "DepthwiseSeparableConv1D_as_2D",
    "TinyKAN_ASR",
    "CudaFastKANConv1x1",
    "WhisperTinyTeacher",
]
