"""Weight-Space Programming модуль."""

from .extract import WhisperWeightSpaceProgrammer
from .project import project_mlp_to_fastkan
from .initialize import initialize_student, extract_conv_subspace
from .hyper_adapter import WeightSpaceHyperAdapter

__all__ = [
    "WhisperWeightSpaceProgrammer",
    "project_mlp_to_fastkan",
    "initialize_student",
    "extract_conv_subspace",
    "WeightSpaceHyperAdapter",
]
