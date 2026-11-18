"""KAN-ASR data модуль."""

from .mel import PureMelSpectrogram
from .vocab import VOCAB, CHAR2IDX, IDX2CHAR, text_to_ids, greedy_ctc_decode
from .augment import add_augmentations, SAMPLE_RATE
from .generate import generate_synthetic_dataset, synthesize_with_piper, SAMPLE_PHRASES
from .dataset import ASRDataset, collate_fn

__all__ = [
    "PureMelSpectrogram",
    "VOCAB",
    "CHAR2IDX",
    "IDX2CHAR",
    "text_to_ids",
    "greedy_ctc_decode",
    "add_augmentations",
    "SAMPLE_RATE",
    "generate_synthetic_dataset",
    "synthesize_with_piper",
    "SAMPLE_PHRASES",
    "ASRDataset",
    "collate_fn",
]
