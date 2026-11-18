"""
Dataset / DataLoader для KAN-ASR.

Перенесено из KAN_ASR_Pipeline_Bundle/train_and_export.py (легаси-каталог).
"""

import csv
import os

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .vocab import text_to_ids


class ASRDataset(Dataset):
    """Датасет из каталога с ``metadata.csv`` (формат ``filename|text``)."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.items = []
        meta_file = os.path.join(root_dir, "metadata.csv")
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                self.items.append((os.path.join(root_dir, row["filename"]), row["text"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        _, data = wav.read(wav_path)
        audio = torch.from_numpy(data.astype(np.float32) / 32768.0)
        targets = torch.tensor(text_to_ids(text), dtype=torch.long)
        return audio, targets


def collate_fn(batch):
    """Паддинг аудио и конкатенация CTC-таргетов."""
    audios, targets = zip(*batch)
    audio_lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
    padded_audios = nn.utils.rnn.pad_sequence(audios, batch_first=True)
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    concat_targets = torch.cat(targets)
    return padded_audios, audio_lens, concat_targets, target_lens
