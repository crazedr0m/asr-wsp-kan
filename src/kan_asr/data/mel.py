"""
Чистый Mel-спектрограф без TorchAudio.

Модуль содержит класс PureMelSpectrogram для вычисления Log-Mel спектрограммы.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureMelSpectrogram(nn.Module):
    """
    Вычисление Log-Mel спектрограммы на чистом PyTorch (совместимо с C++ / NPU).
    """
    
    def __init__(self, sr: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 40):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        fb = self._build_mel_filterbank(sr, n_fft, n_mels)
        self.register_buffer("mel_fb", fb)

    @staticmethod
    def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        mel_points = torch.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = torch.floor((n_fft + 1) * hz_points / sr).long()
        
        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
            for k in range(f_m_minus, f_m):
                fb[m - 1, k] = (k - f_m_minus) / max(1, (f_m - f_m_minus))
            for k in range(f_m, f_m_plus):
                fb[m - 1, k] = (f_m_plus - k) / max(1, (f_m_plus - f_m))
        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [Batch, Samples]
        returns: [Batch, n_mels, 1, Time]
        """
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True
        )
        power_spec = stft.abs().pow(2.0)  # [B, Freq, Time]
        mel_spec = torch.matmul(self.mel_fb, power_spec)  # [B, n_mels, Time]
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        # Нормализация по каждому примеру
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-5
        norm_mel = (log_mel - mean) / std
        return norm_mel.unsqueeze(2)  # [B, n_mels, 1, Time]
