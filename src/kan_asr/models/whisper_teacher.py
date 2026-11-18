"""
Whisper-учитель для CKA-выравнивания.

Перенесено из Whisper_CKA_Alignment/standalone_whisper_teacher.py (легаси-каталог).
Автономный энкодер Whisper-tiny на чистом PyTorch (без transformers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        head_dim = C // self.n_head
        q = self.query(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)


class WhisperResidualBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = WhisperAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperTinyTeacher(nn.Module):
    """
    Легкий автономный энкодер Whisper-tiny на чистом PyTorch (без transformers).
    Вход: Log-Mel спектрограмма [Batch, 80, Time] (шаг 10 мс).
    Выход: словарь промежуточных скрытых состояний [Batch, 384, Time // 2] (шаг 20 мс).
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_ctx: int = 1500,
        n_state: int = 384,
        n_head: int = 6,
        n_layer: int = 4,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.zeros(n_ctx, n_state))
        self.blocks = nn.ModuleList([WhisperResidualBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def load_hf_weights(self, state_dict_path: str, device: str = "cpu"):
        """Загрузка весов из HF state_dict с переименованием ключей под эту архитектуру."""
        raw_sd = torch.load(state_dict_path, map_location=device, weights_only=True)
        mapped = {}
        for k, v in raw_sd.items():
            if not k.startswith("model.encoder."):
                continue
            new_k = k.replace("model.encoder.", "")
            new_k = new_k.replace("embed_positions.weight", "positional_embedding")
            new_k = new_k.replace("layers.", "blocks.")
            new_k = new_k.replace("self_attn.q_proj", "attn.query")
            new_k = new_k.replace("self_attn.k_proj", "attn.key")
            new_k = new_k.replace("self_attn.v_proj", "attn.value")
            new_k = new_k.replace("self_attn.out_proj", "attn.out")
            new_k = new_k.replace("self_attn_layer_norm", "attn_ln")
            new_k = new_k.replace("fc1", "mlp.0")
            new_k = new_k.replace("fc2", "mlp.2")
            new_k = new_k.replace("final_layer_norm", "mlp_ln")
            new_k = new_k.replace("layer_norm.", "ln_post.")
            mapped[new_k] = v

        self.load_state_dict(mapped, strict=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_features(self, mel_80: torch.Tensor) -> dict:
        # Вход mel_80: B x 80 x T
        x = F.gelu(self.conv1(mel_80))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [B, T//2, 384]

        T_half = x.size(1)
        pos = self.positional_embedding[:T_half].unsqueeze(0)
        x = x + pos

        features = {}
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in (1, 3):
                # Возвращаем в формате [B, C, T//2]
                features[f"block_{idx}"] = x.permute(0, 2, 1).contiguous()

        features["final"] = self.ln_post(x).permute(0, 2, 1).contiguous()
        return features
