"""
Цикл обучения TinyKAN_ASR (CTC) с сохранением калибровочных данных для RKNN.

Порт из ``KAN_ASR_Pipeline_Bundle/train_and_export.py`` с пакетными импортами
и единым резолвером путей.

Дополнительно поддерживает:
- загрузку WSP-инициализации (``wsp_checkpoint``) перед стартом обучения;
- CKA-выравнивание с учителем Whisper-tiny (``use_alignment``).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import ASRDataset, collate_fn
from ..data.mel import PureMelSpectrogram
from ..models.tiny_kan_asr import TinyKAN_ASR
from ..utils.logging import get_logger
from ..utils.paths import resolver
from .alignment import WhisperKANAlignmentLoss
from .ctc import ctc_log_probs

logger = get_logger("training.train")


def _load_wsp_checkpoint(model: nn.Module, wsp_checkpoint: str, device: torch.device) -> bool:
    """Загружает WSP-инициализацию в модель. Возвращает True при успехе."""
    if not wsp_checkpoint:
        return False
    if not os.path.exists(wsp_checkpoint):
        logger.warning("WSP-чекпоинт не найден: %s — обучение со случайной инициализации", wsp_checkpoint)
        return False

    state = torch.load(wsp_checkpoint, map_location=device)
    # Допускаем как «голый» state_dict, так и обёртку {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("WSP: отсутствующие ключи (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("WSP: лишние ключи (%d): %s", len(unexpected), unexpected[:5])
    logger.info("WSP-инициализация загружена: %s", wsp_checkpoint)
    return True


def _build_alignment(
    model: nn.Module,
    whisper_state: str,
    device: torch.device,
    anchor_weight: float = 1e-3,
):
    """
    Создаёт учителя Whisper и критерий CKA-выравнивания.

    Возвращает ``(teacher, align_criterion)`` либо ``(None, None)``, если
    веса учителя недоступны (тогда выравнивание отключается).
    """
    from ..models.whisper_teacher import WhisperTinyTeacher

    if not whisper_state or not os.path.exists(whisper_state):
        logger.warning(
            "CKA-выравнивание отключено: веса Whisper не найдены (%s)", whisper_state
        )
        return None, None

    teacher = WhisperTinyTeacher().to(device)
    teacher.load_hf_weights(whisper_state, device=str(device))
    teacher.eval()

    align_criterion = WhisperKANAlignmentLoss(
        student_dim=128, teacher_dim=384, anchor_weight=anchor_weight
    ).to(device)
    # Якорь фиксирует веса KAN ПОСЛЕ загрузки WSP-инициализации
    align_criterion.register_wsp_anchors(model)
    logger.info("CKA-выравнивание включено (учитель: %s)", whisper_state)
    return teacher, align_criterion


def train(
    data_dir: str = None,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    calib_dir: str = None,
    calib_file: str = None,
    checkpoint_path: str = None,
    device: str = None,
    wsp_checkpoint: str = None,
    use_alignment: bool = False,
    align_weight: float = 0.3,
    whisper_state: str = None,
    anchor_weight: float = 1e-3,
    num_calib_samples: int = 20,
    calib_window_frames: int = 200,
    use_hyper_adapter: bool = False,
    adapter_rank: int = 8,
    adapter_latent_dim: int = 32,
    base_checkpoint: str = None,
):
    """
    Обучает TinyKAN_ASR и сохраняет калибровочные данные для RKNN INT8.

    Параметры
    ---------
    wsp_checkpoint : str, optional
        Путь к state_dict, полученному ``scripts/wsp_init.py``. Загружается
        до старта обучения (Weight-Space Initialization).
    use_alignment : bool
        Включить CKA-выравнивание с учителем Whisper-tiny.
    align_weight : float
        Вес компоненты выравнивания в общем лоссе (по умолчанию 0.3).
    whisper_state : str, optional
        Путь к state_dict Whisper-tiny для учителя.

    Возвращает обученную модель.
    """
    if data_dir is None:
        data_dir = str(resolver.synth)
    if calib_dir is None:
        calib_dir = str(resolver.calib)
    if calib_file is None:
        calib_file = str(resolver.calib / "calib_dataset.txt")
    if checkpoint_path is None:
        checkpoint_path = str(resolver.checkpoints / "best.pt")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    logger.info("Device: %s | BF16: %s", device, use_bf16)

    dataset = ASRDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    mel_extractor = PureMelSpectrogram(n_mels=40).to(device)
    model = TinyKAN_ASR().to(device)

    # 1. Weight-Space Initialization (до создания оптимизатора и якорей)
    _load_wsp_checkpoint(model, wsp_checkpoint, device)

    # 2. CKA-выравнивание (якоря регистрируются после WSP-загрузки)
    teacher = None
    align_criterion = None
    mel_teacher = None
    if use_alignment:
        teacher, align_criterion = _build_alignment(
            model, whisper_state, device, anchor_weight=anchor_weight
        )
        if teacher is not None:
            mel_teacher = PureMelSpectrogram(n_mels=80).to(device)

    # 3. Hyper-Adapter: дообучение только низкоранговых поправок Delta-W
    #    (защита от катастрофического забывания на малых данных из бота).
    adapter = None
    base_rbf = {}
    if use_hyper_adapter:
        from ..wsp.hyper_adapter import WeightSpaceHyperAdapter

        if base_checkpoint:
            _load_wsp_checkpoint(model, base_checkpoint, device)
        for p in model.parameters():
            p.requires_grad_(False)
        adapter = WeightSpaceHyperAdapter(
            model, latent_dim=adapter_latent_dim, rank=adapter_rank
        ).to(device)
        base_rbf = {
            "kan1": model.kan1.rbf_conv.weight.detach().clone(),
            "kan2": model.kan2.rbf_conv.weight.detach().clone(),
        }
        logger.info(
            "Hyper-Adapter включён: обучаем %d параметров адаптера",
            sum(p.numel() for p in adapter.parameters()),
        )

    if adapter is not None:
        params = list(adapter.parameters())
    else:
        params = list(model.parameters())
    if align_criterion is not None:
        params += list(align_criterion.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(calib_dir, exist_ok=True)
    calib_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_ctc = 0.0
        total_cka = 0.0
        total_cos = 0.0
        n_batches = 0

        for audios, audio_lens, targets, target_lens in loader:
            audios, targets = audios.to(device), targets.to(device)
            with torch.no_grad():
                mels = mel_extractor(audios)  # [B, 40, 1, T]
                # Сохраняем первые N спектрограмм для калибровки RKNN INT8
                if epoch == 1 and len(calib_list) < num_calib_samples:
                    for b in range(mels.shape[0]):
                        if len(calib_list) >= num_calib_samples:
                            break
                        # Фиксируем окно для статического входа NPU
                        sample_mel = F.pad(
                            mels[b : b + 1, :, :, :calib_window_frames],
                            (0, max(0, calib_window_frames - mels.shape[-1])),
                        )
                        npy_path = os.path.join(
                            calib_dir, f"calib_{len(calib_list):03d}.npy"
                        )
                        np.save(npy_path, sample_mel.cpu().numpy())
                        calib_list.append(npy_path)

            if adapter is not None:
                d1, d2 = adapter.get_delta_weights()
                model.kan1.rbf_conv.weight = nn.Parameter(base_rbf["kan1"] + d1)
                model.kan2.rbf_conv.weight = nn.Parameter(base_rbf["kan2"] + d2)

            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=device.type, enabled=use_bf16, dtype=torch.bfloat16
            ):
                if teacher is not None:
                    logits, h1, h2 = model(mels, return_all_features=True)
                else:
                    logits = model(mels)  # [B, Vocab, 1, T_out]

                log_probs = ctc_log_probs(logits)  # [T_out, B, Vocab]
                input_lengths = torch.full(
                    (logits.shape[0],), logits.shape[-1], dtype=torch.long, device=device
                )
                loss_ctc = ctc_loss(log_probs, targets, input_lengths, target_lens.to(device))
                loss = loss_ctc

                if teacher is not None:
                    with torch.no_grad():
                        mel80 = mel_teacher(audios).squeeze(2)  # [B, 80, T]
                        t_feats = teacher.forward_features(mel80)
                    align = align_criterion(
                        student_mid=h1.squeeze(2),
                        student_top=h2.squeeze(2),
                        teacher_mid=t_feats["block_1"],
                        teacher_top=t_feats["final"],
                        student_model=model,
                    )
                    loss = loss_ctc + align_weight * align["total_align"]
                    total_cka += float(align["cka"])
                    total_cos += float(align["cosine"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_ctc += float(loss_ctc.item())
            n_batches += 1

        n_batches = max(1, n_batches)
        if teacher is not None:
            logger.info(
                "Epoch %d/%d | Total: %.4f | CTC: %.4f | CKA: %.4f | Cosine: %.4f",
                epoch,
                epochs,
                total_loss / n_batches,
                total_ctc / n_batches,
                total_cka / n_batches,
                total_cos / n_batches,
            )
        else:
            logger.info(
                "Epoch %d/%d | CTC Loss: %.4f", epoch, epochs, total_ctc / n_batches
            )

    # Сохраняем калибровочный список для rknn-toolkit2
    os.makedirs(os.path.dirname(calib_file) or ".", exist_ok=True)
    with open(calib_file, "w", encoding="utf-8") as f:
        for p in calib_list:
            f.write(p + "\n")

    # Сохраняем чекпоинт (для Hyper-Adapter — уже слитые base + Delta-W)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Чекпоинт сохранён: %s", checkpoint_path)
    if adapter is not None:
        adapter_path = os.path.join(
            os.path.dirname(checkpoint_path) or ".", "hyper_adapter.pt"
        )
        torch.save(adapter.state_dict(), adapter_path)
        logger.info("Hyper-Adapter сохранён: %s", adapter_path)
    logger.info("Список калибровки: %s", calib_file)

    return model


if __name__ == "__main__":
    train()
