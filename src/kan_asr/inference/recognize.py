"""
Единый пайплайн распознавания речи WAV → текст для KAN-ASR.

Модуль объединяет шаги, которые раньше были разбросаны по коду:

    WAV → load_audio → audio_to_mel → fit_mel_length → модель → greedy_ctc_decode

и предоставляет три уровня инференса:

- :func:`recognize_torch` — хост (PyTorch, float32);
- :func:`recognize_rknn_simulator` — симулятор NPU (RKNN INT8, ``target=None``);
- :func:`recognize_on_device` — реальная плата Luckfox (RKNN INT8, ``target='rv1103'``).

Плюс сквозная оценка качества :func:`evaluate_recognition`, которая считает
долю точных совпадений (``recognized_text == reference_text``), CER и WER.
Она переиспользуется тестами и этапами пайплайна (обучение, дообучение,
конвертация INT8, прогон на плате).

Модуль не требует ``rknn-toolkit2`` на импорте: RKNN импортируется лениво
внутри соответствующих функций.
"""

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
import json
import os
import time

import numpy as np

from ..data.mel import PureMelSpectrogram
from ..data.vocab import greedy_ctc_decode
from ..training.metrics import cer as _cer, wer as _wer
from ..utils.paths import resolver

# --- Константы -------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 40
DEFAULT_TIME_FRAMES = 200
DEFAULT_INPUT_SHAPE = (1, DEFAULT_N_MELS, 1, DEFAULT_TIME_FRAMES)
DEFAULT_TARGET = "rv1103"

# Порог CER по умолчанию мягкий: крошечная модель может ошибаться.
DEFAULT_MAX_CER = 1.0
# Доля точных совпадений по умолчанию не ограничивается (0.0 — только отчёт).
DEFAULT_MIN_EXACT_MATCH = 0.0

DEFAULT_RECOGNITION_REPORT = "recognition_report.json"

_MEL_CACHE: dict[tuple[int, int], PureMelSpectrogram] = {}


# --- Загрузка аудио и признаки --------------------------------------------


def load_audio(path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    Читает WAV и приводит его к моно float32 с частотой ``target_sr``.

    Args:
        path: путь к WAV-файлу.
        target_sr: целевая частота дискретизации (по умолчанию 16 кГц).

    Returns:
        Одномерный массив ``float32`` в диапазоне ``[-1, 1]``.
    """
    import scipy.io.wavfile as wav
    from scipy.signal import resample

    if not os.path.exists(path):
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")

    sr, data = wav.read(path)
    audio = np.asarray(data)

    # Приведение к float32 в диапазоне [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    # Стерео → моно
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Ресемплинг до целевой частоты
    if sr != target_sr and audio.size > 0:
        num_samples = round(audio.size * target_sr / float(sr))
        audio = resample(audio, num_samples).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _get_mel(n_mels: int = DEFAULT_N_MELS, sr: int = DEFAULT_SAMPLE_RATE) -> PureMelSpectrogram:
    """Кэшированный экстрактор Mel-спектрограммы."""
    key = (n_mels, sr)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = PureMelSpectrogram(sr=sr, n_mels=n_mels).eval()
    return _MEL_CACHE[key]


def audio_to_mel(
    waveform: np.ndarray,
    n_mels: int = DEFAULT_N_MELS,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Считает Log-Mel спектрограмму: ``[Samples]`` или ``[B, Samples]`` → ``[B, n_mels, 1, T]``.

    Использует тот же :class:`PureMelSpectrogram`, что и обучение, поэтому
    признаки на хосте, в симуляторе и на плате согласованы.
    """
    import torch

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Ожидался 1D или 2D массив, получено {arr.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    with torch.no_grad():
        mel = _get_mel(n_mels, sr)(tensor)
    return mel.cpu().numpy().astype(np.float32)


def fit_mel_length(mel: np.ndarray, time_frames: int = DEFAULT_TIME_FRAMES) -> np.ndarray:
    """
    Приводит Mel к статическому входу модели ``[B, n_mels, 1, time_frames]``.

    Более длинные последовательности обрезаются, короткие — паддятся нулями
    (модель экспортируется с фиксированным окном 2 с = 200 кадров).
    """
    arr = np.asarray(mel, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Ожидался 4D тензор [B, n_mels, 1, T], получено {arr.shape}")

    current = arr.shape[-1]
    if current == time_frames:
        return arr
    if current > time_frames:
        return np.ascontiguousarray(arr[..., :time_frames])
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, time_frames - current)]
    return np.pad(arr, pad_width, mode="constant").astype(np.float32)


def wav_to_mel_input(
    wav_path: str,
    time_frames: int = DEFAULT_TIME_FRAMES,
    n_mels: int = DEFAULT_N_MELS,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Полный путь WAV → готовый вход модели ``[1, n_mels, 1, time_frames]``."""
    audio = load_audio(wav_path, target_sr=sr)
    mel = audio_to_mel(audio, n_mels=n_mels, sr=sr)
    return fit_mel_length(mel, time_frames=time_frames)


# --- Кейсы распознавания ---------------------------------------------------


@dataclass
class RecognitionCase:
    """Пара «аудио + эталонный текст» для E2E-проверки распознавания."""

    wav_path: str
    text: str


def load_recognition_cases(
    n: int | None = None,
    data_dir: str | None = None,
    offset: int = 0,
) -> list[RecognitionCase]:
    """
    Читает ``metadata.csv`` (формат ``filename|text``) и возвращает кейсы.

    Отбор детерминированный: строки берутся в порядке файла, начиная с
    ``offset``. Это позволяет выделить held-out «хвост» для честной оценки
    после обучения (``offset`` = размер train-части).

    Args:
        n: сколько кейсов вернуть (``None`` — все до конца файла).
        data_dir: каталог с ``metadata.csv`` (по умолчанию ``data/synth``).
        offset: сколько первых строк пропустить.

    Returns:
        Список :class:`RecognitionCase` (может быть пустым, если нет данных).
    """
    if data_dir is None:
        data_dir = str(resolver.synth)

    meta_file = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta_file):
        return []

    cases: list[RecognitionCase] = []
    with open(meta_file, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                continue
            cases.append(RecognitionCase(os.path.join(data_dir, filename), text))

    selected = cases[offset:]
    if n is not None:
        selected = selected[:n]
    return selected


# --- Декодирование ---------------------------------------------------------


def _decode_logits(logits: np.ndarray) -> str:
    """Декодирует CTC-логиты ``[B, Vocab, 1, T]`` в строку (greedy)."""
    import torch

    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, None, :]
    tensor = torch.from_numpy(arr)
    decoded = greedy_ctc_decode(tensor)
    return decoded[0] if decoded else ""


# --- Хост (PyTorch) --------------------------------------------------------


def _load_torch_model(checkpoint_path: str, device: str = "cpu"):
    """Загружает TinyKAN_ASR из чекпоинта (поддерживает обёртку ``state_dict``)."""
    import torch

    from ..models.tiny_kan_asr import TinyKAN_ASR

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    model = TinyKAN_ASR()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def recognize_torch(
    wav_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    time_frames: int = DEFAULT_TIME_FRAMES,
) -> str:
    """
    Распознаёт WAV на хосте (PyTorch, float32).

    Args:
        wav_path: путь к WAV.
        checkpoint_path: путь к ``.pt`` со ``state_dict`` модели.
        device: ``cpu`` или ``cuda``.
        time_frames: длина окна Mel (по умолчанию 200 кадров = 2 с).

    Returns:
        Распознанный текст.
    """
    import torch

    model = _load_torch_model(checkpoint_path, device=device)
    mel = wav_to_mel_input(wav_path, time_frames=time_frames)
    with torch.no_grad():
        logits = model(torch.from_numpy(mel).to(device))
    return _decode_logits(logits.cpu().numpy())


# --- Симулятор NPU ---------------------------------------------------------


def recognize_rknn_simulator(
    wav_path: str,
    onnx_model: str | None = None,
    rknn_model: str | None = None,
    calib_dataset: str | None = None,
    time_frames: int = DEFAULT_TIME_FRAMES,
    data_format: str = "nchw",
    verbose: bool = False,
) -> str:
    """
    Распознаёт WAV в симуляторе NPU (RKNN INT8, ``target=None``).

    Mel считается на хосте, затем подаётся в INT8-модель. ``.rknn`` собирается
    из ONNX в текущем процессе (симулятор не принимает ``load_rknn``).

    Args:
        wav_path: путь к WAV.
        onnx_model: путь к ONNX (по умолчанию ``artifacts/onnx/tiny_kan_asr_rv1103.onnx``).
        rknn_model: путь к ``.rknn`` (по умолчанию ``artifacts/rknn/tiny_kan_asr_rv1103.rknn``).
        calib_dataset: файл со списком калибровочных ``.npy``.
        time_frames: длина окна Mel.
        data_format: формат входа для ``rknn.inference`` (``nchw``/``nhwc``).
        verbose: печатать прогресс.

    Returns:
        Распознанный текст.
    """
    from ..export.convert_rknn import build_rknn_model
    from ..export.verify_rknn import _import_rknn

    RKNN = _import_rknn()

    if onnx_model is None:
        onnx_model = str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")
    if rknn_model is None:
        rknn_model = str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")
    if calib_dataset is None:
        calib_dataset = str(resolver.calib / "calib_dataset.txt")

    if not os.path.exists(onnx_model):
        raise FileNotFoundError(f"ONNX-модель не найдена: {onnx_model}")
    if not os.path.exists(calib_dataset):
        raise FileNotFoundError(f"Калибровочный датасет не найден: {calib_dataset}")

    mel = wav_to_mel_input(wav_path, time_frames=time_frames)

    rknn = RKNN(verbose=verbose)
    try:
        if verbose:
            print("--> Сборка RKNN INT8 из ONNX...")
        build_rknn_model(rknn, onnx_model, rknn_model, calib_dataset)

        if verbose:
            print("--> Инициализация симулятора NPU (target=None)...")
        ret = rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError("Ошибка init_runtime(target=None)")

        if verbose:
            print("--> Инференс в эмуляторе NPU (INT8)...")
        outputs = rknn.inference(inputs=[mel], data_format=[data_format])
        if not outputs:
            raise RuntimeError("Эмулятор NPU не вернул выходов")
    finally:
        rknn.release()

    return _decode_logits(outputs[0])


# --- Реальная плата --------------------------------------------------------


@dataclass
class DeviceRecognitionResult:
    """Результат распознавания на реальной плате: текст + latency."""

    text: str
    latency_ms: float
    target: str
    device_id: str | None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "target": self.target,
            "device_id": self.device_id,
        }


def _read_input_quant(rknn) -> tuple[float, int] | None:
    """
    Читает параметры аффинного квантования входного тензора из runtime.

    Возвращает ``(scale, zero_point)`` или ``None``, если toolkit не отдаёт
    атрибуты. Нужен, чтобы квантовать Mel теми же scale/zp, что и модель,
    и не «срезать» значения при подаче на плату.
    """
    try:
        runtime = rknn.rknn_base.rknn_runtime
        attr = runtime.get_tensor_attr(0)
        scale = float(attr.scale)
        zp = int(attr.zp)
    except Exception:
        return None
    if scale <= 0:
        return None
    return scale, zp


def _quantize_mel_uint8(mel: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """
    Аффинное квантование Mel ``float32`` → ``uint8``.

    ``q = clip(round(x / scale) + zp, 0, 255)`` — те же ``scale``/``zp``, что
    и у входного тензора INT8-модели на плате.
    """
    q = np.round(np.asarray(mel, dtype=np.float32) / float(scale)) + float(zp)
    quantized: np.ndarray = np.clip(q, 0, 255).astype(np.uint8)
    return quantized


def _nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    """``[1, C, 1, T]`` → ``[1, 1, T, C]`` (NHWC-раскладка входа NPU)."""
    transposed: np.ndarray = np.transpose(arr, (0, 2, 3, 1))
    return np.ascontiguousarray(transposed)


def _prepare_device_input(
    mel: np.ndarray,
    rknn,
    data_format: str = "nhwc",
    input_dtype: str = "uint8",
    verbose: bool = False,
) -> np.ndarray:
    """
    Готовит вход для INT8-модели на плате.

    Квантованная модель на RV1103 принимает ``uint8`` в раскладке
    NHWC ``[1, 1, T, C]``. scale/zero-point берутся из самого runtime, чтобы
    значения Mel квантовались корректно (а не грубо «срезались» в [0, 255]).
    """
    from ..export.verify_rknn import _to_uint8_input

    if input_dtype != "uint8":
        arr = np.asarray(mel, dtype=np.float32)
        return _nchw_to_nhwc(arr) if data_format == "nhwc" else arr

    quant = _read_input_quant(rknn)
    if quant is None:
        if verbose:
            print("[WARN] Не удалось прочитать scale/zp входа — грубое квантование")
        arr = _to_uint8_input(mel)
    else:
        scale, zp = quant
        if verbose:
            print(f"--> Квантование входа: scale={scale:.6f}, zp={zp}")
        arr = _quantize_mel_uint8(mel, scale, zp)
    return _nchw_to_nhwc(arr) if data_format == "nhwc" else arr


def recognize_on_device(
    wav_path: str,
    rknn_model: str | None = None,
    target: str = DEFAULT_TARGET,
    device_id: str | None = None,
    time_frames: int = DEFAULT_TIME_FRAMES,
    data_format: str = "nhwc",
    input_dtype: str = "uint8",
    auto_start_server: bool = True,
    verbose: bool = False,
) -> DeviceRecognitionResult:
    """
    Распознаёт WAV на реальной плате Luckfox (RKNN INT8, ``target='rv1103'``).

    Mel считается на хосте (C++-Mel на устройстве ещё не реализован) и
    подаётся в INT8-модель по USB/ADB. При необходимости поднимает
    ``rknn_server`` на плате.

    Вход квантуется в ``uint8`` теми же scale/zp, что и модель (берутся из
    runtime), и раскладывается в NHWC ``[1, 1, T, C]`` — единственный формат,
    который принимает квантованная модель на плате.

    Args:
        wav_path: путь к WAV.
        rknn_model: путь к ``.rknn`` (по умолчанию ``artifacts/rknn/tiny_kan_asr_rv1103.rknn``).
        target: целевая платформа (``rv1103``).
        device_id: серийный номер платы (по умолчанию — единственное устройство).
        time_frames: длина окна Mel.
        data_format: формат входа для ``rknn.inference`` (``nhwc``/``nchw``).
        input_dtype: ``uint8`` (обязателен для квантованной модели) или ``float32``.
        auto_start_server: поднять ``rknn_server``, если он не запущен.
        verbose: печатать прогресс.

    Returns:
        :class:`DeviceRecognitionResult` с текстом и latency.
    """
    from ..export.verify_rknn import (
        _check_rknn_server,
        _ensure_device,
        _import_rknn,
        _start_rknn_server,
    )

    RKNN = _import_rknn()

    if rknn_model is None:
        rknn_model = str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")
    if not os.path.exists(rknn_model):
        raise FileNotFoundError(f"RKNN-модель не найдена: {rknn_model}")

    adb = None
    device_id = _ensure_device(device_id, adb)

    if auto_start_server and not _check_rknn_server(device_id, adb):
        if verbose:
            print("--> rknn_server не запущен, поднимаю на плате...")
        _start_rknn_server(device_id, adb)

    mel = wav_to_mel_input(wav_path, time_frames=time_frames)

    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.load_rknn(rknn_model)
        if ret != 0:
            raise RuntimeError("Ошибка загрузки .rknn")

        if verbose:
            print(f"--> init_runtime(target={target}, device_id={device_id})...")
        ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            raise RuntimeError(f"Ошибка init_runtime(target={target}, device_id={device_id})")

        sample_input = _prepare_device_input(
            mel,
            rknn,
            data_format=data_format,
            input_dtype=input_dtype,
            verbose=verbose,
        )

        start = time.perf_counter()
        outputs = rknn.inference(inputs=[sample_input], data_format=[data_format])
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not outputs:
            raise RuntimeError("Плата не вернула выходов")
    finally:
        rknn.release()

    return DeviceRecognitionResult(
        text=_decode_logits(outputs[0]),
        latency_ms=latency_ms,
        target=target,
        device_id=device_id,
    )


# --- Сквозная оценка качества ---------------------------------------------


@dataclass
class RecognitionReport:
    """Отчёт распознавания: доля точных совпадений, CER, WER, per-case детали."""

    n_cases: int
    exact_matches: int
    exact_match_rate: float
    cer: float
    wer: float
    max_cer: float | None = None
    min_exact_match: float | None = None
    cases: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Пройдены ли пороги (``max_cer`` и ``min_exact_match``)."""
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "exact_matches": self.exact_matches,
            "exact_match_rate": self.exact_match_rate,
            "cer": self.cer,
            "wer": self.wer,
            "max_cer": self.max_cer,
            "min_exact_match": self.min_exact_match,
            "passed": self.passed,
            "failures": list(self.failures),
            "cases": list(self.cases),
        }

    def summary(self) -> str:
        headline = (
            f"Распознавание: {self.n_cases} кейсов | "
            f"точных совпадений {self.exact_matches} "
            f"({self.exact_match_rate * 100:.1f}%) | "
            f"CER={self.cer:.4f} | WER={self.wer:.4f}"
        )
        lines = [headline]
        for failure in self.failures:
            lines.append(f"  [FAIL] {failure}")
        return "\n".join(lines)


def evaluate_recognition(
    cases: Sequence[RecognitionCase],
    recognize_fn: Callable[[str], str],
    max_cer: float | None = None,
    min_exact_match: float | None = None,
    verbose: bool = False,
) -> RecognitionReport:
    """
    Прогоняет кейсы через ``recognize_fn`` и считает качество распознавания.

    Главная метрика — доля точных совпадений ``recognized_text == reference_text``.
    Дополнительно считаются CER и WER. Пороги ``max_cer`` и ``min_exact_match``
    опциональны: если заданы и нарушены, попадают в ``failures``.

    Args:
        cases: список :class:`RecognitionCase`.
        recognize_fn: функция ``wav_path -> str`` (любой из ``recognize_*``).
        max_cer: максимально допустимый CER (``None`` — не проверять).
        min_exact_match: минимальная доля точных совпадений (``None`` — не проверять).
        verbose: печатать прогресс по кейсам.

    Returns:
        :class:`RecognitionReport`.
    """
    references: list[str] = []
    hypotheses: list[str] = []
    case_details: list[dict] = []
    exact_matches = 0

    for case in cases:
        hypothesis = recognize_fn(case.wav_path)
        reference = case.text
        is_exact = hypothesis.strip() == reference.strip()
        if is_exact:
            exact_matches += 1
        references.append(reference)
        hypotheses.append(hypothesis)
        case_details.append(
            {
                "wav": case.wav_path,
                "reference": reference,
                "hypothesis": hypothesis,
                "exact_match": is_exact,
            }
        )
        if verbose:
            mark = "OK " if is_exact else "ERR"
            print(f"  [{mark}] {os.path.basename(case.wav_path)}: '{hypothesis}'")

    n_cases = len(cases)
    exact_match_rate = (exact_matches / n_cases) if n_cases else 0.0
    cer_value = _cer(references, hypotheses) if n_cases else 0.0
    wer_value = _wer(references, hypotheses) if n_cases else 0.0

    failures: list[str] = []
    if max_cer is not None and cer_value > max_cer:
        failures.append(f"CER {cer_value:.4f} превышает порог {max_cer}")
    if min_exact_match is not None and exact_match_rate < min_exact_match:
        failures.append(
            f"Доля точных совпадений {exact_match_rate:.4f} ниже порога {min_exact_match}"
        )

    return RecognitionReport(
        n_cases=n_cases,
        exact_matches=exact_matches,
        exact_match_rate=exact_match_rate,
        cer=cer_value,
        wer=wer_value,
        max_cer=max_cer,
        min_exact_match=min_exact_match,
        cases=case_details,
        failures=failures,
    )


def save_recognition_report(report: RecognitionReport, path: str | None = None) -> str:
    """
    Сохраняет JSON-отчёт распознавания.

    Args:
        report: результат :func:`evaluate_recognition`.
        path: путь к JSON (по умолчанию ``artifacts/reports/recognition_report.json``).

    Returns:
        Путь к сохранённому файлу.
    """
    if path is None:
        path = str(resolver.artifacts / "reports" / DEFAULT_RECOGNITION_REPORT)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
    return path


__all__ = [
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_MAX_CER",
    "DEFAULT_MIN_EXACT_MATCH",
    "DEFAULT_N_MELS",
    "DEFAULT_RECOGNITION_REPORT",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TARGET",
    "DEFAULT_TIME_FRAMES",
    "DeviceRecognitionResult",
    "RecognitionCase",
    "RecognitionReport",
    "audio_to_mel",
    "evaluate_recognition",
    "fit_mel_length",
    "load_audio",
    "load_recognition_cases",
    "recognize_on_device",
    "recognize_rknn_simulator",
    "recognize_torch",
    "save_recognition_report",
    "wav_to_mel_input",
]
