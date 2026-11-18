"""
Верификация RKNN-модели: симулятор NPU на ПК и замер на реальной плате.

Позволяет без физической платы Luckfox Pico Plus:

- поднять эмулятор NPU (``rknn.init_runtime(target=None)``);
- прогнать вход через INT8-модель (``rknn.inference``);
- сравнить выход с float32 ONNX Runtime (косинусное сходство, max abs diff);
- проверить размер ``.rknn`` (лимит 4 МБ) и оценку памяти (``eval_memory``);
- опционально запустить ``rknn.accuracy_analysis``.

А при подключённой по USB/ADB плате (``target='rv1103'``):

- проверить наличие устройства (``adb devices``) и ``rknn_server`` на плате;
- при необходимости поднять ``rknn_server`` (``start-stop-daemon --background``);
- прогреть NPU и замерить latency/FPS/RTF по N итерациям;
- сохранить JSON-отчёт бенчмарка.

Модуль требует установленного ``rknn-toolkit2`` (импортируется лениво), поэтому
его можно безопасно импортировать в окружении без NPU-тулчейна.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

from ..utils.paths import resolver

# Лимиты платы Luckfox Pico Plus (RV1103)
DEFAULT_MAX_SIZE_MB = 4.0
DEFAULT_MAX_MEMORY_MB = 64.0
DEFAULT_MIN_COSINE = 0.95
DEFAULT_INPUT_SHAPE = (1, 40, 1, 200)

# Параметры бенчмарка на реальной плате
DEFAULT_TARGET = "rv1103"
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 20
DEFAULT_AUDIO_SECONDS = 2.0
DEFAULT_MAX_LATENCY_MS = 200.0
DEFAULT_MIN_FPS = 5.0
DEFAULT_RKNN_SERVER = "/oem/usr/bin/rknn_server"
DEFAULT_PERF_REPORT = "perf_report.json"


@dataclass
class PerformanceResult:
    """Результат замера производительности RKNN-модели на реальной плате."""

    target: str
    device_id: str | None
    iterations: int
    warmup: int
    latency_ms_mean: float
    latency_ms_min: float
    latency_ms_max: float
    latency_ms_p50: float
    latency_ms_p90: float
    fps: float
    rtf: float
    audio_seconds: float
    toolkit_version: str | None = None
    peak_memory_mb: float | None = None
    max_latency_ms: float | None = None
    min_fps: float | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Все пороговые проверки производительности пройдены."""
        return not self.failures

    def to_dict(self) -> dict:
        """Сериализуемое представление (для JSON-отчёта)."""
        return {
            "target": self.target,
            "device_id": self.device_id,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "latency_ms": {
                "mean": self.latency_ms_mean,
                "min": self.latency_ms_min,
                "max": self.latency_ms_max,
                "p50": self.latency_ms_p50,
                "p90": self.latency_ms_p90,
            },
            "fps": self.fps,
            "rtf": self.rtf,
            "audio_seconds": self.audio_seconds,
            "toolkit_version": self.toolkit_version,
            "peak_memory_mb": self.peak_memory_mb,
            "thresholds": {
                "max_latency_ms": self.max_latency_ms,
                "min_fps": self.min_fps,
            },
            "passed": self.passed,
            "failures": list(self.failures),
        }

    def summary(self) -> str:
        """Человекочитаемая сводка."""
        lines = [
            f"Плата:           {self.target} (device_id={self.device_id})",
            f"Итераций:        {self.iterations} (warmup={self.warmup})",
            (
                f"Latency, мс:     mean={self.latency_ms_mean:.2f} "
                f"min={self.latency_ms_min:.2f} max={self.latency_ms_max:.2f} "
                f"p50={self.latency_ms_p50:.2f} p90={self.latency_ms_p90:.2f}"
            ),
            f"FPS:             {self.fps:.2f}",
            f"RTF:             {self.rtf:.4f} (окно {self.audio_seconds:.2f} с)",
        ]
        if self.peak_memory_mb is not None:
            lines.append(f"Пиковая память:  {self.peak_memory_mb:.2f} МБ")
        if self.toolkit_version:
            lines.append(f"rknn-toolkit2:   {self.toolkit_version}")
        if self.failures:
            lines.append("Проваленные проверки производительности:")
            lines.extend(f"  - {item}" for item in self.failures)
        else:
            lines.append("Итог: ПРОИЗВОДИТЕЛЬНОСТЬ В НОРМЕ")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Результат верификации RKNN-модели в симуляторе NPU."""

    onnx_model: str
    rknn_model: str
    rknn_size_mb: float
    size_ok: bool
    cosine_similarity: float
    max_abs_diff: float
    mean_abs_diff: float
    output_shape: tuple[int, ...]
    memory_report: str | None = None
    peak_memory_mb: float | None = None
    memory_ok: bool | None = None
    accuracy_analysis_dir: str | None = None
    performance: PerformanceResult | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Все обязательные проверки пройдены."""
        return not self.failures

    def summary(self) -> str:
        """Человекочитаемая сводка."""
        lines = [
            f"ONNX:            {self.onnx_model}",
            f"RKNN:            {self.rknn_model}",
            f"Размер .rknn:    {self.rknn_size_mb:.3f} МБ (ok={self.size_ok})",
            f"Cosine similarity: {self.cosine_similarity:.6f}",
            f"Max abs diff:    {self.max_abs_diff:.6f}",
            f"Mean abs diff:   {self.mean_abs_diff:.6f}",
            f"Форма выхода:    {self.output_shape}",
        ]
        if self.peak_memory_mb is not None:
            lines.append(f"Пиковая память:  {self.peak_memory_mb:.2f} МБ (ok={self.memory_ok})")
        if self.accuracy_analysis_dir:
            lines.append(f"Accuracy analysis: {self.accuracy_analysis_dir}")
        if self.performance is not None:
            lines.append("")
            lines.append("--- Замер на реальной плате ---")
            lines.append(self.performance.summary())
        if self.failures:
            lines.append("Проваленные проверки:")
            lines.extend(f"  - {item}" for item in self.failures)
        else:
            lines.append("Итог: ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        return "\n".join(lines)


def _import_rknn():
    """Ленивый импорт ``rknn.api.RKNN`` с понятной ошибкой."""
    try:
        from rknn.api import RKNN
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Не найден пакет 'rknn-toolkit2'. Установите его отдельно:\n"
            '    pip install -e ".[rknn]"\n'
            "или\n"
            "    pip install -r requirements-optional.txt"
        ) from exc
    return RKNN


def _toolkit_version() -> str | None:
    """Версия установленного ``rknn-toolkit2`` (или ``None``)."""
    try:
        from importlib.metadata import version

        return version("rknn-toolkit2")
    except Exception:  # pragma: no cover - зависит от окружения
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство двух массивов (по сплющенным значениям)."""
    flat_a = np.asarray(a, dtype=np.float64).ravel()
    flat_b = np.asarray(b, dtype=np.float64).ravel()
    if flat_a.size != flat_b.size:
        raise ValueError(f"Размеры выходов не совпадают: {flat_a.size} != {flat_b.size}")
    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    if norm_a == 0.0 or norm_b == 0.0:
        # Оба нулевых -> идеальное совпадение; иначе сходство не определено
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(flat_a, flat_b) / (norm_a * norm_b))


def _align_output(rknn_out: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Приводит выход эмулятора к форме эталонного выхода.

    RKNN может вернуть тензор в другом layout (например, NHWC вместо NCHW).
    Если общее число элементов совпадает, выход переформируется.
    """
    rknn_out = np.asarray(rknn_out)
    if rknn_out.shape == reference.shape:
        return rknn_out
    if rknn_out.size == reference.size:
        return rknn_out.reshape(reference.shape)
    raise ValueError(
        f"Форма выхода эмулятора несовместима с ONNX: {rknn_out.shape} vs {reference.shape}"
    )


def _run_onnx(onnx_model: str, sample_input: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Прогон входа через ONNX Runtime (float32) на CPU."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_meta.name: sample_input})
    reference = np.asarray(outputs[0])
    return reference, tuple(reference.shape)


_MEMORY_UNITS = {"KB": 1.0 / 1024.0, "MB": 1.0, "GB": 1024.0}


def _parse_peak_memory_mb(report: str | dict | None) -> float | None:
    """
    Извлекает пиковую память (МБ) из отчёта ``eval_memory``.

    На реальной плате ``eval_memory`` возвращает ``dict`` (например
    ``{"Weight Memory": 269.75, ...}`` в KiB), в симуляторе — строку.
    Сначала ищет ключи/строки про Total/Peak/Memory, затем — любые значения
    с единицами. Возвращает ``None``, если распарсить не удалось.
    """
    if not report:
        return None

    # На устройстве eval_memory возвращает dict со значениями в байтах/KiB.
    if isinstance(report, dict):
        total = None
        for key, value in report.items():
            if not isinstance(value, (int, float)):
                continue
            lowered = str(key).lower()
            if "total" in lowered:
                total = float(value)
                break
        if total is None:
            numeric = [float(v) for v in report.values() if isinstance(v, (int, float))]
            if not numeric:
                return None
            total = max(numeric)
        # Значения в байтах: переводим в МБ.
        return total / (1024.0 * 1024.0)

    pattern = re.compile(r"([\d.]+)\s*(KB|MB|GB)", re.IGNORECASE)
    prioritized: list[float] = []
    fallback: list[float] = []

    for line in report.splitlines():
        matches = pattern.findall(line)
        if not matches:
            continue
        values = [float(value) * _MEMORY_UNITS[unit.upper()] for value, unit in matches]
        lowered = line.lower()
        if any(key in lowered for key in ("total", "peak", "memory", "max")):
            prioritized.extend(values)
        else:
            fallback.extend(values)

    candidates = prioritized or fallback
    return max(candidates) if candidates else None


def _default_sample_input(input_shape: Sequence[int]) -> np.ndarray:
    """Детерминированный вход float32 для сравнения float32 и INT8."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(tuple(input_shape)).astype(np.float32)


def _default_device_input(input_shape: Sequence[int]) -> np.ndarray:
    """
    Детерминированный вход uint8 для замера на реальной плате.

    Квантованная модель на устройстве принимает UINT8-тензор, поэтому
    для бенчмарка достаточно случайных значений в диапазоне [0, 255].
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=tuple(input_shape), dtype=np.uint8)


def _to_uint8_input(sample_input: np.ndarray) -> np.ndarray:
    """Приводит вход к uint8 (требование квантованной модели на устройстве)."""
    arr = np.asarray(sample_input)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def _latency_stats(samples_ms: Sequence[float]) -> dict[str, float]:
    """Статистика latency (мс): mean/min/max/p50/p90."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Нет замеров latency")
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


# --- ADB-хелперы -----------------------------------------------------------


def _adb_binary() -> str:
    """Путь к ``adb`` или понятная ошибка."""
    adb = shutil.which("adb")
    if not adb:
        raise RuntimeError(
            "Не найден 'adb'. Установите Android platform-tools и добавьте "
            "adb в PATH (или используйте adb из rknn-toolkit2: "
            "rknn/3rdparty/platform-tools/adb)."
        )
    return adb


def _adb_devices(adb: str | None = None) -> list[str]:
    """
    Список серийных номеров устройств в состоянии ``device``.

    Парсит вывод ``adb devices`` (первая строка — заголовок).
    """
    adb = adb or _adb_binary()
    try:
        output = subprocess.check_output([adb, "devices"], text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(f"Не удалось выполнить 'adb devices': {exc}") from exc

    devices: list[str] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def _ensure_device(device_id: str | None = None, adb: str | None = None) -> str:
    """
    Возвращает серийный номер платы.

    Если ``device_id`` задан — проверяет, что устройство подключено.
    Если не задан — требует ровно одно подключённое устройство.
    """
    adb = adb or _adb_binary()
    devices = _adb_devices(adb)

    if device_id is not None:
        if device_id not in devices:
            raise RuntimeError(
                f"Устройство '{device_id}' не найдено среди подключённых: "
                f"{devices or 'нет устройств'}. Проверьте USB/ADB "
                "(adb devices)."
            )
        return device_id

    if not devices:
        raise RuntimeError(
            "Не найдено ни одного устройства по ADB. Подключите плату "
            "Luckfox Pico Plus по USB и проверьте 'adb devices'."
        )
    if len(devices) > 1:
        raise RuntimeError(f"Подключено несколько устройств: {devices}. Укажите device_id явно.")
    return devices[0]


def _check_rknn_server(
    device_id: str, adb: str | None = None, server_path: str = DEFAULT_RKNN_SERVER
) -> bool:
    """Проверяет, запущен ли ``rknn_server`` на плате."""
    adb = adb or _adb_binary()
    try:
        output = subprocess.check_output(
            [adb, "-s", device_id, "shell", "ps | grep rknn_server | grep -v grep"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return bool(output.strip())


def _start_rknn_server(
    device_id: str,
    adb: str | None = None,
    server_path: str = DEFAULT_RKNN_SERVER,
    timeout_s: float = 5.0,
) -> None:
    """
    Запускает ``rknn_server`` на плате в фоне.

    Использует ``start-stop-daemon --background`` (как это делает adbd),
    чтобы процесс пережил завершение adb-сессии. Сервер создаёт
    abstract-сокет ``@transfer_proxy``, который тулчейн пробрасывает
    через ``adb forward localabstract:transfer_proxy``.
    """
    adb = adb or _adb_binary()
    cmd = [
        adb,
        "-s",
        device_id,
        "shell",
        f"start-stop-daemon --start --quiet --background --exec {server_path}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(f"Не удалось запустить rknn_server на плате: {exc}") from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _check_rknn_server(device_id, adb, server_path):
            return
        time.sleep(0.3)
    raise RuntimeError(
        f"rknn_server не запустился на плате. Проверьте наличие {server_path} и права (adb shell)."
    )


# --- Бенчмарк на реальной плате -------------------------------------------


def benchmark_on_device(
    rknn_model: str | None = None,
    target: str = DEFAULT_TARGET,
    device_id: str | None = None,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    sample_input: np.ndarray | None = None,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    audio_seconds: float = DEFAULT_AUDIO_SECONDS,
    max_latency_ms: float | None = None,
    min_fps: float | None = None,
    data_format: str = "nhwc",
    auto_start_server: bool = True,
    verbose: bool = True,
) -> PerformanceResult:
    """
    Замеряет latency/FPS/RTF RKNN-модели на реальной плате (RV1103 NPU).

    Args:
        rknn_model: путь к ``.rknn`` (по умолчанию
            ``artifacts/rknn/tiny_kan_asr_rv1103.rknn``).
        target: целевая платформа (``rv1103``).
        device_id: серийный номер платы (по умолчанию — единственное устройство).
        input_shape: форма входа, если ``sample_input`` не задан.
        sample_input: вход; приводится к uint8 (требование квантованной модели).
        warmup: число прогревочных прогонов (не учитываются в статистике).
        iterations: число замеряемых прогонов.
        audio_seconds: длительность аудиоокна для расчёта RTF.
        max_latency_ms: порог средней latency (мс); ``None`` — не проверять.
        min_fps: минимальный FPS; ``None`` — не проверять.
        data_format: формат входа для ``rknn.inference`` (``nhwc``/``nchw``).
        auto_start_server: поднять ``rknn_server``, если он не запущен.
        verbose: печатать прогресс.

    Returns:
        :class:`PerformanceResult` с метриками и списком проваленных проверок.
    """
    RKNN = _import_rknn()

    if rknn_model is None:
        rknn_model = str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")
    if not os.path.exists(rknn_model):
        raise FileNotFoundError(f"RKNN-модель не найдена: {rknn_model}")

    adb = _adb_binary()
    device_id = _ensure_device(device_id, adb)

    if auto_start_server and not _check_rknn_server(device_id, adb):
        if verbose:
            print("--> rknn_server не запущен, поднимаю на плате...")
        _start_rknn_server(device_id, adb)
    elif not _check_rknn_server(device_id, adb):
        raise RuntimeError(
            "rknn_server не запущен на плате. Запустите его вручную:\n"
            f"    adb -s {device_id} shell "
            f"start-stop-daemon --start --quiet --background "
            f"--exec {DEFAULT_RKNN_SERVER}"
        )

    if sample_input is None:
        sample_input = _default_device_input(input_shape)
    sample_input = _to_uint8_input(sample_input)

    rknn = RKNN(verbose=verbose)
    samples_ms: list[float] = []
    memory_report: str | None = None
    try:
        ret = rknn.load_rknn(rknn_model)
        if ret != 0:
            raise RuntimeError("Ошибка загрузки .rknn")

        if verbose:
            print(f"--> init_runtime(target={target}, device_id={device_id})...")
        # eval_mem=True нужен, чтобы rknn.eval_memory() работал на устройстве
        ret = rknn.init_runtime(target=target, device_id=device_id, eval_mem=True)
        if ret != 0:
            raise RuntimeError(f"Ошибка init_runtime(target={target}, device_id={device_id})")

        if verbose:
            print(f"--> Прогрев NPU ({warmup} прогонов)...")
        for _ in range(max(0, warmup)):
            rknn.inference(inputs=[sample_input], data_format=[data_format])

        if verbose:
            print(f"--> Замер latency ({iterations} прогонов)...")
        for _ in range(max(1, iterations)):
            start = time.perf_counter()
            rknn.inference(inputs=[sample_input], data_format=[data_format])
            samples_ms.append((time.perf_counter() - start) * 1000.0)

        # eval_memory работает только на реальном устройстве (не в симуляторе)
        try:
            memory_report = rknn.eval_memory()
        except Exception as exc:  # pragma: no cover - зависит от версии toolkit
            memory_report = None
            if verbose:
                print(f"[WARN] eval_memory на плате недоступен: {exc}")
    finally:
        rknn.release()

    stats = _latency_stats(samples_ms)
    fps = 1000.0 / stats["mean"] if stats["mean"] > 0 else 0.0
    rtf = (stats["mean"] / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0
    peak_memory_mb = _parse_peak_memory_mb(memory_report)

    failures: list[str] = []
    if max_latency_ms is not None and stats["mean"] > max_latency_ms:
        failures.append(
            f"Средняя latency {stats['mean']:.2f} мс превышает порог {max_latency_ms} мс"
        )
    if min_fps is not None and fps < min_fps:
        failures.append(f"FPS {fps:.2f} ниже порога {min_fps}")

    result = PerformanceResult(
        target=target,
        device_id=device_id,
        iterations=len(samples_ms),
        warmup=warmup,
        latency_ms_mean=stats["mean"],
        latency_ms_min=stats["min"],
        latency_ms_max=stats["max"],
        latency_ms_p50=stats["p50"],
        latency_ms_p90=stats["p90"],
        fps=fps,
        rtf=rtf,
        audio_seconds=audio_seconds,
        toolkit_version=_toolkit_version(),
        peak_memory_mb=peak_memory_mb,
        max_latency_ms=max_latency_ms,
        min_fps=min_fps,
        failures=failures,
    )

    if verbose:
        print(f"--> Плата: latency mean={stats['mean']:.2f} мс, FPS={fps:.2f}, RTF={rtf:.4f}")
    return result


def save_perf_report(result: PerformanceResult, path: str | None = None) -> str:
    """
    Сохраняет JSON-отчёт бенчмарка.

    Args:
        result: результат :func:`benchmark_on_device`.
        path: путь к JSON (по умолчанию ``artifacts/rknn/perf_report.json``).

    Returns:
        Путь к сохранённому файлу.
    """
    if path is None:
        path = str(resolver.rknn / DEFAULT_PERF_REPORT)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    return path


def verify_rknn_simulator(
    onnx_model: str | None = None,
    rknn_model: str | None = None,
    calib_dataset: str | None = None,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    sample_input: np.ndarray | None = None,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
    run_accuracy_analysis: bool = False,
    accuracy_output_dir: str | None = None,
    rebuild: bool = False,
    data_format: str = "nchw",
    target: str | None = None,
    device_id: str | None = None,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    audio_seconds: float = DEFAULT_AUDIO_SECONDS,
    max_latency_ms: float | None = None,
    min_fps: float | None = None,
    perf_report_path: str | None = None,
    save_report: bool = True,
    auto_start_server: bool = True,
    verbose: bool = True,
) -> VerificationResult:
    """
    Проверяет RKNN-модель в симуляторе NPU и (опционально) на реальной плате.

    Порядок действий:
        1. Собирает ``.rknn`` из ONNX (или загружает готовый, если ``rebuild=False``).
        2. Опционально запускает ``accuracy_analysis``.
        3. Поднимает эмулятор NPU: ``init_runtime(target=None)``.
        4. Прогоняет один и тот же вход через ONNX Runtime и эмулятор.
        5. Сравнивает выходы, проверяет размер и память.
        6. Если задан ``target`` — замеряет latency/FPS/RTF на реальной плате
           (:func:`benchmark_on_device`) и сохраняет JSON-отчёт.

    Args:
        onnx_model: путь к ONNX (по умолчанию ``artifacts/onnx/tiny_kan_asr_rv1103.onnx``).
        rknn_model: путь к ``.rknn`` (по умолчанию ``artifacts/rknn/tiny_kan_asr_rv1103.rknn``).
        calib_dataset: файл со списком калибровочных ``.npy``
            (по умолчанию ``data/calib/calib_dataset.txt``).
        input_shape: форма входа, если ``sample_input`` не задан.
        sample_input: готовый вход; по умолчанию детерминированный случайный.
        min_cosine: минимально допустимое косинусное сходство.
        max_size_mb: лимит размера ``.rknn`` в МБ.
        max_memory_mb: лимит пиковой памяти в МБ.
        run_accuracy_analysis: запустить ``rknn.accuracy_analysis``.
        accuracy_output_dir: каталог отчётов accuracy analysis.
        rebuild: пересобрать ``.rknn`` из ONNX, даже если файл уже существует.
        data_format: формат входных данных для ``rknn.inference`` (``nchw``/``nhwc``).
        target: целевая платформа для замера на железе (``None`` — только симулятор).
        device_id: серийный номер платы (по умолчанию — единственное устройство).
        warmup: число прогревочных прогонов на плате.
        iterations: число замеряемых прогонов на плате.
        audio_seconds: длительность аудиоокна для расчёта RTF.
        max_latency_ms: порог средней latency на плате (мс).
        min_fps: минимальный FPS на плате.
        perf_report_path: путь к JSON-отчёту бенчмарка
            (по умолчанию ``artifacts/rknn/perf_report.json``).
        save_report: сохранять JSON-отчёт бенчмарка.
        auto_start_server: поднять ``rknn_server`` на плате, если он не запущен.
        verbose: печатать прогресс.

    Returns:
        :class:`VerificationResult` с метриками и списком проваленных проверок.
    """
    RKNN = _import_rknn()

    if onnx_model is None:
        onnx_model = str(resolver.onnx / "tiny_kan_asr_rv1103.onnx")
    if rknn_model is None:
        rknn_model = str(resolver.rknn / "tiny_kan_asr_rv1103.rknn")
    if calib_dataset is None:
        calib_dataset = str(resolver.calib / "calib_dataset.txt")
    if accuracy_output_dir is None:
        accuracy_output_dir = str(resolver.artifacts / "rknn_accuracy")

    if not os.path.exists(onnx_model):
        raise FileNotFoundError(f"ONNX-модель не найдена: {onnx_model}")

    if sample_input is None:
        sample_input = _default_sample_input(input_shape)
    sample_input = np.asarray(sample_input, dtype=np.float32)

    failures: list[str] = []

    # Эталонный выход float32 (ONNX Runtime)
    if verbose:
        print("--> Прогон ONNX Runtime (float32)...")
    reference, output_shape = _run_onnx(onnx_model, sample_input)

    rknn = RKNN(verbose=verbose)
    try:
        # Симулятор NPU (target=None) работает только с моделью, собранной в
        # текущем процессе: ``load_rknn`` + ``init_runtime(target=None)``
        # тулчейн отвергает ("please set 'target' first"). Поэтому для
        # симулятора всегда собираем INT8 из ONNX (``rebuild`` оставлен для
        # совместимости API и явного запроса пересборки).
        if not os.path.exists(calib_dataset):
            raise FileNotFoundError(
                "Калибровочный датасет не найден: "
                f"{calib_dataset}. Сначала обучите модель "
                "(scripts/train.py) или укажите --calib-file."
            )
        from .convert_rknn import build_rknn_model

        if verbose:
            print("--> Сборка RKNN INT8 из ONNX...")
        build_rknn_model(rknn, onnx_model, rknn_model, calib_dataset)

        if run_accuracy_analysis:
            if verbose:
                print("--> Accuracy analysis (float32 vs INT8)...")
            os.makedirs(accuracy_output_dir, exist_ok=True)
            ret = rknn.accuracy_analysis(inputs=[sample_input], output_dir=accuracy_output_dir)
            if ret != 0:
                failures.append("accuracy_analysis завершился с ошибкой")

        if verbose:
            print("--> Инициализация симулятора NPU (target=None)...")
        ret = rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError("Ошибка init_runtime(target=None)")

        if verbose:
            print("--> Инференс в эмуляторе NPU (INT8)...")
        outputs = rknn.inference(inputs=[sample_input], data_format=[data_format])
        if not outputs:
            raise RuntimeError("Эмулятор NPU не вернул выходов")
        rknn_out = _align_output(outputs[0], reference)

        # eval_memory не поддерживается в симуляторе (target=None) — там
        # память оценивается только на реальной плате (см. benchmark_on_device).
        memory_report = None
        if target is not None:
            try:
                memory_report = rknn.eval_memory()
            except Exception as exc:  # pragma: no cover - зависит от версии toolkit
                if verbose:
                    print(f"[WARN] eval_memory недоступен: {exc}")
    finally:
        rknn.release()

    # Метрики точности
    cosine = _cosine_similarity(reference, rknn_out)
    diff = np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(rknn_out, dtype=np.float64))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0

    # Проверка размера
    rknn_size_mb = os.path.getsize(rknn_model) / (1024.0 * 1024.0)
    size_ok = rknn_size_mb <= max_size_mb
    if not size_ok:
        failures.append(f"Размер .rknn {rknn_size_mb:.3f} МБ превышает лимит {max_size_mb} МБ")

    # Проверка точности
    if cosine < min_cosine:
        failures.append(f"Косинусное сходство {cosine:.6f} ниже порога {min_cosine}")

    # Проверка памяти
    peak_memory_mb = _parse_peak_memory_mb(memory_report)
    memory_ok: bool | None = None
    if peak_memory_mb is not None:
        memory_ok = peak_memory_mb <= max_memory_mb
        if not memory_ok:
            failures.append(
                f"Пиковая память {peak_memory_mb:.2f} МБ превышает лимит {max_memory_mb} МБ"
            )

    # Замер на реальной плате (опционально)
    performance: PerformanceResult | None = None
    if target is not None:
        if verbose:
            print()
            print(f"--> Замер на реальной плате (target={target})...")
        performance = benchmark_on_device(
            rknn_model=rknn_model,
            target=target,
            device_id=device_id,
            input_shape=input_shape,
            warmup=warmup,
            iterations=iterations,
            audio_seconds=audio_seconds,
            max_latency_ms=max_latency_ms,
            min_fps=min_fps,
            auto_start_server=auto_start_server,
            verbose=verbose,
        )
        failures.extend(performance.failures)
        if save_report:
            saved = save_perf_report(performance, perf_report_path)
            if verbose:
                print(f"--> Отчёт бенчмарка сохранён: {saved}")

    result = VerificationResult(
        onnx_model=onnx_model,
        rknn_model=rknn_model,
        rknn_size_mb=rknn_size_mb,
        size_ok=size_ok,
        cosine_similarity=cosine,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        output_shape=output_shape,
        memory_report=memory_report,
        peak_memory_mb=peak_memory_mb,
        memory_ok=memory_ok,
        accuracy_analysis_dir=accuracy_output_dir if run_accuracy_analysis else None,
        performance=performance,
        failures=failures,
    )

    if verbose:
        print(result.summary())
    return result


__all__ = [
    "DEFAULT_AUDIO_SECONDS",
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_ITERATIONS",
    "DEFAULT_MAX_LATENCY_MS",
    "DEFAULT_MAX_MEMORY_MB",
    "DEFAULT_MAX_SIZE_MB",
    "DEFAULT_MIN_COSINE",
    "DEFAULT_MIN_FPS",
    "DEFAULT_PERF_REPORT",
    "DEFAULT_RKNN_SERVER",
    "DEFAULT_TARGET",
    "DEFAULT_WARMUP",
    "PerformanceResult",
    "VerificationResult",
    "_adb_devices",
    "_check_rknn_server",
    "_ensure_device",
    "_latency_stats",
    "benchmark_on_device",
    "save_perf_report",
    "verify_rknn_simulator",
]
