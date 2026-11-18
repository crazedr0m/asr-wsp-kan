"""
Единый резолвер путей к данным и артефактам.

Модуль заменяет жёстко закодированные относительные пути на параметризованные.
"""

from pathlib import Path


class PathResolver:
    """
    Резолвер путей для KAN-ASR проекта.

    Предоставляет единый интерфейс для доступа к:
    - data/synth, data/calib, data/samples, data/raw, data/real
    - artifacts/piper, artifacts/onnx, artifacts/rknn, artifacts/checkpoints
    - configs, docs, tests, scripts
    """

    def __init__(self, project_root: str = None):
        """
        project_root: корень проекта (по умолчанию текущая директория)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Базовые директории
        self.src = self.project_root / "src" / "kan_asr"
        self.data = self.project_root / "data"
        self.artifacts = self.project_root / "artifacts"
        self.configs = self.project_root / "configs"
        self.docs = self.project_root / "docs"
        self.tests = self.project_root / "tests"
        self.scripts = self.project_root / "scripts"
        self.archive = self.project_root / "archive"
        self.csrc = self.project_root / "csrc"
        self.edge = self.project_root / "edge"

        # Подкаталоги data
        self.synth = self.data / "synth"
        self.calib = self.data / "calib"
        self.samples = self.data / "samples"
        self.raw = self.data / "raw"
        self.real = self.data / "real"

        # Подкаталоги artifacts
        self.piper = self.artifacts / "piper"
        self.onnx = self.artifacts / "onnx"
        self.rknn = self.artifacts / "rknn"
        self.checkpoints = self.artifacts / "checkpoints"

    def get(self, path: str) -> Path:
        """
        Получить путь, расширяя относительный путь относительно project_root.

        Примеры:
            resolver.get("data/synth") -> /path/to/project/data/synth
            resolver.get("artifacts/piper/ru_RU-irina-medium.onnx") -> /path/to/project/artifacts/piper/ru_RU-irina-medium.onnx
        """
        return self.project_root / path

    def get_data_path(self, dataset: str) -> Path:
        """Получить путь к датасету."""
        if dataset == "synth":
            return self.synth
        elif dataset == "calib":
            return self.calib
        elif dataset == "samples":
            return self.samples
        elif dataset == "raw":
            return self.raw
        elif dataset == "real":
            return self.real
        else:
            raise ValueError(f"Неизвестный датасет: {dataset}")

    def get_artifact_path(self, artifact: str) -> Path:
        """Получить путь к артефакту."""
        if artifact == "piper":
            return self.piper
        elif artifact == "onnx":
            return self.onnx
        elif artifact == "rknn":
            return self.rknn
        elif artifact == "checkpoints":
            return self.checkpoints
        else:
            raise ValueError(f"Неизвестный артефакт: {artifact}")

    def get_config_path(self, config_name: str) -> Path:
        """Получить путь к конфигурации."""
        return self.configs / config_name

    def get_doc_path(self, doc_name: str) -> Path:
        """Получить путь к документации."""
        return self.docs / doc_name

    def get_test_path(self, test_name: str) -> Path:
        """Получить путь к тесту."""
        return self.tests / test_name

    def get_script_path(self, script_name: str) -> Path:
        """Получить путь к скрипту."""
        return self.scripts / script_name

    def get_archive_path(self, archive_name: str) -> Path:
        """Получить путь к архиву."""
        return self.archive / archive_name

    def get_csrc_path(self, filename: str) -> Path:
        """Получить путь к CUDA-коду."""
        return self.csrc / filename

    def get_edge_path(self, filename: str) -> Path:
        """Получить путь к C++ раннеру."""
        return self.edge / filename


# Глобальный инстанс резолвера
resolver = PathResolver()
