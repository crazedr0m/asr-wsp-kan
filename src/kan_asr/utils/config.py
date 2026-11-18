"""
Загрузчик YAML-конфигураций проекта.

Конфиги в ``configs/*.yaml`` описывают параметры архитектуры, WSP, обучения
и бота. Этот модуль даёт единый способ их прочитать и получить значение по
точечному пути (``"train.batch_size"``).

PyYAML импортируется лениво: если пакет не установлен, выбрасывается понятная
ошибка с подсказкой установить ``pyyaml``.
"""

import os
from typing import Any

from .paths import resolver

__all__ = ["CONFIG_FILES", "get_config_value", "load_config"]

# Имена доступных конфигов (без расширения)
CONFIG_FILES = ("model", "wsp", "train", "bot")


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as e:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Для загрузки YAML-конфигов требуется PyYAML. Установите: pip install pyyaml"
        ) from e
    return yaml


def load_config(name: str, config_dir: str = None) -> dict:
    """
    Загружает конфиг ``configs/<name>.yaml`` и возвращает словарь.

    Параметры
    ---------
    name : str
        Имя конфига без расширения (``model``, ``wsp``, ``train``, ``bot``).
    config_dir : str, optional
        Каталог с конфигами (по умолчанию ``resolver.configs``).

    Возвращает пустой словарь, если файл не найден.
    """
    if config_dir is None:
        config_dir = str(resolver.configs)

    path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(path):
        return {}

    yaml = _import_yaml()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_config_value(name: str, dotted_key: str, default: Any = None) -> Any:
    """
    Возвращает значение по точечному пути из конфига.

    Пример::

        get_config_value("train", "train.batch_size", 16)  # -> 16
        get_config_value("wsp", "wsp.project.ridge_alpha")  # -> 0.01
    """
    data = load_config(name)
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
