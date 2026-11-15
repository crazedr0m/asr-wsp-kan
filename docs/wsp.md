# Weight-Space Programming: перенос знаний из Whisper в TinyKAN_ASR

Этот модуль позволяет инициализировать компактную модель `TinyKAN_ASR`
(для Luckfox Pico Plus RV1103) знаниями из предобученного **Whisper-tiny**
без установки библиотеки `transformers` и без многочасового обучения с нуля.

> Источник: перенесено из `README_WSP.md`. Код: [`src/kan_asr/wsp/`](../src/kan_asr/wsp/).

## Как это работает математически

1. **Спектральная SVD-проекция свёрток** ([`wsp/extract.py`](../src/kan_asr/wsp/extract.py)):
   * Берёт матрицу входных фильтров `encoder.conv1.weight` `[384, 80, 3]` из Whisper.
   * Выполняет билинейную интерполяцию частотной оси (`80 -> 40` мел-бинов)
     и временной оси (`3 -> 5`).
   * Через сингулярное разложение (SVD) отбирает 64 главных частотно-временных
     детектора формант и записывает их в первый блок `DepthwiseSeparableConv` ученика.

2. **Функциональная проекция FFN -> FastKAN** ([`wsp/project.py`](../src/kan_asr/wsp/project.py)):
   * Блоки `fc1 + GELU + fc2` энкодера Whisper сжимаются с размерности `384`
     до `128` через ортогональные SVD-проекторы.
   * На синтетическом потоке активаций вычисляется отклик учителя.
   * Методом наименьших квадратов с L2-регуляризацией (`torch.linalg.solve`)
     находятся оптимальные коэффициенты при Гаусссианах `FastKAN`, аппроксимирующие
     нелинейность Whisper с минимальной дисперсией весов (что критично для INT8
     квантования на NPU RV1103).

3. **Weight-Space HyperAdapter** ([`wsp/hyper_adapter.py`](../src/kan_asr/wsp/hyper_adapter.py)):
   * Компактная гиперсеть (~10K параметров), которая генерирует низкоранговые
     поправки `Delta-W` для весов `FastKAN`.
   * Используется при дообучении на небольших батчах из Telegram-бота, защищая
     модель от катастрофического забывания.

## Быстрый старт на RTX 3060 / GTX 1050 Ti

```bash
# 0. Подготовить state_dict Whisper-tiny (см. ниже) -> whisper_tiny.pt

# 1. Инициализация студента из весов Whisper (WSP)
python scripts/wsp_init.py --whisper-state whisper_tiny.pt

# 2. Дообучение (CTC + опционально CKA-выравнивание)
python scripts/train.py --wsp-checkpoint artifacts/checkpoints/tiny_kan_wsp_initialized.pt
# с CKA-выравниванием:
# python scripts/train.py --wsp-checkpoint ... --align --whisper-state whisper_tiny.pt

# 3. Экспорт ONNX -> RKNN INT8
python scripts/export_onnx.py
python scripts/convert_rknn.py
```

### Подготовка весов Whisper-tiny

Скрипт [`scripts/wsp_init.py`](../scripts/wsp_init.py) **не скачивает** веса
автоматически — путь к `state_dict` передаётся явно через обязательный флаг
`--whisper-state`. Получить файл можно, например, так:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import torch
p = hf_hub_download("openai/whisper-tiny", "pytorch_model.bin")
sd = torch.load(p, map_location="cpu")
torch.save(sd, "whisper_tiny.pt")
PY
```

Скрипт выполнит анализ и проекцию за 1–2 секунды и сохранит готовый чекпоинт
`artifacts/checkpoints/tiny_kan_wsp_initialized.pt` (путь переопределяется флагом
`--output`). Далее в цикле обучения эти веса подгружаются через
`--wsp-checkpoint`, после чего дообучается выходная голова `head` под русский
алфавит.

## Конфигурация

Гиперпараметры WSP вынесены в [`configs/wsp.yaml`](../configs/wsp.yaml):
`extract` (SVD), `project` (аппроксимация), `initialize`, `hyper_adapter`, `teacher`.

## Связанные документы

* [Архитектура проекта](architecture.md)
* [План миграции структуры](../plans/project_structure_migration.md)
