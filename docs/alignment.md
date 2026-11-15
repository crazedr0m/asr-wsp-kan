# Выравнивание представлений (CKA + Cosine Alignment) для TinyKAN_ASR

> Перенесено из `Whisper_CKA_Alignment/README_ALIGNMENT.md` (легаси-каталог удалён).
> Код: [`src/kan_asr/training/alignment.py`](../src/kan_asr/training/alignment.py),
> учитель: [`src/kan_asr/models/whisper_teacher.py`](../src/kan_asr/models/whisper_teacher.py).

Этот модуль дополняет инициализацию весов из `Whisper-tiny` (Weight-Space Programming)
и решает две задачи при дообучении на реальном аудио:

1. **Синхронизация временного шага (20 мс -> 40 мс):**
   Энкодер `Whisper-tiny` сжимает спектрограмму в 2 раза (шаг 20 мс на фрейм), а `TinyKAN_ASR`
   (для экономии RAM на **Luckfox Pico Plus / RV1103**) сжимает её в 4 раза (шаг 40 мс).
   Функция `_align_time` автоматически объединяет пары фреймов учителя перед расчётом функции потерь.
2. **Геометрическое и пофреймовое выравнивание:**
   - **Linear CKA** сравнивает матрицы сходства между фреймами, не требуя совпадения числа
     каналов (`128` против `384`).
   - **Проекторы `proj_mid` и `proj_top`** используются **только во время обучения** на GPU
     и не попадают в финальный `.onnx` / `.rknn` файл для Luckfox.
   - **WSP Anchor** удерживает веса `FastKAN` рядом со стартовыми значениями, предотвращая
     появление больших выбросов, которые портят точность при квантовании в `INT8`.

## Включение в обучении

CKA-выравнивание включается флагом `--align` у [`scripts/train.py`](../scripts/train.py):

```bash
python scripts/train.py \
    --wsp-checkpoint artifacts/checkpoints/tiny_kan_wsp_initialized.pt \
    --align --align-weight 0.3 \
    --whisper-state whisper_tiny.pt \
    --anchor-weight 1e-3
```

## Рекомендации по GPU

- **RTX 3060 (12 ГБ VRAM):** основной цикл обучения с учителем `WhisperTinyTeacher` в режиме
  `bfloat16` (`lambda_align = 0.3` на первых 5 эпохах, затем снижение до `0.05`).
- **GTX 1050 Ti (4 ГБ VRAM):** работает строго в `FP32`; удобно использовать для валидации WER
  или дообучения только `WeightSpaceHyperAdapter` без учителя в памяти.
