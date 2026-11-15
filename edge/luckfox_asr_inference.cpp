// =========================================================================
// Минимальный нативный C++ раннер KAN-ASR для Luckfox Pico Plus (RV1103 NPU).
//
// Перенесено из KAN_ASR_Pipeline_Bundle/luckfox_asr_inference.cpp (легаси).
// Сборка — см. edge/CMakeLists.txt (кросс-компилятор arm-rockchip830-...).
// =========================================================================
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include "rknn_api.h"

// Словарь должен строго совпадать с Python-скриптом обучения (kan_asr.data.vocab)
static const std::vector<std::string> VOCAB = {
    "<blank>", " ",
    "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й",
    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
    "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"
};

// Greedy CTC декодер для выхода NPU [1, Vocab, 1, Time_Out]
std::string decode_ctc_greedy(const float* logits, int vocab_size, int time_steps) {
    std::string result;
    int prev_idx = -1;

    for (int t = 0; t < time_steps; ++t) {
        int best_idx = 0;
        float max_val = -1e9f;

        // NCHW layout: [0, c, 0, t] -> индекс c * time_steps + t
        for (int c = 0; c < vocab_size; ++c) {
            float val = logits[c * time_steps + t];
            if (val > max_val) {
                max_val = val;
                best_idx = c;
            }
        }

        if (best_idx != 0 && best_idx != prev_idx) {
            if (best_idx < static_cast<int>(VOCAB.size())) {
                result += VOCAB[best_idx];
            }
        }
        prev_idx = best_idx;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./luckfox_kan_asr <model.rknn>" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];
    rknn_context ctx = 0;

    // 1. Инициализация RKNN контекста на RV1103
    int ret = rknn_init(&ctx, (void*)model_path, 0, 0, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return -1;
    }

    // 2. Подготовка входного тензора [1, 40, 1, 200] (2 секунды Mel-спектрограммы)
    const int n_mels = 40;
    const int time_in = 200;
    std::vector<float> mel_input(n_mels * time_in, 0.0f);
    // TODO: Заполнить mel_input данными с ALSA / микрофона после FFT + Mel-фильтров

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = mel_input.size() * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NCHW;
    inputs[0].buf = mel_input.data();

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 3. Запуск инференса на NPU (0.5 TOPS)
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 4. Получение выходных логитов [1, 35, 1, 50]
    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // NPU сам деквантует INT8 -> float32
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    const float* logits = static_cast<const float*>(outputs[0].buf);
    int vocab_size = VOCAB.size(); // 35
    int time_out = time_in / 4;    // 50 (два слоя stride=2)

    std::string text = decode_ctc_greedy(logits, vocab_size, time_out);
    std::cout << "[NPU Recognized]: " << text << std::endl;

    // 5. Освобождение памяти (критично для 64 МБ RAM на RV1103)
    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    return 0;
}
