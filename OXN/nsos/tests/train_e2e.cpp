#include "../include/jamba.h"
#include "../include/self_healer.h"
#include "../include/trainer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace nsos;

static std::vector<int> make_cyclic_dataset(int length, int vocab) {
    std::vector<int> tokens(length);
    for (int i = 0; i < length; ++i) {
        tokens[i] = i % vocab;
    }
    return tokens;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   NSOS - End-to-End Training Smoke     \n";
    std::cout << "========================================\n\n";

    const int VOCAB_SIZE = 64;
    const int D_MODEL = 64;
    const int N_LAYERS = 1;
#ifdef USE_CUDA
    const Device RUNTIME_DEVICE = Device::GPU;
    std::cout << "[Backend] CUDA build detected. This E2E now attempts the GPU path.\n\n";
#else
    const Device RUNTIME_DEVICE = Device::CPU;
    std::cout << "[Backend] CPU-only build.\n\n";
#endif

    std::cout << "[Config] vocab=" << VOCAB_SIZE
              << " d_model=" << D_MODEL
              << " layers=" << N_LAYERS << "\n\n";

    JambaModel model(N_LAYERS, D_MODEL, VOCAB_SIZE, RUNTIME_DEVICE);

    const int SEQ_LEN = 16;
    const int DS_LEN = 2048;
    auto dataset = make_cyclic_dataset(DS_LEN, VOCAB_SIZE);

    std::cout << "[Dataset] " << DS_LEN << " cyclic tokens\n";
    std::cout << "[Dataset] First 16 tokens: ";
    for (int i = 0; i < 16; ++i) std::cout << dataset[i] << " ";
    std::cout << "\n\n";

    const float LR = 3e-3f;
    const int EPOCHS = 40;
    const int BATCH = 4;
    const int MAX_STEPS = 120;

    Trainer trainer(&model, LR);
    trainer.weight_decay = 0.0f;
    trainer.max_grad_norm = 2.0f;
    trainer.min_learning_rate_scale = 0.2f;
    trainer.warmup_steps = 10;

    int last_step = 0;
    float last_loss = 9999.0f;
    const auto t0 = std::chrono::steady_clock::now();

    std::cout << "[Train] epochs=" << EPOCHS
              << " lr=" << LR
              << " batch=" << BATCH
              << " max_steps=" << MAX_STEPS << "\n";

    trainer.train_loop(
        dataset,
        EPOCHS,
        BATCH,
        SEQ_LEN,
        [&](int step, float loss) {
            last_step = step;
            last_loss = loss;
            if (step == 1 || step % 20 == 0 || loss < 0.5f) {
                const auto now = std::chrono::steady_clock::now();
                const float elapsed = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count()) /
                    1000.0f;
                std::cout << "  step=" << step
                          << " loss=" << loss
                          << " t=" << elapsed << "s\n";
            }
        },
        MAX_STEPS);

    const auto t1 = std::chrono::steady_clock::now();
    const float total_s = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) /
        1000.0f;

    std::cout << "[Train] done in " << total_s << "s"
              << " steps=" << last_step
              << " loss=" << last_loss << "\n";

    int correct = 0;
    int evaluated = 0;
    for (int start = 0; start + SEQ_LEN < 256; start += SEQ_LEN) {
        std::vector<int> prompt_window(dataset.begin() + start, dataset.begin() + start + SEQ_LEN);
        model.reset_session();
        Tensor logits = model.forward_ids(prompt_window, nullptr);
        Tensor host_logits = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
        const int vocab = host_logits.shape.back();
        const float* last_row = host_logits.data() + host_logits.size - vocab;
        const int pred = static_cast<int>(std::max_element(last_row, last_row + vocab) - last_row);
        const int target = dataset[start + SEQ_LEN];
        correct += (pred == target) ? 1 : 0;
        ++evaluated;
    }
    std::cout << "[Eval] next-token accuracy=" << correct << "/" << evaluated
              << " (" << (evaluated > 0 ? (100.0f * correct / evaluated) : 0.0f) << "%)\n\n";

    std::vector<int> prompt = {0, 1, 2, 3};
    D2FDecoder decoder(&model);
    const auto gen_t0 = std::chrono::steady_clock::now();
    std::vector<int> generated =
        decoder.generate(prompt, 16, nullptr, 0.8f, 0.9f, 32, VOCAB_SIZE - 1);
    const auto gen_t1 = std::chrono::steady_clock::now();
    const float gen_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(gen_t1 - gen_t0).count());

    std::cout << "[Generate] prompt: ";
    for (int t : prompt) std::cout << t << " ";
    std::cout << "\n[Generate] output: ";
    for (size_t i = prompt.size(); i < generated.size(); ++i) std::cout << generated[i] << " ";
    std::cout << "\n[Generate] latency=" << gen_ms << "ms\n\n";

    Context ctx;
    Tensor latent = model.forward_ids(prompt, &ctx);
    Tensor refined = model.reason(latent, 50);
    Tensor final_state = model.run_reasoning_loop(latent, 2);
    std::cout << "[Reasoning] latent_norm=" << latent.norm()
              << " refined_norm=" << refined.norm()
              << " final_norm=" << final_state.norm() << "\n\n";

    HealingConfig heal_cfg;
    heal_cfg.confidence_threshold = 0.01f;
    heal_cfg.entropy_threshold = 20.0f;
    heal_cfg.consistency_samples = 2;
    heal_cfg.max_retries = 2;
    heal_cfg.temperature_start = 0.8f;

    NeuralSelfHealer healer(heal_cfg);
    healer.set_logits_extractor([&](const std::vector<int>& ids) -> Tensor {
        Context c2;
        return model.forward_ids(ids, &c2);
    });
    healer.set_token_decoder([&](const std::vector<int>& p, int max_len, float temp,
                                 float top_p_arg, int top_k_arg, int eos, Context* c) {
        return decoder.generate(p, max_len, c, temp, top_p_arg, top_k_arg, eos);
    });

    HealingReport report;
    const bool ok = healer.verify(generated, report);
    std::cout << "[SelfHealer] " << (ok ? "approved" : "rejected")
              << " confidence=" << report.final_confidence
              << " entropy=" << report.final_entropy << "\n";

    std::cout << "\n========================================\n";
    std::cout << "  NSOS E2E PIPELINE COMPLETE\n";
    std::cout << "========================================\n";
    return 0;
}
