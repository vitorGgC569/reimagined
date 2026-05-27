// ============================================================================
//  bench_chrass_overhead.cpp -- mede CUSTO de ter CHRASS ativo no JambaBlock
// ============================================================================
//
//  Compara construct + N steps fwd+bwd em dois modelos idênticos EXCETO
//  pelo flag use_chrass.  Reporta:
//    1. Delta de parameters() count (extras adicionados por CHRASS)
//    2. Tempo de N forward+backward por step (média, throughput)
//    3. Tempo total por config
//
//  NÃO mede qualidade (perplexidade, loss convergence) — isso exige treino
//  longo com eval set, que é o que o run do amigo na RTX 2080 Ti vai fazer.
// ============================================================================

#include "../include/jamba.h"
#include "../include/nsos_config.h"
#include "../include/tensor.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace nsos;

namespace {

ModelConfig make_config(bool with_chrass, int layers, int dim, int vocab) {
    ModelConfig cfg;
    cfg.num_layers = layers;
    cfg.d_model = dim;
    cfg.vocab_size = vocab;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 2;
    cfg.attention_period = 2;
    cfg.attention_slot = 1;
    cfg.use_moe = false;
    cfg.use_ttt = false;
    cfg.dropout = 0.0f;
    cfg.use_cuda = false;
    cfg.max_context_tokens = 64;
    cfg.use_chrass = with_chrass;
    cfg.chrass_density = 0.10f;  // 10% nonzero edges
    cfg.chrass_seed = 0xC11A55u;
    return cfg;
}

size_t count_params(JambaModel& model) {
    size_t total = 0;
    for (auto& blk : model.layers) {
        for (auto* p : blk->parameters()) {
            total += p->data.size;
        }
    }
    return total;
}

size_t count_chrass_params(JambaModel& model) {
    size_t total = 0;
    for (auto& blk : model.layers) {
        if (!blk->chrass_layer) continue;
        for (auto* p : blk->chrass_layer->parameters()) {
            total += p->data.size;
        }
    }
    return total;
}

double bench_forward_backward(JambaModel& model, int steps, int seq_len, int d_model) {
    std::vector<int> input_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) input_ids[i] = i + 1;

    Tensor dy({seq_len, d_model}, Device::CPU);
    for (int i = 0; i < dy.size; ++i) dy.data()[i] = 0.005f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        Tensor y = model.forward_ids(input_ids, nullptr);
        Tensor dy_iter = dy;
        for (int li = (int)model.layers.size() - 1; li >= 0; --li) {
            dy_iter = model.layers[li]->backward(dy_iter, nullptr);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
}

}  // namespace

int main() {
    std::cout << "==========================================================\n";
    std::cout << " CHRASS overhead benchmark — JambaModel + integration cost\n";
    std::cout << "==========================================================\n\n";

    // Moderate config: 8 layers, d=256, vocab=8192 → ~5M params total
    const int LAYERS = 8;
    const int DIM = 256;
    const int VOCAB = 8192;
    const int SEQ = 32;
    const int STEPS = 20;

    std::cout << "Config: " << LAYERS << " layers, d_model=" << DIM
              << ", vocab=" << VOCAB << ", seq=" << SEQ << ", steps=" << STEPS << "\n";
    std::cout << "CHRASS density: 10%\n\n";

    std::cout << "─── Building CHRASS-OFF baseline ───\n";
    auto cfg_off = make_config(false, LAYERS, DIM, VOCAB);
    JambaModel m_off(cfg_off, Device::CPU);
    size_t p_off = count_params(m_off);
    std::cout << "  params total: " << p_off << "\n";

    std::cout << "\n─── Building CHRASS-ON ───\n";
    auto cfg_on = make_config(true, LAYERS, DIM, VOCAB);
    JambaModel m_on(cfg_on, Device::CPU);
    size_t p_on = count_params(m_on);
    size_t p_chrass = count_chrass_params(m_on);
    std::cout << "  params total: " << p_on << "\n";
    std::cout << "  params chrass: " << p_chrass << "\n";
    std::cout << "  delta vs OFF:  " << (p_on - p_off) << " (";
    std::cout << std::fixed << std::setprecision(3)
              << (100.0 * (p_on - p_off) / p_off) << "%)\n";

    // Warmup
    std::cout << "\n─── Warmup (3 steps each) ───\n";
    bench_forward_backward(m_off, 3, SEQ, DIM);
    bench_forward_backward(m_on, 3, SEQ, DIM);

    std::cout << "\n─── Timing " << STEPS << " forward+backward steps ───\n";
    double t_off = bench_forward_backward(m_off, STEPS, SEQ, DIM);
    double t_on  = bench_forward_backward(m_on,  STEPS, SEQ, DIM);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  CHRASS-OFF: " << t_off << "s total ("
              << (t_off * 1000.0 / STEPS) << " ms/step)\n";
    std::cout << "  CHRASS-ON : " << t_on  << "s total ("
              << (t_on  * 1000.0 / STEPS) << " ms/step)\n";
    double overhead_pct = (t_on - t_off) / t_off * 100.0;
    std::cout << "  Overhead:   " << overhead_pct << "%\n";

    std::cout << "\n──────────────────────────────────────────────────────────\n";
    std::cout << " SUMMARY\n";
    std::cout << "──────────────────────────────────────────────────────────\n";
    std::cout << "  Extra params:    " << p_chrass << " ("
              << std::fixed << std::setprecision(2)
              << (100.0 * p_chrass / p_on) << "% of total)\n";
    std::cout << "  Time overhead:   " << overhead_pct << "% per step\n";
    std::cout << "  Avg step ratio:  ON/OFF = "
              << (t_on / t_off) << "x\n";
    std::cout << "==========================================================\n";
    return 0;
}
