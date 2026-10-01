#include "nsos/coordinator.h"
#include "mcts_reasoning.h"
#include "memory_system.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>

namespace nsos {
namespace swarm {

namespace {

MemorySystem& shared_memory_system() {
    static MemorySystem memory(64);
    return memory;
}

} // namespace

void Worker::run_loop() {
    while (running_ && parent_) {
        TaskNotification task;
        // Task Claiming from the Global Queue (blocks on condition_variable)
        if (parent_->claim_task(task, identity_.id)) {
            if (task.type == "stop") break;
            execute_reasoning(task);
        }
    }
}

void Worker::execute_reasoning(const TaskNotification& task) {
    Mailbox& mailbox = Mailbox::get_instance();

    // 1. Acknowledge task in the persistent mailbox
    TaskNotification ack;
    ack.id          = "ack_" + task.id;
    ack.sender_id   = identity_.id;
    ack.receiver_id = "coordinator";
    ack.type        = "status";
    ack.content     = "[Worker " + identity_.name + "] Claimed task: " + task.id;
    ack.timestamp   = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    mailbox.deliver(ack);

    // =========================================================================
    // 2. SEMANTIC TASK ROUTING
    // O tipo da task determina qual subsistema NSOS é acionado.
    // Nenhum sleep() falso: cada rota executa computação real.
    // =========================================================================
    std::string result_text;
    float confidence = 0.0f;
    auto t_start = std::chrono::steady_clock::now();

    if (task.type == "critical_research") {
        // --- Rota: MCTS Search (System 2 Reasoning) ---
        std::cout << "[Swarm:" << identity_.name << "] MCTS Search para: "
                  << task.content << "\n";

        // Dimensão do estado proxy: comprimento da string (par, min 64)
        const int raw_dim = std::max(
            static_cast<int>(task.content.size()), 64);
        const int even_dim = (raw_dim % 2 == 0) ? raw_dim : raw_dim + 1;

        // Estado raiz: embedding determinístico da query
        Tensor root_state({even_dim}, Device::CPU);
        float* rs = root_state.data();
        for (int i = 0; i < even_dim; ++i) {
            // Hash posicional determinístico → range [-1, 1]
            size_t h = std::hash<std::string>{}(task.content)
                     ^ (static_cast<size_t>(i) * 2654435761ULL);
            rs[i] = static_cast<float>(h % 1000) / 500.0f - 1.0f;
        }

        // Evaluator: negativo da norma L2 (estados mais compactos = melhor)
        auto evaluator = [](const Tensor& s) -> float {
            if (s.size == 0) return 0.0f;
            float norm_sq = 0.0f;
            const float* d = s.data();
            for (int i = 0; i < s.size; ++i) norm_sq += d[i] * d[i];
            return -std::sqrt(norm_sq / std::max<int64_t>(s.size, 1));
        };

        MCTSConfig cfg;
        cfg.num_simulations            = 200;
        cfg.max_depth                  = 8;
        cfg.num_children_per_expansion = 3;
        cfg.use_noise                  = true;

        MCTSReasoning mcts(root_state, evaluator, cfg);
        mcts.search();

        float best_val = mcts.get_best_value();
        confidence     = std::tanh(-best_val); // Mapear para [0, 1]

        result_text = "[MCTS Result] Melhor valor: " + std::to_string(best_val)
                    + " | Nós: " + std::to_string(mcts.num_nodes())
                    + " | Confiança: " + std::to_string(confidence)
                    + " | Query: " + task.content;

    } else if (task.type == "maintenance") {
        // --- Rota: AutoDream + UltraCompact (Memory Compression) ---
        std::cout << "[Swarm:" << identity_.name << "] Compressão de memória: "
                  << task.content << "\n";

        MemorySystem& mem_sys = shared_memory_system();

        // Popular com estados episódicos baseados na task
        for (int i = 0; i < 8; ++i) {
            Tensor state({64}, Device::CPU);
            float* sd = state.data();
            size_t seed = std::hash<std::string>{}(task.content + std::to_string(i));
            for (int j = 0; j < 64; ++j) {
                // LCG simples para gerar valores pseudo-aleatórios reprodutíveis
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                sd[j] = (static_cast<float>(seed & 0xFFFF) / 32767.5f) - 1.0f;
            }
            mem_sys.store_episodic(state);
        }

        mem_sys.run_auto_dream();    // Comprime clusters com > 5 itens via TurboQuant
        mem_sys.run_ultra_compact(); // Consolida clusters estagnados (> 10 min)

        confidence  = 0.95f;
        result_text = "[Maintenance Result] AutoDream + UltraCompact executados para: "
                    + task.content;

    } else if (task.type == "retrieval") {
        // --- Rota: Memory Retrieval ---
        std::cout << "[Swarm:" << identity_.name << "] Memory Retrieval: "
                  << task.content << "\n";

        MemorySystem& mem_sys = shared_memory_system();
        Tensor query({64}, Device::CPU);
        float* qd = query.data();
        size_t seed = std::hash<std::string>{}(task.content);
        for (int j = 0; j < 64; ++j) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            qd[j] = (static_cast<float>(seed & 0xFFFF) / 32767.5f) - 1.0f;
        }

        Tensor ctx_result = mem_sys.retrieve(query);
        confidence  = (ctx_result.size > 0) ? 0.8f : 0.1f;
        result_text = "[Retrieval Result] Contexto dim=" + std::to_string(ctx_result.size)
                    + " para: " + task.content;

    } else {
        // --- Rota padrão ---
        std::cout << "[Swarm:" << identity_.name << "] Task genérica: "
                  << task.content << "\n";
        confidence  = 0.5f;
        result_text = "[Generic Result] " + identity_.name + " processou: " + task.content;
    }

    // 3. Medir latência real
    auto t_end     = std::chrono::steady_clock::now();
    float latency_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

    // 4. Entregar resultado com métricas estruturadas
    TaskNotification response;
    response.id          = "resp_" + task.id;
    response.sender_id   = identity_.id;
    response.receiver_id = "coordinator";
    response.type        = "status";

    std::string metrics =
        "\n<metrics confidence=\"" + std::to_string(confidence)
        + "\" latency_ms=\""       + std::to_string(latency_ms)
        + "\" model=\""            + identity_.model + "\"/>";

    response.content  = mailbox.wrap_xml(identity_.id,
                                         result_text + metrics,
                                         identity_.color);
    response.timestamp = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());

    mailbox.deliver(response);
}

} // namespace swarm
} // namespace nsos
