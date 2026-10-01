#include "../include/jamba.h"
#include "../include/trainer.h"
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>

using namespace nsos;

// Structure to hold datasets
struct Dataset {
    std::vector<std::vector<int>> train_prompts;
    std::vector<std::vector<int>> train_answers;
    std::vector<std::vector<int>> val_in_prompts;
    std::vector<std::vector<int>> val_in_answers;
    std::vector<std::vector<int>> val_ood_prompts;
    std::vector<std::vector<int>> val_ood_answers;
};

// Function to generate the Sorting dataset
Dataset generate_sorting_dataset() {
    Dataset ds;
    std::mt19937 rng(1337);
    
    // In-distribution: numbers 1 to 8. We sample 50 unique permutations of length 4.
    std::vector<std::vector<int>> all_id;
    while (all_id.size() < 50) {
        std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8};
        std::shuffle(vals.begin(), vals.end(), rng);
        std::vector<int> perm(vals.begin(), vals.begin() + 4);
        if (std::find(all_id.begin(), all_id.end(), perm) == all_id.end()) {
            all_id.push_back(perm);
        }
    }
    
    // Split: 40 training, 10 in-distribution validation
    for (size_t i = 0; i < all_id.size(); ++i) {
        std::vector<int> prompt = all_id[i];
        prompt.push_back(99); // separator token
        
        std::vector<int> answer = all_id[i];
        std::sort(answer.begin(), answer.end());
        answer.push_back(0); // EOS
        
        if (i < 40) {
            ds.train_prompts.push_back(prompt);
            ds.train_answers.push_back(answer);
        } else {
            ds.val_in_prompts.push_back(prompt);
            ds.val_in_answers.push_back(answer);
        }
    }
    
    // Out-of-distribution: numbers 10 to 13 (larger numbers).
    // Let's generate 5 unique permutations of length 4.
    std::vector<std::vector<int>> all_ood;
    while (all_ood.size() < 5) {
        std::vector<int> vals = {10, 11, 12, 13};
        std::shuffle(vals.begin(), vals.end(), rng);
        std::vector<int> perm(vals.begin(), vals.begin() + 4);
        if (std::find(all_ood.begin(), all_ood.end(), perm) == all_ood.end()) {
            all_ood.push_back(perm);
        }
    }
    for (size_t i = 0; i < all_ood.size(); ++i) {
        std::vector<int> prompt = all_ood[i];
        prompt.push_back(99);
        
        std::vector<int> answer = all_ood[i];
        std::sort(answer.begin(), answer.end());
        answer.push_back(0);
        
        ds.val_ood_prompts.push_back(prompt);
        ds.val_ood_answers.push_back(answer);
    }
    
    return ds;
}

// Function to generate the Sequence Reversal dataset
Dataset generate_reversal_dataset() {
    Dataset ds;
    std::mt19937 rng(4242);
    
    // In-distribution: numbers 1 to 10. We sample 50 unique sequences of length 5.
    std::vector<std::vector<int>> all_id;
    while (all_id.size() < 50) {
        std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::shuffle(vals.begin(), vals.end(), rng);
        std::vector<int> seq(vals.begin(), vals.begin() + 5);
        if (std::find(all_id.begin(), all_id.end(), seq) == all_id.end()) {
            all_id.push_back(seq);
        }
    }
    
    // Split: 40 training, 10 in-distribution validation
    for (size_t i = 0; i < all_id.size(); ++i) {
        std::vector<int> prompt = all_id[i];
        prompt.push_back(99); // separator token
        
        std::vector<int> answer = all_id[i];
        std::reverse(answer.begin(), answer.end());
        answer.push_back(0); // EOS
        
        if (i < 40) {
            ds.train_prompts.push_back(prompt);
            ds.train_answers.push_back(answer);
        } else {
            ds.val_in_prompts.push_back(prompt);
            ds.val_in_answers.push_back(answer);
        }
    }
    
    // Out-of-distribution: numbers 50 to 54 (larger numbers).
    // Let's generate 5 unique sequences of length 5.
    std::vector<std::vector<int>> all_ood;
    while (all_ood.size() < 5) {
        std::vector<int> vals = {50, 51, 52, 53, 54};
        std::shuffle(vals.begin(), vals.end(), rng);
        std::vector<int> seq(vals.begin(), vals.begin() + 5);
        if (std::find(all_ood.begin(), all_ood.end(), seq) == all_ood.end()) {
            all_ood.push_back(seq);
        }
    }
    for (size_t i = 0; i < all_ood.size(); ++i) {
        std::vector<int> prompt = all_ood[i];
        prompt.push_back(99);
        
        std::vector<int> answer = all_ood[i];
        std::reverse(answer.begin(), answer.end());
        answer.push_back(0);
        
        ds.val_ood_prompts.push_back(prompt);
        ds.val_ood_answers.push_back(answer);
    }
    
    return ds;
}

int predict_next_token(JambaModel& model, const std::vector<int>& context) {
    model.reset_session();
    Tensor logits = model.forward_ids(context, nullptr);
    Tensor host = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
    int vocab = host.shape.back();
    const float* row = host.data() + host.size - vocab;
    int best_id = 0;
    float best_logit = row[0];
    for (int i = 1; i < vocab; ++i) {
        if (row[i] > best_logit) {
            best_logit = row[i];
            best_id = i;
        }
    }
    return best_id;
}

void print_sequence(const std::vector<int>& seq) {
    std::printf("[");
    for (size_t i = 0; i < seq.size(); ++i) {
        std::printf("%d%s", seq[i], (i + 1 < seq.size()) ? ", " : "");
    }
    std::printf("]");
}

// Evaluation helper
float evaluate_model(JambaModel& model, const std::vector<std::vector<int>>& prompts, const std::vector<std::vector<int>>& targets, bool print_samples = false) {
    int correct_count = 0;
    for (size_t p = 0; p < prompts.size(); ++p) {
        std::vector<int> ctx = prompts[p];
        std::vector<int> target = targets[p];
        std::vector<int> predicted;
        bool is_correct = true;
        
        for (size_t i = 0; i < target.size(); ++i) {
            int next_token = predict_next_token(model, ctx);
            predicted.push_back(next_token);
            ctx.push_back(next_token);
            if (next_token != target[i]) {
                is_correct = false;
            }
        }
        if (is_correct) {
            correct_count++;
        }
        
        if (print_samples && p < 1) {
            std::printf("       Sample - Input: ");
            std::vector<int> inp(prompts[p].begin(), prompts[p].end() - 1);
            print_sequence(inp);
            std::printf(" -> Target: ");
            print_sequence(target);
            std::printf(" | Pred: ");
            print_sequence(predicted);
            std::printf(" %s\n", is_correct ? "CORRECT" : "WRONG");
        }
    }
    return (float)correct_count / prompts.size();
}

void run_experiment(const std::string& name, ModelConfig config, const Dataset& ds, const std::string& task_name, Device dev) {
    std::printf("\n--- EXPERIMENT: %s [%s] on %s ---\n", name.c_str(), task_name.c_str(), (dev == Device::GPU) ? "GPU" : "CPU");
    
    config.d_model = 64;
    config.vocab_size = 101;
    config.dropout = 0.0f;
    config.use_cuda = (dev == Device::GPU);

    JambaModel model(config, dev);
    
    // Print layer audit types
    std::printf("   Layer configuration:\n");
    for (size_t i = 0; i < model.layers.size(); ++i) {
        std::printf("     Layer %zu: %s\n", i, model.layers[i]->audit_block_type().c_str());
    }

    Trainer trainer(&model, 0.01f);
    trainer.weight_decay = 0.0f;
    trainer.max_grad_norm = 1.0f;
    trainer.warmup_steps = 5;
    if (config.pantheon_vib_beta > 0.0f) {
        trainer.pantheon_vib_beta = config.pantheon_vib_beta;
    }

    // Train on batch
    int steps = 300;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= steps; ++step) {
        float loss = trainer.train_supervised_batch(ds.train_prompts, ds.train_answers);
        if (step == 1 || step % 100 == 0 || step == steps) {
            std::printf("     Step %3d - Loss: %.6f\n", step, loss);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::printf("   Training completed in %.2f ms\n", duration);

    // Evaluate
    std::printf("   Evaluating:\n");
    float train_acc = evaluate_model(model, ds.train_prompts, ds.train_answers, true);
    float val_in_acc = evaluate_model(model, ds.val_in_prompts, ds.val_in_answers, true);
    float val_ood_acc = evaluate_model(model, ds.val_ood_prompts, ds.val_ood_answers, true);

    std::printf("   >>> RESULTS <<<\n");
    std::printf("     Train Accuracy:              %.2f%%\n", train_acc * 100.0f);
    std::printf("     In-Distribution Val Acc:     %.2f%%\n", val_in_acc * 100.0f);
    std::printf("     Out-of-Distribution Val Acc: %.2f%%\n", val_ood_acc * 100.0f);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("=====================================================================\n");
    std::printf(" NSOS Multi-Technology Validation & Generalization Suite\n");
    std::printf("=====================================================================\n");

    std::printf("\nGenerating Datasets...\n");
    Dataset sort_ds = generate_sorting_dataset();
    Dataset rev_ds = generate_reversal_dataset();
    std::printf("Datasets generated successfully.\n");
    std::printf("  Sorting:  %zu train, %zu ID val, %zu OOD val\n", sort_ds.train_prompts.size(), sort_ds.val_in_prompts.size(), sort_ds.val_ood_prompts.size());
    std::printf("  Reversal: %zu train, %zu ID val, %zu OOD val\n", rev_ds.train_prompts.size(), rev_ds.val_in_prompts.size(), rev_ds.val_ood_prompts.size());

    // Define configurations
    std::vector<std::pair<std::string, ModelConfig>> configs;

    // 1. Pure Mamba-only
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = false;
        cfg.attention_period = 8;
        configs.push_back({"Mamba-Only", cfg});
    }

    // 2. Pure Attention-only
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = false;
        cfg.attention_period = 1;
        cfg.attention_slot = 0;
        configs.push_back({"Attention-Only", cfg});
    }

    // 3. Pure TTT-only (Commented out to bypass double-backprop bottlenecks on CPU/GPU)
    /*
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = true;
        cfg.ttt_period = 1;
        cfg.ttt_slot = 0;
        configs.push_back({"TTT-Only", cfg});
    }
    */

    // 4. MoE + Mamba
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = true;
        cfg.moe_period = 1;
        cfg.moe_slot = 0;
        cfg.num_experts = 4;
        cfg.num_experts_per_token = 1;
        cfg.use_ttt = false;
        cfg.attention_period = 8;
        configs.push_back({"MoE + Mamba", cfg});
    }

    // 5. MoE + Attention
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = true;
        cfg.moe_period = 1;
        cfg.moe_slot = 0;
        cfg.num_experts = 4;
        cfg.num_experts_per_token = 1;
        cfg.use_ttt = false;
        cfg.attention_period = 1;
        cfg.attention_slot = 0;
        configs.push_back({"MoE + Attention", cfg});
    }

    // 6. Deep Hybrid (4 layers) (Commented out to bypass double-backprop bottlenecks on CPU/GPU)
    /*
    {
        ModelConfig cfg;
        cfg.num_layers = 4;
        cfg.attention_period = 4;
        cfg.attention_slot = 1;
        cfg.use_ttt = true;
        cfg.ttt_period = 4;
        cfg.ttt_slot = 2;
        cfg.use_moe = true;
        cfg.moe_period = 4;
        cfg.moe_slot = 3;
        cfg.num_experts = 4;
        cfg.num_experts_per_token = 1;
        configs.push_back({"Deep Hybrid (4 layers)", cfg});
    }
    */

    // 7. CHRASS Hybrid
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = false;
        cfg.attention_period = 2;
        cfg.attention_slot = 0;
        cfg.use_chrass = true;
        cfg.chrass_density = 0.15f;
        cfg.chrass_seed = 12345u;
        configs.push_back({"CHRASS Hybrid", cfg});
    }

    // 8. KAN Hybrid
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = false;
        cfg.attention_period = 2;
        cfg.attention_slot = 0;
        cfg.use_kan = true;
        configs.push_back({"KAN Hybrid", cfg});
    }

    // 9. Slender Quantized Embedding Hybrid
    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.use_moe = false;
        cfg.use_ttt = false;
        cfg.attention_period = 2;
        cfg.attention_slot = 0;
        cfg.use_slender_embedding = true;
        configs.push_back({"Slender Quantized Embedding Hybrid", cfg});
    }

    // 10. Unified Mega-Hybrid Ultimate (Commented out to bypass double-backprop bottlenecks on CPU/GPU)
    /*
    {
        ModelConfig cfg;
        cfg.num_layers = 4;
        cfg.attention_period = 4;
        cfg.attention_slot = 1;
        cfg.use_ttt = true;
        cfg.ttt_period = 4;
        cfg.ttt_slot = 2;
        cfg.use_moe = true;
        cfg.moe_period = 4;
        cfg.moe_slot = 3;
        cfg.num_experts = 4;
        cfg.num_experts_per_token = 1;
        cfg.use_chrass = true;
        cfg.chrass_density = 0.15f;
        cfg.use_kan = true;
        cfg.use_slender_embedding = true;
        cfg.pantheon_vib_beta = 0.05f;
        configs.push_back({"Unified Mega-Hybrid Ultimate", cfg});
    }
    */

    // Run all configurations on both tasks
    for (const auto& item : configs) {
        Device dev = (item.second.use_ttt || item.second.use_slender_embedding) ? Device::CPU : Device::GPU;
        run_experiment(item.first, item.second, sort_ds, "Sorting", dev);
        run_experiment(item.first, item.second, rev_ds, "Reversal", dev);
    }

    std::printf("\n=====================================================================\n");
    std::printf(" All validation experiments completed.\n");
    std::printf("=====================================================================\n");

    return 0;
}
