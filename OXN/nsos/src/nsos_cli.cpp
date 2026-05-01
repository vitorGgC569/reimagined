#include "../include/nsos_sdk.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace nsos;

namespace {

std::string get_flag(const std::vector<std::string>& args,
                     const std::string& name,
                     const std::string& fallback = "") {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == name) {
            return args[i + 1];
        }
    }
    return fallback;
}

bool has_flag(const std::vector<std::string>& args, const std::string& name) {
    return std::find(args.begin(), args.end(), name) != args.end();
}

int get_int_flag(const std::vector<std::string>& args,
                 const std::string& name,
                 int fallback) {
    const std::string value = get_flag(args, name);
    return value.empty() ? fallback : std::stoi(value);
}

float get_float_flag(const std::vector<std::string>& args,
                     const std::string& name,
                     float fallback) {
    const std::string value = get_flag(args, name);
    return value.empty() ? fallback : std::stof(value);
}

void print_help() {
    std::cout
        << "NSOS CLI\n"
        << "Usage:\n"
        << "  nsos_cli generate --model <path> --prompt <text> [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--stream] [--cuda]\n"
        << "  nsos_cli train-text --model <path> --text <text> [--steps N] [--save-pack <dir>] [--cuda]\n"
        << "  nsos_cli pack --model <path> --out <dir> [--cuda]\n"
        << "  nsos_cli info --model <path> [--cuda]\n";
}

ModelConfig read_runtime_config(const std::vector<std::string>& args) {
    ModelConfig config;
    if (has_flag(args, "--cuda")) {
        config.use_cuda = true;
    }
    if (has_flag(args, "--max-context")) {
        config.max_context_tokens = get_int_flag(args, "--max-context", config.max_context_tokens);
    }
    return config;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (args.empty()) {
            print_help();
            return 0;
        }

        const std::string command = args.front();
        const std::string model_path = get_flag(args, "--model", "");

        if (command == "generate") {
            const std::string prompt = get_flag(args, "--prompt", "");
            if (prompt.empty()) {
                std::cerr << "Missing --prompt\n";
                return 1;
            }

            InferenceEngine engine;
            if (!engine.load_model(model_path, read_runtime_config(args))) {
                std::cerr << "Failed to load model\n";
                return 1;
            }

            GenerationOptions options;
            options.max_tokens = get_int_flag(args, "--max-tokens", options.max_tokens);
            options.temperature = get_float_flag(args, "--temperature", options.temperature);
            options.top_p = get_float_flag(args, "--top-p", options.top_p);
            options.top_k = get_int_flag(args, "--top-k", options.top_k);
            options.max_context_tokens =
                get_int_flag(args, "--max-context", engine.config.max_context_tokens);
            options.stream = has_flag(args, "--stream");

            std::string result;
            if (options.stream) {
                result = engine.generate_stream(
                    prompt, options, [](const std::string& chunk) {
                        std::cout << chunk << std::flush;
                    });
                std::cout << "\n";
            } else {
                result = engine.generate(prompt, options);
                std::cout << result << "\n";
            }

            const auto& metrics = engine.last_generation_metrics();
            std::cerr << "[metrics] prompt_total=" << metrics.prompt_tokens_total
                      << " prompt_used=" << metrics.prompt_tokens_used
                      << " generated=" << metrics.generated_tokens
                      << " elapsed_ms=" << metrics.elapsed_ms
                      << " pack=" << (metrics.loaded_from_pack ? "yes" : "no") << "\n";
            return 0;
        }

        if (command == "train-text") {
            const std::string text = get_flag(args, "--text", "");
            const int steps = std::max(get_int_flag(args, "--steps", 1), 1);
            const std::string save_pack_dir = get_flag(args, "--save-pack", "");
            if (text.empty()) {
                std::cerr << "Missing --text\n";
                return 1;
            }

            InferenceEngine engine;
            if (!engine.load_model(model_path, read_runtime_config(args))) {
                std::cerr << "Failed to load model\n";
                return 1;
            }

            float last_loss = 0.0f;
            for (int step = 0; step < steps; ++step) {
                last_loss = engine.train_step(text);
                std::cerr << "[train] step=" << (step + 1) << " loss=" << last_loss << "\n";
            }

            if (!save_pack_dir.empty()) {
                if (!engine.save_model_pack(save_pack_dir)) {
                    std::cerr << "Failed to save model pack\n";
                    return 1;
                }
                std::cerr << "[train] saved model pack to " << save_pack_dir << "\n";
            }
            return 0;
        }

        if (command == "pack") {
            const std::string out_dir = get_flag(args, "--out", "");
            if (out_dir.empty()) {
                std::cerr << "Missing --out\n";
                return 1;
            }

            InferenceEngine engine;
            if (!engine.load_model(model_path, read_runtime_config(args))) {
                std::cerr << "Failed to load model\n";
                return 1;
            }
            if (!engine.save_model_pack(out_dir)) {
                std::cerr << "Failed to save model pack\n";
                return 1;
            }
            std::cout << out_dir << "\n";
            return 0;
        }

        if (command == "info") {
            InferenceEngine engine;
            if (!engine.load_model(model_path, read_runtime_config(args))) {
                std::cerr << "Failed to load model\n";
                return 1;
            }

            std::cout << "num_layers=" << engine.config.num_layers << "\n"
                      << "d_model=" << engine.config.d_model << "\n"
                      << "vocab_size=" << engine.config.vocab_size << "\n"
                      << "max_context_tokens=" << engine.config.max_context_tokens << "\n"
                      << "default_batch_size=" << engine.config.default_batch_size << "\n"
                      << "use_cuda=" << (engine.config.use_cuda ? "true" : "false") << "\n"
                      << "memory_bytes=" << engine.get_memory_usage() << "\n";
            return 0;
        }

        print_help();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "nsos_cli failed: " << ex.what() << "\n";
        return 1;
    }
}
