#include "jamba.h"
#include "trainer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace nsos;

namespace {

struct TrainingConfig {
    int num_layers = 2;
    int d_model = 128;
    int vocab_size = 256;
    int epochs = 1;
    int batch_size = 1;
    int seq_len = 128;
    int max_steps = 256;
    int checkpoint_every = 64;
    float learning_rate = 0.001f;
    bool resume = false;
    std::string dataset_path = "circuit_data/train.raw";
    std::string checkpoint_path = "circuit_brain_checkpoint.bin";
};

int parse_int(const std::string& value, const char* flag_name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + flag_name + ": " + value);
    }
}

float parse_float(const std::string& value, const char* flag_name) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid float for ") + flag_name + ": " + value);
    }
}

TrainingConfig parse_args(int argc, char** argv) {
    TrainingConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* flag_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag_name);
            }
            return argv[++i];
        };

        if (arg == "--dataset") {
            config.dataset_path = require_value("--dataset");
        } else if (arg == "--checkpoint") {
            config.checkpoint_path = require_value("--checkpoint");
        } else if (arg == "--epochs") {
            config.epochs = parse_int(require_value("--epochs"), "--epochs");
        } else if (arg == "--batch-size") {
            config.batch_size = parse_int(require_value("--batch-size"), "--batch-size");
        } else if (arg == "--seq-len") {
            config.seq_len = parse_int(require_value("--seq-len"), "--seq-len");
        } else if (arg == "--max-steps") {
            config.max_steps = parse_int(require_value("--max-steps"), "--max-steps");
        } else if (arg == "--checkpoint-every") {
            config.checkpoint_every = parse_int(require_value("--checkpoint-every"), "--checkpoint-every");
        } else if (arg == "--learning-rate") {
            config.learning_rate = parse_float(require_value("--learning-rate"), "--learning-rate");
        } else if (arg == "--layers") {
            config.num_layers = parse_int(require_value("--layers"), "--layers");
        } else if (arg == "--d-model") {
            config.d_model = parse_int(require_value("--d-model"), "--d-model");
        } else if (arg == "--vocab-size") {
            config.vocab_size = parse_int(require_value("--vocab-size"), "--vocab-size");
        } else if (arg == "--resume") {
            config.resume = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.num_layers <= 0 || config.d_model <= 0 || config.vocab_size <= 0 ||
        config.epochs <= 0 || config.batch_size <= 0 || config.seq_len <= 0 ||
        config.learning_rate <= 0.0f) {
        throw std::runtime_error("Training parameters must be positive");
    }

    if (config.max_steps == 0 || config.checkpoint_every == 0) {
        throw std::runtime_error("max_steps and checkpoint_every must be positive or -1");
    }

    return config;
}

std::vector<int> load_byte_level_dataset(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open dataset: " + path.string());
    }

    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());
    if (bytes.size() < 512) {
        throw std::runtime_error("Dataset is too small for training");
    }

    std::vector<int> tokens;
    tokens.reserve(bytes.size());
    for (unsigned char value : bytes) {
        tokens.push_back(static_cast<int>(value));
    }
    return tokens;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "Starting NSOS circuit trainer..." << std::endl;

    TrainingConfig config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Argument parsing failed: " << error.what() << std::endl;
        return 1;
    }

    const std::filesystem::path dataset_path = config.dataset_path;
    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "Missing dataset at " << dataset_path
                  << ". Generate it first with generate_circuit_data.py"
                  << std::endl;
        return 1;
    }

    Device device = Device::CPU;
#ifdef USE_CUDA
    device = Device::GPU;
    std::cout << "CUDA build detected. Training on GPU." << std::endl;
#else
    std::cout << "Training on CPU." << std::endl;
#endif

    auto tokens = load_byte_level_dataset(dataset_path);
    std::cout << "Loaded " << tokens.size() << " byte-level circuit tokens."
              << std::endl;
    std::cout << "Config: layers=" << config.num_layers
              << " d_model=" << config.d_model
              << " vocab=" << config.vocab_size
              << " epochs=" << config.epochs
              << " batch_size=" << config.batch_size
              << " seq_len=" << config.seq_len
              << " max_steps=" << config.max_steps
              << " checkpoint_every=" << config.checkpoint_every
              << " resume=" << (config.resume ? "true" : "false")
              << " lr=" << config.learning_rate << std::endl;

    auto model = std::make_unique<JambaModel>(
        config.num_layers, config.d_model, config.vocab_size, device);
    if (config.resume && std::filesystem::exists(config.checkpoint_path)) {
        model->load(config.checkpoint_path);
        std::cout << "Resumed weights from " << config.checkpoint_path << std::endl;
    }
    Trainer trainer(model.get(), config.learning_rate);

    try {
        auto callback = [&](int step, float loss) {
            if (step == 1 || step % 25 == 0) {
                std::cout << "[step " << step << "] loss=" << loss << std::endl;
            }

            if (config.checkpoint_every > 0 && step % config.checkpoint_every == 0) {
                model->save(config.checkpoint_path);
                std::cout << "Saved checkpoint at step " << step
                          << " -> " << config.checkpoint_path << std::endl;
            }
        };

        trainer.train_loop(tokens,
                           config.epochs,
                           config.batch_size,
                           config.seq_len,
                           callback,
                           config.max_steps);
        model->save(config.checkpoint_path);
        std::cout << "Training finished. Saved checkpoint to "
                  << config.checkpoint_path << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Circuit training failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
