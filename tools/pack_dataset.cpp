
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "../OXN/nsos/include/tokenizer.h"
#include "../OXB/aion_core_cpp/include/OX3Serializer.h"

// Simple Packer: Text -> OX3
// Usage: ./pack_dataset input.jsonl output.ox3

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.jsonl> <output.ox3>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "[Packer] Loading Tokenizer..." << std::endl;
    Tokenizer tokenizer;
    // Add special tokens manually for now as we don't have a vocab file yet
    tokenizer.add_special_tokens({"<think>", "</think>"});

    std::cout << "[Packer] Opening " << input_path << "..." << std::endl;
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    OX3Serializer serializer(output_path);

    std::string line;
    int count = 0;
    while (std::getline(infile, line)) {
        // Parse JSON? Or assume raw text for simplicity in V1 packer?
        // The generator outputs JSONL: {"input": "...", "target": "..."}
        // We need a JSON parser.
        // For V1, let's do a naive string search for "input" and "target" to avoid heavy deps
        // or just tokenize the whole line.
        // Industrial Standard: Use simdjson. Here: Naive.

        // Let's tokenize the whole line for now to prove the flow.
        std::vector<int> tokens = tokenizer.encode(line);

        // Write Record
        // Score = 1.0 (placeholder)
        serializer.write_record(tokens, 1.0f);
        count++;

        if (count % 1000 == 0) std::cout << "[Packer] Packed " << count << " records." << std::endl;
    }

    std::cout << "[Packer] Done. Written to " << output_path << std::endl;
    return 0;
}
