#include "memory_system.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nsos;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        const auto temp_root = std::filesystem::temp_directory_path() / "nsos_oxtamem_ffi_test";
        std::filesystem::remove_all(temp_root);
        std::filesystem::create_directories(temp_root);

        {
            MemorySystem memory(4);
            require(memory.enable_oxtamem_store("", (temp_root / "ffi_store.db").string(), 16),
                    "failed to enable OxtaMem FFI backend");

            Message first;
            first.role = "user";
            first.agent_id = "alpha";
            first.timestamp = "2026-04-04T00:00:00Z";
            first.content = "primeira mensagem";

            Message second;
            second.role = "assistant";
            second.agent_id = "beta";
            second.timestamp = "2026-04-04T00:00:01Z";
            second.content = "segunda mensagem";

            memory.add_message(first);
            memory.add_message(second);

            const auto recalled = memory.recall_recent_messages(2);
            require(recalled.size() == 2, "unexpected message count from OxtaMem backend");
            require(recalled[0].content == second.content, "latest message mismatch");
            require(recalled[1].content == first.content, "historical message mismatch");
        }

        // Similarity results are part of the model-visible OxtaMEM contract.
        // Equal-score hits must have a stable durable-address tie-break, both
        // across repeated calls and after rebuilding HNSW from the sidecar.
        const auto vector_store_path = temp_root / "deterministic_vectors.db";
        const std::vector<std::vector<uint8_t>> expected = {
            {'f', 'i', 'r', 's', 't'},
            {'s', 'e', 'c', 'o', 'n', 'd'},
            {'t', 'h', 'i', 'r', 'd'},
        };
        std::vector<float> tied_vector(128, 0.0f);
        tied_vector[0] = 1.0f;

        {
            OxtaMemFFI store;
            require(store.load(""),
                    "failed to load OxtaMem for deterministic search");
            require(store.open(vector_store_path.string(), 16),
                    "failed to open deterministic OxtaMem store");
            for (const auto& payload : expected) {
                require(store.write_with_vector(
                            "ties", payload, tied_vector),
                        "failed to write deterministic OxtaMem vector");
            }
            for (int repeat = 0; repeat < 100; ++repeat) {
                require(store.search_similar(tied_vector, expected.size()) ==
                            expected,
                        "OxtaMem tie order changed within one process");
            }
        }

        {
            OxtaMemFFI reopened;
            require(reopened.load(""),
                    "failed to reload OxtaMem for deterministic search");
            require(reopened.open(vector_store_path.string(), 16),
                    "failed to reopen deterministic OxtaMem store");
            for (int repeat = 0; repeat < 100; ++repeat) {
                require(reopened.search_similar(
                            tied_vector, expected.size()) == expected,
                        "OxtaMem tie order changed after index rebuild");
            }
        }

        const auto episodic_path = temp_root / "reopened_episodic.db";
        Tensor old_state = Tensor::zeros({4}, Device::CPU);
        old_state.data()[0] = 1.0f;
        {
            MemorySystem memory(4);
            require(memory.enable_oxtamem_store("", episodic_path.string(), 16),
                    "failed to open episodic store");
            memory.store_episodic(old_state);
            require(memory.retrieve(old_state, 8).size() == 1, "durable/RAM duplicate");
        }
        {
            MemorySystem reopened(4);
            require(reopened.enable_oxtamem_store("", episodic_path.string(), 16),
                    "failed to reopen episodic store");
            Tensor newer = Tensor::zeros({4}, Device::CPU);
            newer.data()[1] = 1.0f;
            reopened.store_episodic(newer);
            const auto recalled = reopened.retrieve(old_state, 1);
            require(recalled.size() == 1 && recalled[0].data()[0] == 1.0f &&
                        recalled[0].data()[1] == 0.0f,
                    "new RAM memory hid persisted OxtaMem history");
            require(reopened.retrieve(old_state, 8).size() == 2,
                    "OxtaMem/RAM retrieval did not deduplicate");
        }

        // Windows keeps an mmap-backed arena locked until MemorySystem and the
        // Rust FFI handle have both been destroyed.
        std::filesystem::remove_all(temp_root);
        std::cout << "OxtaMem FFI test passed!" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "OxtaMem FFI test failed: " << ex.what() << std::endl;
        return 1;
    }
}
