#include "memory_system.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

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

        std::filesystem::remove_all(temp_root);
        std::cout << "OxtaMem FFI test passed!" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "OxtaMem FFI test failed: " << ex.what() << std::endl;
        return 1;
    }
}
