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
    const std::filesystem::path store_path =
        std::filesystem::temp_directory_path() / "nsos_memory_causal_store.bin";
    std::filesystem::remove(store_path);

    MemorySystem memory(4);
    memory.enable_causal_store(store_path.string());

    Message first;
    first.role = "user";
    first.agent_id = "alpha";
    first.timestamp = "2026-04-04T00:00:00Z";
    first.content = "hello\nworld";

    Message second;
    second.role = "assistant";
    second.agent_id = "nsos";
    second.timestamp = "2026-04-04T00:00:01Z";
    second.content = "world";

    memory.add_message(first);
    memory.add_message(second);

    Tensor state({4}, Device::CPU);
    for (int i = 0; i < state.size; ++i) state.data()[i] = static_cast<float>(i);
    memory.store_episodic(state);

    const auto history = memory.recall_recent_messages(2);
    require(history.size() == 2, "expected two persisted messages");
    require(history[0].content == "world", "latest message mismatch");
    require(history[1].content == "hello\nworld", "oldest message mismatch");
    require(std::filesystem::exists(store_path), "causal store file was not created");

    MemorySystem volatile_memory(4);
    volatile_memory.add_message(first);
    volatile_memory.add_message(second);
    const auto in_memory_history = volatile_memory.recall_recent_messages(2);
    require(in_memory_history.size() == 2, "in-memory recall_recent_messages fallback failed");
    require(in_memory_history[0].content == "hello\nworld", "in-memory oldest message mismatch");
    require(in_memory_history[1].content == "world", "in-memory latest message mismatch");

    std::cout << "Memory causal store test passed!" << std::endl;
    std::filesystem::remove(store_path);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Memory causal store test failed: " << ex.what() << std::endl;
    return 1;
  }
}
