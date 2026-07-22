#include "memory_system.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
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

    const auto corrupt_path =
        std::filesystem::temp_directory_path() / "nsos_memory_causal_corrupt.bin";
    std::filesystem::remove(corrupt_path);
    {
      CausalMemoryStore store(corrupt_path.string());
      store.append("k", {1});
      store.append("k", {2});
    }
    constexpr int64_t header_bytes =
        sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t) +
        sizeof(int64_t) + sizeof(uint64_t) + sizeof(uint64_t);
    constexpr int64_t footer_bytes = sizeof(uint32_t) + sizeof(uint64_t);
    const int64_t second_offset = header_bytes + 1 + 1 + footer_bytes;
    {
      std::fstream file(corrupt_path,
                        std::ios::binary | std::ios::in | std::ios::out);
      require(file.is_open(), "failed to open corruption fixture");
      file.seekp(second_offset + sizeof(uint32_t) + sizeof(uint32_t) +
                 sizeof(uint64_t));
      file.write(reinterpret_cast<const char*>(&second_offset),
                 sizeof(second_offset));
    }
    bool rejected_corrupt_lineage = false;
    try {
      CausalMemoryStore corrupt(corrupt_path.string());
      (void)corrupt;
    } catch (const std::runtime_error&) {
      rejected_corrupt_lineage = true;
    }
    require(rejected_corrupt_lineage,
            "causal store accepted a self-referential lineage");
    std::filesystem::remove(corrupt_path);

    bool rejected_empty_key = false;
    try {
      CausalMemoryStore store(corrupt_path.string());
      store.append("", {1});
    } catch (const std::invalid_argument&) {
      rejected_empty_key = true;
    }
    std::filesystem::remove(corrupt_path);
    require(rejected_empty_key, "causal store accepted an empty key");

    const auto torn_path =
        std::filesystem::temp_directory_path() / "nsos_memory_causal_torn.bin";
    std::filesystem::remove(torn_path);
    {
      CausalMemoryStore store(torn_path.string());
      store.append("k", {7, 8, 9});
    }
    {
      std::ofstream tail(torn_path, std::ios::binary | std::ios::app);
      const uint8_t partial[] = {0x32, 0x54, 0x53, 0x43, 1, 2, 3};
      tail.write(reinterpret_cast<const char*>(partial), sizeof(partial));
    }
    {
      CausalMemoryStore recovered(torn_path.string());
      const auto latest = recovered.read_latest("k");
      require(latest.has_value() && *latest == std::vector<uint8_t>({7, 8, 9}),
              "causal store did not recover a torn final append");
    }
    std::filesystem::remove(torn_path);

    const auto checksum_path =
        std::filesystem::temp_directory_path() / "nsos_memory_causal_checksum.bin";
    std::filesystem::remove(checksum_path);
    {
      CausalMemoryStore store(checksum_path.string());
      store.append("k", {1, 2, 3});
    }
    {
      std::fstream file(checksum_path,
                        std::ios::binary | std::ios::in | std::ios::out);
      file.seekp(header_bytes + 1);
      const uint8_t corrupt = 0xff;
      file.write(reinterpret_cast<const char*>(&corrupt), 1);
    }
    bool rejected_checksum = false;
    try {
      CausalMemoryStore corrupt(checksum_path.string());
      (void)corrupt;
    } catch (const std::runtime_error&) {
      rejected_checksum = true;
    }
    require(rejected_checksum, "causal store accepted a checksum mismatch");
    std::filesystem::remove(checksum_path);

    const auto locked_path =
        std::filesystem::temp_directory_path() / "nsos_memory_causal_locked.bin";
    std::filesystem::remove(locked_path);
    bool rejected_concurrent_open = false;
    {
      CausalMemoryStore first_handle(locked_path.string());
      try {
        CausalMemoryStore second_handle(locked_path.string());
        (void)second_handle;
      } catch (const std::runtime_error&) {
        rejected_concurrent_open = true;
      }
    }
    require(rejected_concurrent_open,
            "causal store accepted two simultaneous writers");
    std::filesystem::remove(locked_path);

    MemorySystem volatile_memory(4);
    volatile_memory.add_message(first);
    volatile_memory.add_message(second);
    const auto in_memory_history = volatile_memory.recall_recent_messages(2);
    require(in_memory_history.size() == 2, "in-memory recall_recent_messages fallback failed");
    require(in_memory_history[0].content == "world", "in-memory latest message mismatch");
    require(in_memory_history[1].content == "hello\nworld", "in-memory oldest message mismatch");

    std::cout << "Memory causal store test passed!" << std::endl;
    std::filesystem::remove(store_path);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Memory causal store test failed: " << ex.what() << std::endl;
    return 1;
  }
}
