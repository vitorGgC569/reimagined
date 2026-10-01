// SmartLoader async semantics test.
//
// Validates the contract documented in include/smart_loader.h:
//   * submit_request returns a future that resolves with a LoadResult
//   * future blocks until the read actually completes (no sleep games)
//   * drain() blocks until every in-flight request finishes
//   * concurrent producers can submit safely
//   * destructor flushes pending promises rather than abandoning them
//
// The previous wait_for_completion(Tensor*) implementation slept for
// one millisecond regardless of state, which silently passed when the
// disk was fast and silently truncated reads when it was slow.  This
// test fails outright on either failure mode.

#include "smart_loader.h"
#include "tensor.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(std::string("CHECK failed: ") + msg);
  }
}

// Writes a deterministic float pattern to disk so we can read it back
// and verify the read landed in the destination tensor untouched.
std::string write_temp_floats(const std::string& tag, std::size_t count) {
  auto path = std::filesystem::temp_directory_path() /
              ("nsos_smart_loader_" + tag + ".bin");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open temp file: " + path.string());
  }
  for (std::size_t i = 0; i < count; ++i) {
    const float v = static_cast<float>(i) * 0.5f - 1.25f;
    out.write(reinterpret_cast<const char*>(&v), sizeof(float));
  }
  out.close();
  return path.string();
}

void test_binary_mode_preserves_control_bytes() {
  const std::vector<unsigned char> expected{0x0d, 0x0a, 0x1a, 0x00};
  const auto path = std::filesystem::temp_directory_path() /
                    "nsos_smart_loader_binary_mode.bin";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(expected.size()));
  }

  nsos::SmartLoader loader;
  nsos::Tensor destination({1}, nsos::Device::CPU);
  const nsos::LoadResult result =
      loader.submit_request(path.string(), 0, expected.size(), &destination)
          .get();
  check(result.success(), "binary control-byte read must succeed");
  check(std::memcmp(destination.raw_data(), expected.data(), expected.size()) ==
            0,
        "SmartLoader must not apply text-mode byte translation");
  std::error_code error;
  std::filesystem::remove(path, error);
  std::cout << "[smart_loader] binary_mode PASS" << std::endl;
}

void test_single_read() {
  constexpr std::size_t N = 256;
  const std::string path = write_temp_floats("single", N);

  nsos::SmartLoader loader;
  nsos::Tensor dest({static_cast<int>(N)}, nsos::Device::CPU);

  auto fut = loader.submit_request(path, /*offset=*/0,
                                   /*size=*/N * sizeof(float), &dest);
  check(fut.valid(), "future returned by submit_request must be valid");

  const nsos::LoadResult result = fut.get();
  check(result.success(),
        ("LoadResult must be success: " + result.error_msg).c_str());
  check(result.bytes_read >= 0 &&
            static_cast<std::size_t>(result.bytes_read) == N * sizeof(float),
        "bytes_read must equal requested size");

  for (std::size_t i = 0; i < N; ++i) {
    const float expected = static_cast<float>(i) * 0.5f - 1.25f;
    const float got = dest.data()[i];
    if (std::abs(expected - got) > 1e-6f) {
      throw std::runtime_error("mismatch at " + std::to_string(i) +
                                " expected=" + std::to_string(expected) +
                                " got=" + std::to_string(got));
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::cout << "[smart_loader] single_read PASS" << std::endl;
}

void test_offset_read() {
  constexpr std::size_t N = 512;
  constexpr std::size_t OFFSET_FLOATS = 128;
  constexpr std::size_t READ_FLOATS = 64;

  const std::string path = write_temp_floats("offset", N);

  nsos::SmartLoader loader;
  nsos::Tensor dest({static_cast<int>(READ_FLOATS)}, nsos::Device::CPU);

  auto fut = loader.submit_request(
      path, /*offset=*/OFFSET_FLOATS * sizeof(float),
      /*size=*/READ_FLOATS * sizeof(float), &dest);

  const nsos::LoadResult result = fut.get();
  check(result.success(), "offset read must succeed");

  for (std::size_t i = 0; i < READ_FLOATS; ++i) {
    const std::size_t source_index = OFFSET_FLOATS + i;
    const float expected = static_cast<float>(source_index) * 0.5f - 1.25f;
    const float got = dest.data()[i];
    if (std::abs(expected - got) > 1e-6f) {
      throw std::runtime_error("offset mismatch at " + std::to_string(i));
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::cout << "[smart_loader] offset_read PASS" << std::endl;
}

void test_missing_file_yields_error() {
  nsos::SmartLoader loader;
  nsos::Tensor dest({16}, nsos::Device::CPU);

  auto fut = loader.submit_request(
      "/nonexistent/path/that/should/not/exist_xyz.bin",
      /*offset=*/0, /*size=*/16 * sizeof(float), &dest);

  const nsos::LoadResult result = fut.get();
  check(!result.success(), "missing-file read must NOT succeed");
  check(!result.error_msg.empty(), "error_msg must be populated on failure");
  std::cout << "[smart_loader] missing_file PASS (error="
            << result.error_msg << ")" << std::endl;
}

void test_short_read_yields_error() {
  constexpr std::size_t N = 8;
  const std::string path = write_temp_floats("short", N);
  nsos::SmartLoader loader;
  nsos::Tensor dest({static_cast<int>(N + 1)}, nsos::Device::CPU);
  auto fut = loader.submit_request(
      path, 0, (N + 1) * sizeof(float), &dest);
  const nsos::LoadResult result = fut.get();
  check(!result.success(), "short read must fail closed");
  check(result.bytes_read < 0,
        "short read must not publish a successful byte count");
  check(result.error_msg.find("short read") != std::string::npos,
        "short read must publish an actionable diagnostic");
  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::cout << "[smart_loader] short_read PASS" << std::endl;
}

void test_destructor_drains_owned_request() {
  constexpr std::size_t N = 128;
  const std::string path = write_temp_floats("destructor", N);
  nsos::Tensor dest({static_cast<int>(N)}, nsos::Device::CPU);
  std::future<nsos::LoadResult> future;
  {
    nsos::SmartLoader loader;
    future = loader.submit_request(path, 0, N * sizeof(float), &dest);
  }
  check(future.wait_for(std::chrono::milliseconds(0)) ==
            std::future_status::ready,
        "SmartLoader destructor must fulfill every owned request");
  check(future.get().success(),
        "request drained by SmartLoader destructor must succeed");
  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::cout << "[smart_loader] destructor_drain PASS" << std::endl;
}

void test_drain_with_many_requests() {
  constexpr std::size_t N_FILES = 16;
  constexpr std::size_t N = 64;

  std::vector<std::string> paths;
  paths.reserve(N_FILES);
  for (std::size_t i = 0; i < N_FILES; ++i) {
    paths.push_back(write_temp_floats("drain_" + std::to_string(i), N));
  }

  nsos::SmartLoader loader;
  std::vector<nsos::Tensor> dests;
  dests.reserve(N_FILES);
  std::vector<std::future<nsos::LoadResult>> futures;
  futures.reserve(N_FILES);

  for (std::size_t i = 0; i < N_FILES; ++i) {
    dests.emplace_back(std::vector<int>{static_cast<int>(N)}, nsos::Device::CPU);
  }
  for (std::size_t i = 0; i < N_FILES; ++i) {
    futures.push_back(loader.submit_request(
        paths[i], /*offset=*/0, /*size=*/N * sizeof(float), &dests[i]));
  }

  loader.drain();

  // After drain, every future must be ready immediately (no blocking).
  for (std::size_t i = 0; i < N_FILES; ++i) {
    const auto status = futures[i].wait_for(std::chrono::milliseconds(0));
    check(status == std::future_status::ready,
          "after drain(), all futures must be ready without further waiting");
    const nsos::LoadResult r = futures[i].get();
    check(r.success(),
          ("drained future must be success: " + r.error_msg).c_str());
  }

  for (const auto& path : paths) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::cout << "[smart_loader] drain_many PASS" << std::endl;
}

void test_concurrent_producers() {
  constexpr std::size_t N_THREADS = 4;
  constexpr std::size_t REQUESTS_PER_THREAD = 8;
  constexpr std::size_t N = 32;

  std::vector<std::string> paths;
  paths.reserve(N_THREADS * REQUESTS_PER_THREAD);
  for (std::size_t i = 0; i < N_THREADS * REQUESTS_PER_THREAD; ++i) {
    paths.push_back(write_temp_floats("conc_" + std::to_string(i), N));
  }

  nsos::SmartLoader loader;
  std::vector<nsos::Tensor> dests;
  dests.reserve(N_THREADS * REQUESTS_PER_THREAD);
  for (std::size_t i = 0; i < N_THREADS * REQUESTS_PER_THREAD; ++i) {
    dests.emplace_back(std::vector<int>{static_cast<int>(N)}, nsos::Device::CPU);
  }

  std::atomic<std::size_t> success_count{0};
  std::vector<std::thread> producers;
  producers.reserve(N_THREADS);

  for (std::size_t t = 0; t < N_THREADS; ++t) {
    producers.emplace_back([&, t] {
      std::vector<std::future<nsos::LoadResult>> local;
      for (std::size_t r = 0; r < REQUESTS_PER_THREAD; ++r) {
        const std::size_t idx = t * REQUESTS_PER_THREAD + r;
        local.push_back(loader.submit_request(
            paths[idx], /*offset=*/0, /*size=*/N * sizeof(float),
            &dests[idx]));
      }
      for (auto& fut : local) {
        if (fut.get().success()) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& th : producers) th.join();

  check(success_count.load() == N_THREADS * REQUESTS_PER_THREAD,
        "all concurrent reads must succeed");

  for (const auto& path : paths) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::cout << "[smart_loader] concurrent PASS" << std::endl;
}

}  // namespace

int main() {
  try {
    test_single_read();
    test_binary_mode_preserves_control_bytes();
    test_offset_read();
    test_missing_file_yields_error();
    test_short_read_yields_error();
    test_destructor_drains_owned_request();
    test_drain_with_many_requests();
    test_concurrent_producers();
    std::cout << "[smart_loader] ALL PASS" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[smart_loader] FAIL: " << ex.what() << std::endl;
    return 1;
  }
}
