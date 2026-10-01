#include "dataloader_v2.h"
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>


using namespace nsos;

DataLoader::DataLoader(const std::string &p, int bs, int dm, int mq)
    : path(p), batch_size(bs), d_model(dm), max_queue(static_cast<size_t>(mq)),
      stop_flag(false) {
  if (bs <= 0 || dm <= 0 || mq <= 0 ||
      static_cast<size_t>(bs) >
          static_cast<size_t>((std::numeric_limits<int>::max)()) /
              static_cast<size_t>(dm)) {
    throw std::invalid_argument(
        "DataLoader dimensions and max_queue must be positive and bounded");
  }
  if (path.empty()) {
    throw std::invalid_argument("DataLoader path must not be empty");
  }

  const size_t elements =
      static_cast<size_t>(batch_size) * static_cast<size_t>(d_model);
  if (elements > (std::numeric_limits<size_t>::max)() / sizeof(float)) {
    throw std::overflow_error("DataLoader batch byte size overflow");
  }
  const size_t batch_bytes = elements * sizeof(float);
  if (batch_bytes >
      static_cast<size_t>((std::numeric_limits<std::streamsize>::max)())) {
    throw std::overflow_error("DataLoader batch exceeds stream limits");
  }

  std::ifstream probe(path, std::ios::binary | std::ios::ate);
  if (!probe.is_open()) {
    throw std::runtime_error("DataLoader could not open dataset: " + path);
  }
  const std::streampos end = probe.tellg();
  if (end < std::streampos(0)) {
    throw std::runtime_error("DataLoader could not determine dataset size: " +
                             path);
  }
  const std::streamoff end_offset = end - std::streampos(0);
  if (end_offset < 0) {
    throw std::runtime_error("DataLoader reported a negative dataset size: " +
                             path);
  }
  const auto file_bytes = static_cast<std::uintmax_t>(end_offset);
  if (file_bytes == 0U) {
    throw std::runtime_error("DataLoader dataset is empty: " + path);
  }
  if (file_bytes % static_cast<std::uintmax_t>(batch_bytes) != 0U) {
    throw std::runtime_error(
        "DataLoader dataset is truncated or not batch-aligned: " + path);
  }
  worker = std::thread(&DataLoader::worker_loop, this);
}

DataLoader::~DataLoader() {
  stop_flag.store(true, std::memory_order_release);
  not_full.notify_all();
  not_empty.notify_all();
  if (worker.joinable())
    worker.join();
}

bool DataLoader::next(nsos::Tensor &batch) {
  std::unique_lock<std::mutex> lock(mutex);
  not_empty.wait(lock, [this]() {
    return !queue.empty() || stop_flag.load(std::memory_order_acquire);
  });

  if (queue.empty() && stop_flag.load(std::memory_order_acquire)) {
    const std::exception_ptr error = worker_error;
    lock.unlock();
    if (error) {
      std::rethrow_exception(error);
    }
    return false;
  }

  batch = std::move(queue.front());
  queue.pop();
  not_full.notify_one();
  return true;
}

void DataLoader::worker_loop() {
  const auto finish = [this](std::exception_ptr error = nullptr) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      worker_error = std::move(error);
      stop_flag.store(true, std::memory_order_release);
    }
    not_empty.notify_all();
    not_full.notify_all();
  };

  try {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("DataLoader dataset disappeared before read: " +
                               path);
    }

    const size_t elements =
        static_cast<size_t>(batch_size) * static_cast<size_t>(d_model);
    const auto stream_bytes =
        static_cast<std::streamsize>(elements * sizeof(float));
    while (!stop_flag.load(std::memory_order_acquire)) {
      nsos::Tensor t({batch_size, d_model}, Device::CPU);
      file.read(reinterpret_cast<char *>(t.data()), stream_bytes);
      const std::streamsize read_bytes = file.gcount();
      if (read_bytes == 0 && file.eof()) {
        break;
      }
      if (read_bytes != stream_bytes) {
        throw std::runtime_error(
            "DataLoader encountered a truncated dataset batch: " + path);
      }

      {
        std::unique_lock<std::mutex> lock(mutex);
        not_full.wait(lock, [this]() {
          return queue.size() < max_queue ||
                 stop_flag.load(std::memory_order_acquire);
        });
        if (stop_flag.load(std::memory_order_acquire)) {
          break;
        }
        queue.push(std::move(t));
      }
      not_empty.notify_one();
    }
    if (file.bad()) {
      throw std::runtime_error("DataLoader I/O failure while reading: " + path);
    }
    finish();
  } catch (...) {
    finish(std::current_exception());
  }
}
