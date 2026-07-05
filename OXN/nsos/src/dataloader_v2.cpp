#include "dataloader_v2.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>


using namespace nsos;

DataLoader::DataLoader(const std::string &p, int bs, int dm, int mq)
    : path(p), batch_size(bs), d_model(dm), max_queue(static_cast<size_t>(mq)),
      stop_flag(false), current_idx(0), total_samples(10000) {
  if (bs <= 0 || dm <= 0 || mq <= 0 ||
      static_cast<size_t>(bs) >
          static_cast<size_t>((std::numeric_limits<int>::max)()) /
              static_cast<size_t>(dm)) {
    throw std::invalid_argument(
        "DataLoader dimensions and max_queue must be positive and bounded");
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

  if (queue.empty() && stop_flag.load(std::memory_order_acquire))
    return false;

  batch = std::move(queue.front());
  queue.pop();
  not_full.notify_one();
  return true;
}

void DataLoader::worker_loop() {
  const auto finish = [this]() {
    stop_flag.store(true, std::memory_order_release);
    not_empty.notify_all();
    not_full.notify_all();
  };
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    while (!stop_flag.load(std::memory_order_acquire) &&
           current_idx < total_samples) {
      nsos::Tensor t = nsos::Tensor::random({batch_size, d_model}, Device::CPU);
      {
        std::unique_lock<std::mutex> lock(mutex);
        not_full.wait(lock, [this]() {
          return queue.size() < max_queue ||
                 stop_flag.load(std::memory_order_acquire);
        });
        if (stop_flag.load(std::memory_order_acquire))
          break;
        queue.push(std::move(t));
        current_idx += batch_size;
      }
      not_empty.notify_one();
    }
    finish();
    return;
  }

  while (!stop_flag.load(std::memory_order_acquire) && file.good()) {
    int num_floats = batch_size * d_model;
    nsos::Tensor t({batch_size, d_model}, Device::CPU);
    file.read(reinterpret_cast<char *>(t.data()), num_floats * sizeof(float));
    int read_count = static_cast<int>(file.gcount() / sizeof(float));

    if (read_count < num_floats)
      break;

    {
      std::unique_lock<std::mutex> lock(mutex);
      not_full.wait(lock, [this]() {
        return queue.size() < max_queue ||
               stop_flag.load(std::memory_order_acquire);
      });
      if (stop_flag.load(std::memory_order_acquire))
        break;
      queue.push(std::move(t));
      current_idx += batch_size;
    }
    not_empty.notify_one();
  }
  finish();
}
