#include "dataloader_v2.h"
#include <chrono>
#include <fstream>
#include <iostream>


using namespace nsos;

DataLoader::DataLoader(const std::string &p, int bs, int dm, int mq)
    : path(p), batch_size(bs), d_model(dm), stop_flag(false), current_idx(0),
      total_samples(10000) {
  worker = std::thread(&DataLoader::worker_loop, this);
}

DataLoader::~DataLoader() {
  stop_flag = true;
  not_full.notify_all();
  if (worker.joinable())
    worker.join();
}

bool DataLoader::next(nsos::Tensor &batch) {
  std::unique_lock<std::mutex> lock(mutex);
  not_empty.wait(lock, [this]() { return !queue.empty() || stop_flag; });

  if (queue.empty() && stop_flag)
    return false;

  batch = std::move(queue.front());
  queue.pop();
  not_full.notify_one();
  return true;
}

void DataLoader::worker_loop() {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    while (!stop_flag && current_idx < total_samples) {
      nsos::Tensor t = nsos::Tensor::random({batch_size, d_model}, Device::CPU);
      {
        std::unique_lock<std::mutex> lock(mutex);
        not_full.wait(lock, [this]() { return queue.size() < 5 || stop_flag; });
        if (stop_flag)
          break;
        queue.push(std::move(t));
        current_idx += batch_size;
      }
      not_empty.notify_one();
    }
    return;
  }

  while (!stop_flag && file.good()) {
    int num_floats = batch_size * d_model;
    nsos::Tensor t({batch_size, d_model}, Device::CPU);
    file.read(reinterpret_cast<char *>(t.data()), num_floats * sizeof(float));
    int read_count = static_cast<int>(file.gcount() / sizeof(float));

    if (read_count < num_floats)
      break;

    {
      std::unique_lock<std::mutex> lock(mutex);
      not_full.wait(lock, [this]() { return queue.size() < 16 || stop_flag; });
      if (stop_flag)
        break;
      queue.push(std::move(t));
      current_idx += batch_size;
    }
    not_empty.notify_one();
  }
  stop_flag = true;
  not_empty.notify_all();
}
