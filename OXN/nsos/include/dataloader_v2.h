#ifndef DATALOADER_V2_H
#define DATALOADER_V2_H

#include "tensor.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>


class DataLoader {
public:
  DataLoader(const std::string &bin_path, int batch_size, int d_model,
             int max_queue = 5);
  ~DataLoader();

  // Returns true if a batch was loaded, false if EOF
  bool next(nsos::Tensor &batch);

private:
  void worker_loop();

  std::string path;
  int batch_size;
  int d_model;
  size_t max_queue;

  std::thread worker;
  std::atomic<bool> stop_flag;

  std::queue<nsos::Tensor> queue;
  std::mutex mutex;
  std::condition_variable not_empty;
  std::condition_variable not_full;

  // Mock file pointer
  size_t current_idx;
  size_t total_samples;
};

#endif
