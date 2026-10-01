#ifndef DATALOADER_V2_H
#define DATALOADER_V2_H

#include "tensor.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <string>
#include <thread>


class DataLoader {
public:
  DataLoader(const std::string &bin_path, int batch_size, int d_model,
             int max_queue = 5);
  ~DataLoader();

  // Returns true if a complete batch was loaded and false at clean EOF.
  // Missing, truncated and failed reads throw instead of synthesizing data.
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

  std::exception_ptr worker_error;
};

#endif
