#ifndef SMART_LOADER_H
#define SMART_LOADER_H

#include "tensor.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace nsos {

// Ring Buffer for Async I/O
struct IORequest {
  std::string filepath;
  size_t offset;
  size_t size;
  Tensor *destination; // Pre-allocated destination
  std::atomic<bool> completed;
};

class SmartLoader {
public:
  SmartLoader(size_t buffer_size = 1024 * 1024 * 128); // 128MB
  ~SmartLoader();

  // Async Fetch
  void submit_request(const std::string &path, size_t offset, size_t size,
                      Tensor *dest);

  // Wait for specific request
  void wait_for_completion(Tensor *t);

private:
  std::thread worker_thread;
  std::atomic<bool> running;

  std::queue<IORequest *> request_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;

  void worker_loop();
};

} // namespace nsos

#endif
