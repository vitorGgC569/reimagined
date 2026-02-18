#include "smart_loader.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>

#ifdef _WIN32
#include <BaseTsd.h>
#include <io.h>
typedef SSIZE_T ssize_t;
#define open _open
#define close _close
#define O_RDONLY _O_RDONLY
// Fallback pread for Windows (uses _lseeki64 + _read, not atomic but
// functional)
inline ssize_t pread(int fd, void *buf, size_t count, long long offset) {
  _lseeki64(fd, offset, SEEK_SET);
  return _read(fd, buf, (unsigned int)count);
}
#else
#include <unistd.h>
#endif

// Simulating io_uring using pread in a worker thread
// This provides the API contract for "Zero Copy" (direct to pointer)
// even if the underlying syscall is POSIX.

namespace nsos {

SmartLoader::SmartLoader(size_t buffer_size) : running(true) {
  worker_thread = std::thread(&SmartLoader::worker_loop, this);
}

SmartLoader::~SmartLoader() {
  running = false;
  queue_cv.notify_all();
  if (worker_thread.joinable())
    worker_thread.join();
}

void SmartLoader::submit_request(const std::string &path, size_t offset,
                                 size_t size, Tensor *dest) {
  IORequest *req = new IORequest{path, offset, size, dest, false};
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    request_queue.push(req);
  }
  queue_cv.notify_one();
}

void SmartLoader::wait_for_completion(Tensor *t) {
  // In a real io_uring, we'd poll completion queue.
  // Here we spin-wait on the request associated with tensor t.
  // Implementation constraint: We assume 't' is unique or caller manages the
  // request mapping. For MVP: Block until queue empty? No. We just sleep
  // briefly. Real implementation requires request ID return.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void SmartLoader::worker_loop() {
  while (running) {
    IORequest *req = nullptr;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      queue_cv.wait(lock,
                    [this] { return !request_queue.empty() || !running; });
      if (!running && request_queue.empty())
        return;
      req = request_queue.front();
      request_queue.pop();
    }

    if (req) {
      int fd = open(req->filepath.c_str(), O_RDONLY);
      if (fd >= 0) {
        // Read directly into tensor memory (Zero-Copy-ish)
        // If Tensor is GPU, this would require cudaHostRegister or direct pread
        // to managed memory. Assuming CPU Tensor or Unified Memory for now.

        ssize_t bytes =
            pread(fd, req->destination->data(), req->size, req->offset);
        if (bytes < 0) {
          std::cerr << "[SmartLoader] Read Error: " << req->filepath
                    << std::endl;
        }
        close(fd);
      } else {
        std::cerr << "[SmartLoader] Open Error: " << req->filepath << std::endl;
      }
      req->completed = true;
      // Clean up request struct logic would go here (callback or future)
      delete req;
    }
  }
}

} // namespace nsos
