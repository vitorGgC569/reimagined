// SmartLoader: real async positioned-read I/O via a single worker
// thread.  Replaces the previous implementation which fulfilled neither
// "async" nor "wait" honestly: the old wait_for_completion(Tensor*)
// slept for one millisecond regardless of whether the read was actually
// finished, and the IORequest::completed flag was never observed by any
// API surface.
//
// The new design uses std::promise / std::future so callers receive a
// real synchronization handle and a structured LoadResult on completion.
// drain() lets shutdown and checkpoint barriers wait for the queue to
// flush without polling.
//
// File I/O strategy:
//   * On POSIX we use pread(2) which is atomic per-call.
//   * On Windows we wrap _lseeki64 + _read (not atomic, but the worker
//     is single-threaded so no race exists).  Each request opens and
//     closes the file independently; the request rate is low enough
//     that connection caching would be premature optimization.

#include "smart_loader.h"

#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <utility>

#ifdef _WIN32
#include <io.h>
#define NSOS_OPEN  _open
#define NSOS_CLOSE _close
#define NSOS_O_RDONLY _O_RDONLY
namespace {
nsos::io_ssize_t pread_compat(int fd, void* buf, size_t count,
                               long long offset) {
  if (_lseeki64(fd, offset, SEEK_SET) < 0) {
    return -1;
  }
  return static_cast<nsos::io_ssize_t>(
      _read(fd, buf, static_cast<unsigned int>(count)));
}
}  // namespace
#define NSOS_PREAD pread_compat
#else
#include <unistd.h>
#include <sys/types.h>
#define NSOS_OPEN  ::open
#define NSOS_CLOSE ::close
#define NSOS_O_RDONLY O_RDONLY
namespace {
nsos::io_ssize_t pread_compat(int fd, void* buf, size_t count,
                               long long offset) {
  return static_cast<nsos::io_ssize_t>(
      ::pread(fd, buf, count, static_cast<off_t>(offset)));
}
}  // namespace
#define NSOS_PREAD pread_compat
#endif

namespace nsos {

SmartLoader::SmartLoader(size_t /*buffer_size*/) {
  worker_thread_ = std::thread(&SmartLoader::worker_loop, this);
}

SmartLoader::~SmartLoader() {
  // Drain first so promises in the queue are fulfilled instead of
  // abandoned (which would surface as broken_promise on future::get()
  // in any caller still holding a future).
  drain();

  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    running_.store(false, std::memory_order_release);
  }
  queue_cv_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

std::future<LoadResult> SmartLoader::submit_request(const std::string& path,
                                                    size_t offset,
                                                    size_t size,
                                                    Tensor* dest) {
  // We allocate IORequest on the heap so the promise survives across
  // the worker's lifetime; ownership transfers to the worker which
  // will delete it after fulfilling the promise.
  IORequest* req = new IORequest{};
  req->filepath = path;
  req->offset = offset;
  req->size = size;
  req->destination = dest;

  std::future<LoadResult> fut = req->promise.get_future();

  // Increment in_flight_ BEFORE pushing so that drain() cannot observe
  // an empty queue + in_flight_ == 0 race window.
  in_flight_.fetch_add(1, std::memory_order_acq_rel);

  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    request_queue_.push(req);
  }
  queue_cv_.notify_one();

  return fut;
}

void SmartLoader::drain() {
  std::unique_lock<std::mutex> lk(drain_mutex_);
  drain_cv_.wait(lk, [this] {
    return in_flight_.load(std::memory_order_acquire) == 0;
  });
}

void SmartLoader::worker_loop() {
  while (true) {
    IORequest* req = nullptr;

    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [this] {
        return !request_queue_.empty() ||
               !running_.load(std::memory_order_acquire);
      });

      if (request_queue_.empty()) {
        // Shutdown path: queue drained, running_ is false.  Exit cleanly.
        return;
      }

      req = request_queue_.front();
      request_queue_.pop();
    }

    // Perform the read outside the queue lock so other producers can
    // continue submitting concurrently.
    LoadResult result;

    if (req->destination == nullptr) {
      result.error_msg = "[SmartLoader] null destination tensor: " + req->filepath;
      std::cerr << result.error_msg << std::endl;
    } else if (req->destination->get_device() == Device::GPU) {
      // The worker performs a host pread directly into Tensor::data().  A
      // GPU-resident destination would have the kernel write into device
      // memory through a host pointer -> memory corruption.  Reject instead.
      result.error_msg =
          "[SmartLoader] GPU destination tensor unsupported: " + req->filepath;
      std::cerr << result.error_msg << std::endl;
    } else if (req->size >
               static_cast<size_t>(req->destination->size) * sizeof(float)) {
      // Refuse to read more bytes than the destination tensor can hold; a
      // positioned read into a too-small buffer is a heap overflow.
      result.error_msg =
          "[SmartLoader] read size " + std::to_string(req->size) +
          " exceeds destination capacity " +
          std::to_string(static_cast<size_t>(req->destination->size) *
                         sizeof(float)) +
          " bytes: " + req->filepath;
      std::cerr << result.error_msg << std::endl;
    } else {
      const int fd = NSOS_OPEN(req->filepath.c_str(), NSOS_O_RDONLY);
      if (fd < 0) {
        result.error_msg = "[SmartLoader] open failed: " + req->filepath;
        std::cerr << result.error_msg << std::endl;
      } else {
        const io_ssize_t bytes = NSOS_PREAD(
            fd, req->destination->data(), req->size,
            static_cast<long long>(req->offset));
        NSOS_CLOSE(fd);

        if (bytes < 0) {
          result.error_msg = "[SmartLoader] pread failed: " + req->filepath;
          std::cerr << result.error_msg << std::endl;
        } else {
          result.bytes_read = bytes;
        }
      }
    }

    // Fulfill the promise BEFORE decrementing in_flight_ so a thread
    // racing in drain() cannot observe in_flight_ == 0 and proceed
    // before the result is visible to the original future holder.
    try {
      req->promise.set_value(std::move(result));
    } catch (const std::future_error& e) {
      // Promise was already satisfied or future was destroyed; log but
      // do not propagate so the worker stays alive.
      std::cerr << "[SmartLoader] promise.set_value failed: " << e.what()
                << std::endl;
    }

    delete req;

    const std::size_t remaining =
        in_flight_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
      // Notify *all* drainers — there may be several barriers waiting.
      std::lock_guard<std::mutex> lk(drain_mutex_);
      drain_cv_.notify_all();
    }
  }
}

}  // namespace nsos
