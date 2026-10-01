#ifndef SMART_LOADER_H
#define SMART_LOADER_H

#include "tensor.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace nsos {

// Signed type wide enough to hold a positioned-read return value on
// every supported platform (POSIX ssize_t and Windows SSIZE_T both fit
// in 64 bits on the architectures we ship).  Defined inside the nsos
// namespace to avoid polluting the global scope.
using io_ssize_t = std::int64_t;

// Result of a single async load.  `bytes_read >= 0` indicates a real read;
// negative values plus a non-empty `error_msg` indicate failure. A successful
// result always represents the complete byte range requested.
struct LoadResult {
  io_ssize_t bytes_read = -1;
  std::string error_msg;

  bool success() const noexcept {
    return bytes_read >= 0 && error_msg.empty();
  }
};

// Internal request node — owned by the worker thread until the promise
// is fulfilled.  Not part of the public API; exposed in the header only
// so the queue's unique_ptr value type is complete in the class body.
struct IORequest {
  std::string filepath;
  size_t offset = 0;
  size_t size = 0;
  Tensor* destination = nullptr;
  std::promise<LoadResult> promise;
};

// SmartLoader: real asynchronous file I/O via a dedicated worker thread.
//
// Design contract:
//   * `submit_request` enqueues a positioned read and returns a future
//     that resolves when the read completes (success OR failure).
//   * The destination Tensor must outlive the returned future; the
//     worker writes directly into Tensor::data() (zero-copy).
//   * `drain()` blocks until every in-flight request has completed.
//     Safe to call concurrently with submit_request; useful in
//     checkpoint barriers and the destructor's pre-stop drain.
//   * The destructor rejects new work, lets the worker drain every request it
//     already owns, and only then joins it; no owned promise is abandoned.
//
// Threading:
//   * One internal worker thread services the FIFO request queue.
//   * Multiple producers may call submit_request concurrently.
//   * A single std::mutex guards the request queue; in_flight count
//     uses std::atomic plus a separate condvar for drain coordination.
//
// The class is non-copyable and non-movable because it owns a thread.
class SmartLoader {
 public:
  // `buffer_size` is retained for source-compatibility with the previous
  // signature.  It is not used by the current implementation: the worker
  // reads directly into the caller-supplied Tensor and does not maintain an
  // internal pool. Passing any value therefore has identical semantics.
  explicit SmartLoader(size_t buffer_size = 1024 * 1024 * 128);
  ~SmartLoader();

  SmartLoader(const SmartLoader&) = delete;
  SmartLoader& operator=(const SmartLoader&) = delete;
  SmartLoader(SmartLoader&&) = delete;
  SmartLoader& operator=(SmartLoader&&) = delete;

  // Enqueues an async positioned read.  Returns a future that resolves
  // once the worker has performed the read (or recorded a failure).
  // Caller is responsible for keeping `*dest` alive until the future
  // is ready.
  std::future<LoadResult> submit_request(const std::string& path,
                                         size_t offset,
                                         size_t size,
                                         Tensor* dest);

  // Blocks until all currently in-flight requests have completed.
  // Newly submitted requests after the call begins may also be drained
  // if they finish before the wait predicate is rechecked, but this is
  // a "best-effort barrier" rather than a strict snapshot.
  void drain();

 private:
  std::thread worker_thread_;
  std::atomic<bool> running_{true};

  std::queue<std::unique_ptr<IORequest>> request_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  std::atomic<std::size_t> in_flight_{0};
  std::mutex drain_mutex_;
  std::condition_variable drain_cv_;

  void worker_loop();
};

}  // namespace nsos

#endif  // SMART_LOADER_H
