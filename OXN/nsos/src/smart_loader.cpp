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

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <io.h>
#define NSOS_OPEN  _open
#define NSOS_CLOSE _close
#define NSOS_O_RDONLY (_O_RDONLY | _O_BINARY)
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
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    running_.store(false, std::memory_order_release);
  }
  queue_cv_.notify_all();
  // No new request can enter after running_ becomes false. The worker drains
  // the already-owned queue before observing shutdown.
  drain();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

std::future<LoadResult> SmartLoader::submit_request(const std::string& path,
                                                    size_t offset,
                                                    size_t size,
                                                    Tensor* dest) {
  auto req = std::make_unique<IORequest>();
  req->filepath = path;
  req->offset = offset;
  req->size = size;
  req->destination = dest;

  std::future<LoadResult> fut = req->promise.get_future();

  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (!running_.load(std::memory_order_acquire)) {
      throw std::runtime_error(
          "SmartLoader cannot accept a request after shutdown");
    }
    const std::size_t prior =
        in_flight_.fetch_add(1, std::memory_order_acq_rel);
    if (prior == (std::numeric_limits<std::size_t>::max)()) {
      in_flight_.fetch_sub(1, std::memory_order_acq_rel);
      throw std::overflow_error("SmartLoader in-flight counter overflow");
    }
    try {
      request_queue_.push(std::move(req));
    } catch (...) {
      in_flight_.fetch_sub(1, std::memory_order_acq_rel);
      if (prior == 0) {
        std::lock_guard<std::mutex> drain_lock(drain_mutex_);
        drain_cv_.notify_all();
      }
      throw;
    }
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
    std::unique_ptr<IORequest> req;

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

      req = std::move(request_queue_.front());
      request_queue_.pop();
    }

    // Perform the read outside the queue lock so other producers can
    // continue submitting concurrently. The outer catch guarantees that an
    // unexpected allocation/runtime exception still fulfills the future and
    // releases the in-flight accounting slot.
    try {
      LoadResult result;

      if (req->destination == nullptr) {
        result.error_msg =
            "[SmartLoader] null destination tensor: " + req->filepath;
        std::cerr << result.error_msg << std::endl;
      } else if (req->destination->get_device() == Device::GPU) {
        // The worker performs a host pread directly into Tensor::data(). A
        // GPU destination would make host I/O target device memory. Reject it.
        result.error_msg =
            "[SmartLoader] GPU destination tensor unsupported: " +
            req->filepath;
        std::cerr << result.error_msg << std::endl;
      } else if (
          req->destination->size < 0 ||
          (req->destination->size > 0 &&
           req->destination->raw_data() == nullptr) ||
          static_cast<uint64_t>(req->destination->size) >
              (std::numeric_limits<size_t>::max)() / sizeof(float)) {
        result.error_msg =
            "[SmartLoader] invalid destination tensor: " + req->filepath;
        std::cerr << result.error_msg << std::endl;
      } else if (req->size >
                 static_cast<size_t>(req->destination->size) * sizeof(float)) {
        // Refuse to read more bytes than the destination tensor can hold.
        result.error_msg =
            "[SmartLoader] read size " + std::to_string(req->size) +
            " exceeds destination capacity " +
            std::to_string(static_cast<size_t>(req->destination->size) *
                           sizeof(float)) +
            " bytes: " + req->filepath;
        std::cerr << result.error_msg << std::endl;
      } else if (req->filepath.empty()) {
        result.error_msg = "[SmartLoader] file path must not be empty";
      } else if (
          req->offset > static_cast<size_t>(
                            (std::numeric_limits<long long>::max)())) {
        result.error_msg =
            "[SmartLoader] file offset exceeds supported range: " +
            req->filepath;
      } else {
        const int fd = NSOS_OPEN(req->filepath.c_str(), NSOS_O_RDONLY);
        if (fd < 0) {
          const int open_errno = errno;
          result.error_msg =
              "[SmartLoader] open failed (errno " +
              std::to_string(open_errno) + ": " +
              std::error_code(open_errno, std::generic_category()).message() +
              "): " + req->filepath;
          std::cerr << result.error_msg << std::endl;
        } else {
          size_t total_read = 0;
          bool read_failed = false;
          while (total_read < req->size) {
            const size_t remaining = req->size - total_read;
            const size_t chunk = (std::min)(
                remaining,
                static_cast<size_t>((std::numeric_limits<int>::max)()));
            if (total_read >
                static_cast<size_t>(
                    (std::numeric_limits<long long>::max)()) - req->offset) {
              result.error_msg =
                  "[SmartLoader] positioned read offset overflow: " +
                  req->filepath;
              read_failed = true;
              break;
            }
            const io_ssize_t bytes = NSOS_PREAD(
                fd,
                reinterpret_cast<unsigned char*>(
                    req->destination->raw_data()) + total_read,
                chunk,
                static_cast<long long>(req->offset + total_read));
            if (bytes < 0) {
              if (errno == EINTR) {
                continue;
              }
              const int read_errno = errno;
              result.error_msg =
                  "[SmartLoader] pread failed (errno " +
                  std::to_string(read_errno) + ": " +
                  std::error_code(read_errno, std::generic_category()).message() +
                  "): " + req->filepath;
              read_failed = true;
              break;
            }
            if (bytes == 0) {
              break;
            }
            total_read += static_cast<size_t>(bytes);
          }
          const int close_status = NSOS_CLOSE(fd);
          if (close_status != 0) {
            const int close_errno = errno;
            const std::string close_error =
                "[SmartLoader] close failed (errno " +
                std::to_string(close_errno) + ": " +
                std::error_code(close_errno, std::generic_category()).message() +
                "): " + req->filepath;
            result.error_msg = result.error_msg.empty()
                                   ? close_error
                                   : result.error_msg + "; " + close_error;
            read_failed = true;
          }
          if (!read_failed && total_read != req->size) {
            result.error_msg =
                "[SmartLoader] short read: requested " +
                std::to_string(req->size) + " bytes, received " +
                std::to_string(total_read) + " bytes: " + req->filepath;
            read_failed = true;
          }

          if (read_failed) {
            std::cerr << result.error_msg << std::endl;
          } else {
            result.bytes_read = static_cast<io_ssize_t>(total_read);
          }
        }
      }

      // Fulfill the promise BEFORE decrementing in_flight_ so a thread
      // racing in drain() cannot observe in_flight_ == 0 and proceed
      // before the result is visible to the original future holder.
      try {
        req->promise.set_value(std::move(result));
      } catch (const std::future_error& e) {
        // A no-state or already-satisfied promise violates the queue ownership
        // contract. Log it, but keep draining unrelated requests.
        std::cerr << "[SmartLoader] promise.set_value failed: " << e.what()
                  << std::endl;
      }
    } catch (...) {
      try {
        req->promise.set_exception(std::current_exception());
      } catch (const std::future_error& error) {
        std::cerr << "[SmartLoader] promise.set_exception failed: "
                  << error.what() << std::endl;
      }
    }

    const std::size_t prior =
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    if (prior == 0) {
      in_flight_.store(0, std::memory_order_release);
      std::cerr << "[SmartLoader] in-flight accounting underflow" << std::endl;
      std::terminate();
    }
    const std::size_t remaining = prior - 1;
    if (remaining == 0) {
      // Notify *all* drainers — there may be several barriers waiting.
      std::lock_guard<std::mutex> lk(drain_mutex_);
      drain_cv_.notify_all();
    }
  }
}

}  // namespace nsos
