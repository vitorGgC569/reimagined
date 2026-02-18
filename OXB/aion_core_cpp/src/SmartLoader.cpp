#include "../include/SmartLoader.hpp"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include "../include/RingBuffer.hpp"

// Mock for libcufile.h if not present
// In a real environment, we would include <cufile.h>
#ifdef HAS_CUFILE
#include <cufile.h>
#else
// Stubs for compilation
typedef void* CUfileHandle_t;
typedef void* CUfileError_t;
#endif

namespace Aion {

    SmartLoader::SmartLoader() {
        // Detect capabilities
        // 1. Check for io_uring (Kernel version check or try init)
        // We assume available if Linux for this prototype
#ifdef AION_PLATFORM_LINUX
        uring_available = true;
#else
        uring_available = false;
#endif

        // 2. Check for GDS (libcufile)
        // Dynamically load library or check compile flag
        // Here we just use the mocked architecture flag
        gds_available = false; // Default to false in sandbox
#ifdef HAS_CUFILE
        gds_available = true;
        // cuFileDriverOpen();
#endif
    }

    SmartLoader::~SmartLoader() {
#ifdef HAS_CUFILE
        // cuFileDriverClose();
#endif
    }

    bool SmartLoader::has_gpu_direct() const { return gds_available; }
    bool SmartLoader::has_io_uring() const { return uring_available; }

    void SmartLoader::register_buffer(void* ptr, size_t size, bool is_gpu_ptr) {
        if (gds_available && is_gpu_ptr) {
            // cuFileBufRegister(ptr, size, 0);
            std::cout << "[GDS] Registered GPU buffer at " << ptr << std::endl;
        }
    }

    void SmartLoader::unregister_buffer(void* ptr) {
        if (gds_available) {
            // cuFileBufDeregister(ptr);
        }
    }

    size_t SmartLoader::load(const std::string& filename, uint64_t offset, size_t size, void* target_ptr) {
        // Determine path
        // 1. GDS Path
        if (gds_available) {
            // Check if pointer is GPU (tracked in map usually, or passed in arg in a real API)
            // For prototype, let's assume if GDS is on, we try.
            // int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
            // cuFileHandle_t cf_handle;
            // cuFileHandleRegister(&cf_handle, &descr);
            // cuFileRead(cf_handle, target_ptr, size, offset, 0);
            return size; // Mock success
        }

        // 2. IO_URING / CPU Path
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd < 0) return 0;

        if (uring_available) {
            // Use our RingBuffer
            // Note: RingBuffer is async, SmartLoader::load implies sync?
            // Usually SmartLoader would submit and return a future.
            // For this synchronous signature, we submit and wait.
            RingBuffer ring(16);
            ring.submit_read(fd, target_ptr, size, offset, 1);
            ring.process_sq(); // Submit

            // Poll for completion
            uint64_t udata;
            int res;
            while (!ring.peek_cq(udata, res)) {
                // Efficient wait: Avoid 100% CPU usage
                // We could use io_uring_wait_cqe but RingBuffer abstraction hides it.
                // For now, simple yield is better than spinlock.
                // In production, RingBuffer should expose wait().
                // Assuming peek_cq is non-blocking.

                // Add sleep/yield to fix busy loop complaint
                usleep(10); // 10 microseconds latency penalty for CPU saving
            }
            close(fd);
            return res;
        } else {
            // 3. Fallback Pread
            ssize_t bytes = pread(fd, target_ptr, size, offset);
            close(fd);
            return (bytes > 0) ? bytes : 0;
        }
    }

    // --- AionDataLoader Implementation ---

    AionDataLoader::AionDataLoader(size_t buffer_sz) : buffer_size(buffer_sz), cursor(0) {
        buffer.resize(buffer_size);
    }

    void AionDataLoader::open(const std::string& filename) {
        current_file = filename;
        cursor = 0;
        // Pre-fill buffer
        loader.load(filename, 0, buffer_size, buffer.data());
    }

    std::pair<void*, size_t> AionDataLoader::next_batch(size_t batch_size_bytes) {
        if (cursor + batch_size_bytes > buffer_size) {
            // Simplistic rewind or refill
            // For prototype: Refill from start (Loop)
            cursor = 0;
            // Real impl: Stream next chunk from file
        }

        void* ptr = buffer.data() + cursor;
        cursor += batch_size_bytes;
        return {ptr, batch_size_bytes};
    }

}
