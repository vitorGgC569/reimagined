#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace Aion {

    // --- GDS / Hybrid Loader Architecture ---

    // Status enum
    enum class LoaderBackend {
        CPU_MEMCPY, // Standard: Read to RAM -> Copy to GPU
        GDS_DIRECT, // Fast: NVMe -> GPU (libcufile)
        IO_URING    // Fast CPU: NVMe -> RAM (io_uring)
    };

    class SmartLoader {
    public:
        SmartLoader();
        ~SmartLoader();

        // Detects system capabilities
        bool has_gpu_direct() const;
        bool has_io_uring() const;

        // Register a buffer.
        // If is_gpu_ptr is true, we treat it as a CUDA pointer.
        void register_buffer(void* ptr, size_t size, bool is_gpu_ptr);
        void unregister_buffer(void* ptr);

        // Load data from file to buffer
        // Dispatches to GDS if (has_gpu_direct && is_gpu_ptr)
        // Dispatches to io_uring if (has_io_uring && !is_gpu_ptr)
        // Fallback to pread
        size_t load(const std::string& filename, uint64_t offset, size_t size, void* target_ptr);

    private:
        bool gds_available;
        bool uring_available;
        // Internal tracking of registered buffers handles
    };

    // Industrial DataLoader (Gold Standard)
    class AionDataLoader {
    public:
        SmartLoader loader;
        std::vector<char> buffer; // Pinned memory ideally
        size_t buffer_size;
        size_t cursor;
        std::string current_file;

        AionDataLoader(size_t buffer_sz = 1024 * 1024 * 10); // 10MB default

        void open(const std::string& filename);

        // Returns raw pointer to next batch.
        // Returns pair {ptr, size}
        // Zero-Copy for Python via PyCapsule
        std::pair<void*, size_t> next_batch(size_t batch_size_bytes);
    };

}
