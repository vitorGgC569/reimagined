#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>

// Platform Detection
#if defined(_WIN32)
    #ifndef AION_PLATFORM_WINDOWS
        #define AION_PLATFORM_WINDOWS
    #endif
#elif defined(__linux__)
    #ifndef AION_PLATFORM_LINUX
        #define AION_PLATFORM_LINUX
    #endif
#endif

namespace Aion {

    // Abstract Base Class for IO Backend
    class RingBufferBackend {
    public:
        virtual ~RingBufferBackend() = default;
        virtual void submit_read(int fd, void* buffer, size_t size, uint64_t offset, uint64_t user_data) = 0;
        virtual int process_sq() = 0;
        virtual bool peek_cq(uint64_t& user_data, int& res) = 0;
    };

    // The main class exposed to users
    class RingBuffer {
    private:
        std::unique_ptr<RingBufferBackend> backend;

    public:
        RingBuffer(uint32_t entries);
        ~RingBuffer();

        void submit_read(int fd, void* buffer, size_t size, uint64_t offset, uint64_t user_data);
        int process_sq();
        bool peek_cq(uint64_t& user_data, int& res);
    };

}
