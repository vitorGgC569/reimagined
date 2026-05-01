#pragma once
#include "nsos_config.h"
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <cstring>
#include <atomic>
#include <thread>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

// ============================================================================
// NSOS ARENA V3: Persistent Thread-Local Pools (Safety Fix)
// ============================================================================

struct ArenaBlock {
    uint8_t* ptr;
    size_t size;
    size_t offset;
    
    ArenaBlock(size_t s) : size(s), offset(0) {
        #ifdef _WIN32
        ptr = (uint8_t*)_aligned_malloc(size, 64);
        #else
        posix_memalign((void**)&ptr, 64, size);
        #endif
        if (!ptr) throw std::runtime_error("Arena Block OOM");
    }
    ~ArenaBlock() {
        #ifdef _WIN32
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

struct ArenaMark {
    size_t offset;
    size_t fallback_count;
};

class ArenaAllocator {
public:
    static ArenaAllocator& instance() {
        static ArenaAllocator inst;
        return inst;
    }

    void* alloc(size_t bytes, Device dev) {
        if (dev == Device::GPU) {
            #ifdef USE_CUDA
            void* ptr; cudaMalloc(&ptr, bytes); return ptr;
            #else
            return nullptr;
            #endif
        }

        // CPU Arena
        size_t padded = (bytes + 63) & ~63;
        ArenaBlock* block = get_thread_block();
        
        if (block->offset + padded > block->size) {
            // Allocate a fallback block and store it to prevent leak
            auto& fallbacks = get_thread_fallbacks();
            ArenaBlock* fallback = new ArenaBlock(padded);
            fallbacks.push_back(std::unique_ptr<ArenaBlock>(fallback));
            fallback->offset = padded;
            return fallback->ptr;
        }
        
        void* ptr = block->ptr + block->offset;
        block->offset += padded;
        return ptr;
    }

    // Scoped Reset
    ArenaMark get_mark() { 
        return { get_thread_block()->offset, get_thread_fallbacks().size() }; 
    }
    
    void rewind(ArenaMark mark) { 
        get_thread_block()->offset = mark.offset; 
        auto& fallbacks = get_thread_fallbacks();
        while (fallbacks.size() > mark.fallback_count) {
            fallbacks.pop_back();
        }
    }

private:
    ArenaAllocator() {}
    
    ArenaBlock* get_thread_block() {
        static thread_local std::unique_ptr<ArenaBlock> t_block = nullptr;
        if (!t_block) {
            // Memory is automatically released upon thread destruction
            t_block = std::make_unique<ArenaBlock>(512 * 1024 * 1024); // 512MB per thread
        }
        return t_block.get();
    }
    
    std::vector<std::unique_ptr<ArenaBlock>>& get_thread_fallbacks() {
        static thread_local std::vector<std::unique_ptr<ArenaBlock>> fallbacks;
        return fallbacks;
    }
};

class ArenaScope {
    ArenaMark mark;
public:
    ArenaScope() { mark = ArenaAllocator::instance().get_mark(); }
    ~ArenaScope() { ArenaAllocator::instance().rewind(mark); }
};

} // namespace nsos
