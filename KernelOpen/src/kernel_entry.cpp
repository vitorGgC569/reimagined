// KernelOpen Library Compilation Unit
// This file includes necessary headers to generate symbols for the library.

// Host Runtime
#include "host/uhk_runtime.h"
#include "host/ring_buffer_manager.h"

// Common
#include "common/uhk_types.h"
#include "common/bitnet_math.h"

// Note: Simulation files (Ghost/Analog) are typically header-only or have their own compilation units
// if they were implemented in .cpp. Here we focus on the Runtime which Pantheon uses.

namespace uhk {
namespace runtime {
    // Stub for the CUDA kernel if not compiling with NVCC
    #ifndef __CUDACC__
    extern "C" void universal_persistent_kernel(RingBufferControl* control_ptr) {
        // This should never be called on CPU in simulation mode
        std::cerr << "FATAL: CUDA Kernel called on CPU stub!" << std::endl;
        std::terminate();
    }
    #endif
}
}
