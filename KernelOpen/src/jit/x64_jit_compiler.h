#ifndef UHK_JIT_X64_COMPILER_H
#define UHK_JIT_X64_COMPILER_H

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <iostream>

namespace uhk {
namespace jit {

// =============================================================================
// Real x86-64 JIT Engine
// Allocates executable memory and writes raw machine code.
// =============================================================================

class X64JIT {
public:
    using FuncPtr = void (*)(int64_t*);

    X64JIT() {
        // Allocate page-aligned memory
        size_t size = 4096;
        memory_ = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_ == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
    }

    ~X64JIT() {
        munmap(memory_, 4096);
    }

    // Compile a sequence of simplified instructions into real x86-64 bytes
    // input_ops: 0=ADD, 1=SUB, 2=IMUL, 3=XOR
    // Targets: RAX (Accumulator), RDI (Pointer to data array)
    FuncPtr compile(const std::vector<uint8_t>& ops) {
        uint8_t* p = (uint8_t*)memory_;

        // 1. Prologue
        // push rbp; mov rbp, rsp;
        *p++ = 0x55;
        *p++ = 0x48; *p++ = 0x89; *p++ = 0xE5;

        // Save RBX (Callee saved)
        *p++ = 0x53;

        // Load args: RDI is int64_t* data
        // mov rax, [rdi] (Load data[0] into accumulator)
        *p++ = 0x48; *p++ = 0x8B; *p++ = 0x07;

        // mov rbx, [rdi + 8] (Load data[1] into operand register)
        *p++ = 0x48; *p++ = 0x8B; *p++ = 0x5F; *p++ = 0x08;

        // 2. Body (Generate Machine Code)
        for (uint8_t op : ops) {
            switch(op % 4) {
                case 0: // ADD RAX, RBX -> 48 01 D8
                    *p++ = 0x48; *p++ = 0x01; *p++ = 0xD8;
                    break;
                case 1: // SUB RAX, RBX -> 48 29 D8
                    *p++ = 0x48; *p++ = 0x29; *p++ = 0xD8;
                    break;
                case 2: // IMUL RAX, RBX -> 48 0F AF C3
                    *p++ = 0x48; *p++ = 0x0F; *p++ = 0xAF; *p++ = 0xC3;
                    break;
                case 3: // XOR RAX, RBX -> 48 31 D8
                    *p++ = 0x48; *p++ = 0x31; *p++ = 0xD8;
                    break;
            }
        }

        // 3. Epilogue
        // mov [rdi], rax (Store result back)
        *p++ = 0x48; *p++ = 0x89; *p++ = 0x07;

        // Restore RBX
        *p++ = 0x5B;

        // pop rbp; ret
        *p++ = 0x5D;
        *p++ = 0xC3;

        return (FuncPtr)memory_;
    }

private:
    void* memory_;
};

} // namespace jit
} // namespace uhk

#endif // UHK_JIT_X64_COMPILER_H
