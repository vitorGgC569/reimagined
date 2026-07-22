// Standalone repro for the TTT-in-model construction crash.
// Builds a JambaModel with a TTT layer (use_ttt, ttt_period=2) under an SEH
// handler that prints the C++ stack (with source lines from the Debug .pdb) so
// we can pinpoint the null-deref / heap corruption.  CPU-only.
#include "jamba.h"
#include "nsos_config.h"

#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#pragma comment(lib, "dbghelp.lib")

static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ep) {
    HANDLE proc = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, NULL, TRUE);

    CONTEXT ctx = *ep->ContextRecord;
    STACKFRAME64 fr;
    std::memset(&fr, 0, sizeof(fr));
    fr.AddrPC.Offset = ctx.Rip;
    fr.AddrPC.Mode = AddrModeFlat;
    fr.AddrFrame.Offset = ctx.Rbp;
    fr.AddrFrame.Mode = AddrModeFlat;
    fr.AddrStack.Offset = ctx.Rsp;
    fr.AddrStack.Mode = AddrModeFlat;

    std::fprintf(stderr, "\n=== CRASH code=0x%08lx addr=%p ===\n",
                 (unsigned long)ep->ExceptionRecord->ExceptionCode,
                 ep->ExceptionRecord->ExceptionAddress);

    char sbuf[sizeof(SYMBOL_INFO) + 512];
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(sbuf);
    for (int i = 0; i < 60; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, thread, &fr, &ctx, NULL,
                         SymFunctionTableAccess64, SymGetModuleBase64, NULL)) {
            break;
        }
        if (fr.AddrPC.Offset == 0) {
            break;
        }
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 500;
        DWORD64 disp = 0;
        const char* nm = "???";
        if (SymFromAddr(proc, fr.AddrPC.Offset, &disp, sym)) {
            nm = sym->Name;
        }
        IMAGEHLP_LINE64 line;
        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        DWORD ldisp = 0;
        if (SymGetLineFromAddr64(proc, fr.AddrPC.Offset, &ldisp, &line)) {
            std::fprintf(stderr, "  #%02d %s  (%s:%lu)\n", i, nm, line.FileName,
                         (unsigned long)line.LineNumber);
        } else {
            std::fprintf(stderr, "  #%02d %s\n", i, nm);
        }
    }
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, char** argv) {
    SetUnhandledExceptionFilter(crash_handler);
    using namespace nsos;

    Device dev = Device::GPU;
    if (argc > 1 && std::string(argv[1]) == "CPU") {
        dev = Device::CPU;
    }
    std::fprintf(stderr, "[repro] device = %s\n", dev == Device::GPU ? "GPU" : "CPU");

    ModelConfig c;
    c.num_layers = 4;
    c.d_model = 128;
    c.vocab_size = 64;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.use_moe = false;
    c.use_ttt = true;
    c.ttt_period = 2;
    c.ttt_slot = 1;
    c.attention_period = 99;

    c.mamba2_faithful = true;

    std::fprintf(stderr, "[repro] constructing JambaModel with TTT (period=2)...\n");
    std::fflush(stderr);
    JambaModel model(c, dev);
    std::fprintf(stderr, "[repro] constructed OK\n");
    std::fflush(stderr);

    std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7, 8};

    std::fprintf(stderr, "[repro] forward (training mode, runs TTT inner loop)...\n");
    std::fflush(stderr);
    model.set_training_mode(true);
    Tensor out = model.forward_ids(ids);
    std::fprintf(stderr, "[repro] forward TRAIN OK (out.size=%zu)\n", (size_t)out.size);
    std::fflush(stderr);

    std::fprintf(stderr, "[repro] forward (eval mode)...\n");
    std::fflush(stderr);
    model.set_training_mode(false);
    Tensor out2 = model.forward_ids(ids);
    std::fprintf(stderr, "[repro] forward EVAL OK (out.size=%zu)\n", (size_t)out2.size);
    std::fflush(stderr);
    return 0;
}
