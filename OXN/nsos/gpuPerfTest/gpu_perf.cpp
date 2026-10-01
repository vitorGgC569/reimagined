// ============================================================================
// NSOS training throughput: GPU vs CPU, across model sizes.
// Builds a real JambaModel on each device, trains N steps (after a warm-up that
// absorbs cuBLAS init / kernel JIT / allocation), and reports sec/step + speedup.
// Self-contained in gpuPerfTest/ — links the existing CUDA nsos_core, touches nothing.
// ============================================================================
#include "../include/jamba.h"
#include "../include/trainer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace nsos;

static std::vector<int> cyclic(int n, int v) {
  std::vector<int> t(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) t[static_cast<size_t>(i)] = i % v;
  return t;
}
static double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

struct Cfg { const char* name; int L, D, V, seq, batch; };

// Returns seconds/step (timed over `measure` steps after `warmup`), or -1 on failure.
static double bench(Device dev, const Cfg& c, int warmup, int measure, float& loss) {
  try {
    JambaModel model(c.L, c.D, c.V, dev);
    Trainer tr(&model, 3e-3f);
    tr.weight_decay = 0.0f; tr.max_grad_norm = 2.0f; tr.warmup_steps = 2;
    const int n = std::max(c.V * 4, c.seq * c.batch * 50);
    std::vector<int> data = cyclic(n, c.V);
    std::chrono::steady_clock::time_point t0, t1;
    bool g0 = false, g1 = false;
    tr.train_loop(data, /*epochs*/ 1000, c.batch, c.seq,
                  [&](int step, float l) {
                    if (step == warmup) { t0 = std::chrono::steady_clock::now(); g0 = true; }
                    if (step == warmup + measure) { t1 = std::chrono::steady_clock::now(); g1 = true; loss = l; }
                  }, warmup + measure);
    if (!g0 || !g1) return -1.0;
    return secs(t0, t1) / measure;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "  [%s] failed: %s\n", (dev == Device::GPU ? "GPU" : "CPU"), e.what());
    return -1.0;
  }
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("=====================================================================\n");
  std::printf(" NSOS training throughput: GPU vs CPU (real JambaModel)\n");
  std::printf("=====================================================================\n");
#ifdef USE_CUDA
  std::printf(" CUDA build. GPU runs use the cuBLAS matmul path.\n");
#else
  std::printf(" NON-CUDA build -- GPU column will be empty.\n");
#endif

  const Cfg cfgs[] = {
    {"tiny  1L d64  v64",  1, 64,   64,   32,  8},
    {"small 2L d256 v4k",  2, 256,  4000, 64,  8},
    {"med   4L d512 v8k",  4, 512,  8000, 128, 8},
  };

  std::printf("\n %-20s | tok/step | CPU ms/step | GPU ms/step | GPU speedup | GPU tok/s\n", "config");
  for (const auto& c : cfgs) {
    float lc = 0, lg = 0;
    const double tc = bench(Device::CPU, c, 3, 15, lc);
    double tg = -1.0;
#ifdef USE_CUDA
    tg = bench(Device::GPU, c, 3, 15, lg);
#endif
    const int tok = c.batch * c.seq;
    std::printf(" %-20s | %7d  |  %9.2f  |  %9.2f  |  %8.2fx  | %9.0f\n",
                c.name, tok,
                tc > 0 ? tc * 1e3 : -1.0,
                tg > 0 ? tg * 1e3 : -1.0,
                (tc > 0 && tg > 0) ? tc / tg : 0.0,
                tg > 0 ? tok / tg : 0.0);
  }
  std::printf("=====================================================================\n");
  return 0;
}
