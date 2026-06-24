#include "../include/tensor.h"
#include "../include/nsos_arena.h"
#include "../include/nsos_math.h"
#include "../include/nsos/determinism.h"
#include "../include/tensor_iterator.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <set>
#include <mutex>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace nsos {

namespace {

int checked_tensor_size(const TensorShape& shape) {
    // Ponto ÚNICO de validação de shape (chamado por TODO construtor de Tensor
    // e por from_blob).  Acumula com guarda de overflow em vez de confiar no
    // wraparound de numel(): uma dimensão negativa convertida para size_t vira
    // um valor gigante e só estouraria o limite int DEPOIS — aqui rejeitamos a
    // CAUSA (dim < 0) com mensagem clara e paramos a multiplicação assim que
    // cruza INT_MAX, sem nunca produzir um produto sem sentido.
    constexpr size_t kMaxElements =
        static_cast<size_t>(std::numeric_limits<int>::max());
    size_t numel = 1;
    for (int d : shape.dims) {
        if (d < 0) {
            throw std::invalid_argument(
                "Tensor dimension is negative (shape invalido na construcao)");
        }
        numel *= static_cast<size_t>(d);
        if (numel > kMaxElements) {
            throw std::overflow_error(
                "Tensor element count exceeds NSOS v1 int storage limit");
        }
    }
    return static_cast<int>(numel);
}

int normalize_dim(int dim, int rank) {
    if (rank <= 0) {
        throw std::runtime_error("Tensor has no dimensions");
    }
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        throw std::out_of_range("Tensor dimension out of range");
    }
    return dim;
}

std::vector<int> make_indices(int rank) {
    return std::vector<int>(rank, 0);
}

std::mt19937& tensor_rng() {
    thread_local std::mt19937 gen;
    thread_local bool initialized = false;
    thread_local uint64_t seen_version = std::numeric_limits<uint64_t>::max();

    auto& manager = determinism::DeterminismManager::instance();
    const uint64_t global_seed = manager.get_global_seed();
    const uint64_t seed_version = manager.get_seed_version();

    if (global_seed != 0 && seed_version != seen_version) {
        // Deterministic path: derive the generator purely from the global seed
        // + seed version (NO thread id), so unseeded random/kaiming/xavier init
        // is reproducible across runs (determinism is a project gate).  Under
        // parallel init, threads then share a stream; for strict per-instance
        // determinism under parallelism use the seeded factories
        // (e.g. kaiming_uniform(shape, dev, seed)).
        auto op_rng = manager.get_rng_for_operation("tensor", "random", 0u);
        std::seed_seq seed{
            static_cast<uint32_t>(op_rng()),
            static_cast<uint32_t>(op_rng()),
            static_cast<uint32_t>(op_rng()),
            static_cast<uint32_t>(op_rng()),
        };
        gen.seed(seed);
        initialized = true;
        seen_version = seed_version;
        return gen;
    }

    if (!initialized) {
        std::random_device rd;
        std::seed_seq seed{
            rd(),
            rd(),
            static_cast<unsigned int>(
                std::hash<std::thread::id>{}(std::this_thread::get_id()))
        };
        gen.seed(seed);
        initialized = true;
        seen_version = seed_version;
    }
    return gen;
}

#ifdef USE_CUDA
void cublas_check(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS failure in ") + op);
    }
}

int& gpu_blas_state() {
    static int state = -1;
    return state;
}

cublasHandle_t& gpu_blas_handle_storage() {
    static cublasHandle_t handle = nullptr;
    return handle;
}

bool gpu_blas_supported() {
    int& state = gpu_blas_state();
    if (state != -1) {
        return state == 1;
    }

    const cudaError_t context_status = cudaFree(nullptr);
    if (context_status != cudaSuccess) {
        state = 0;
        return false;
    }

    cublasHandle_t& handle = gpu_blas_handle_storage();
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        handle = nullptr;
        state = 0;
        return false;
    }

    state = 1;
    return true;
}

cublasHandle_t cublas_handle() {
    if (!gpu_blas_supported()) {
        return nullptr;
    }
    return gpu_blas_handle_storage();
}

// AUDIT (post BATCH 4): the mixed-precision GEMM path used to do
//   cudaMalloc(a_low); cudaMalloc(b_low); <gemm>; cudaFree; cudaFree;
// per matmul call.  Each cudaMalloc/cudaFree pair costs ~1-10 ms AND
// forces an implicit stream synchronization — which on a 12-layer
// model with ~400 matmuls per training step burned 2-8 s/step of pure
// allocator overhead, completely masking the Tensor Core speedup that
// the user enabled with NSOS_MIXED_PRECISION=bf16.
//
// This workspace caches the BF16/FP16 staging buffers across all
// matmul calls and only reallocates when a larger tensor shape comes
// through (geometric 2× growth so we don't churn on minor changes).
// Single static instance is safe because the training loop is single-
// threaded; if we ever go multi-threaded we'd promote to thread_local.
struct GemmLowpWorkspace {
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    size_t a_capacity = 0;  // bytes
    size_t b_capacity = 0;  // bytes
    // Ensure both buffers hold at least the requested bytes.  Returns
    // false if cudaMalloc failed (caller should fall through to FP32
    // for this single call rather than crash).
    bool ensure(size_t a_bytes, size_t b_bytes) {
        if (a_bytes > a_capacity) {
            if (a_ptr) { cudaFree(a_ptr); a_ptr = nullptr; }
            const size_t cap = (a_capacity == 0)
                ? a_bytes
                : std::max(a_bytes, a_capacity * 2);
            if (cudaMalloc(&a_ptr, cap) != cudaSuccess) {
                a_ptr = nullptr;
                a_capacity = 0;
                return false;
            }
            a_capacity = cap;
        }
        if (b_bytes > b_capacity) {
            if (b_ptr) { cudaFree(b_ptr); b_ptr = nullptr; }
            const size_t cap = (b_capacity == 0)
                ? b_bytes
                : std::max(b_bytes, b_capacity * 2);
            if (cudaMalloc(&b_ptr, cap) != cudaSuccess) {
                b_ptr = nullptr;
                b_capacity = 0;
                return false;
            }
            b_capacity = cap;
        }
        return true;
    }
};
// We never free these on shutdown — CUDA context teardown reclaims
// the memory, and freeing static buffers during destruction risks
// touching an already-torn-down CUDA context.
// thread_local (not a single static): the HTTP server runs concurrent
// inference replicas, each on its own worker thread.  A shared static would let
// two threads cudaFree/cudaMalloc/cast into the same staging buffers at once
// (use-after-free / wrong results).  Per-thread instances make the mixed-
// precision GEMM path replica-safe; single-threaded training is unaffected.
GemmLowpWorkspace& gemm_lowp_workspace() {
    thread_local GemmLowpWorkspace ws;
    return ws;
}

void sync_cuda() {
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                                 cudaGetErrorString(launch_status));
    }
    static const bool force_sync = [] {
        const char* env = std::getenv("NSOS_CUDA_SYNC");
        return env != nullptr && std::string(env) == "1";
    }();
    if (force_sync) {
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA synchronize failed: ") +
                                     cudaGetErrorString(sync_status));
        }
    }
}

bool use_gpu_fast_path(const Tensor& tensor) {
    return tensor.get_device() == Device::GPU && tensor.size > 0;
}

bool use_gpu_fast_path(const Tensor& a, const Tensor& b) {
    return a.get_device() == Device::GPU && b.get_device() == Device::GPU &&
           a.size > 0 && b.size > 0;
}

template <typename T>
T copy_scalar_from_device(const T* device_ptr) {
    T value{};
    cudaMemcpy(&value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t count) { allocate(count); }
    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        if (count == 0) {
            return;
        }
        if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed");
        }
    }

    T* get() const { return ptr_; }
    operator T*() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};
#else
void sync_cuda() {}
bool use_gpu_fast_path(const Tensor&) {
    return false;
}
bool use_gpu_fast_path(const Tensor&, const Tensor&) {
    return false;
}

bool gpu_blas_supported() {
    return false;
}
#endif

#ifndef USE_CUDA
template <typename T>
T copy_scalar_from_device(const T*) {
    return T{};
}
#endif

#ifdef USE_CUDA
// Process-wide dedicated stream for host<->device / device<->device tensor
// copies, created once on first use.  Keeping copies off the default (compute)
// stream lets a copy synchronize only ITSELF instead of draining the whole
// device (cudaMemcpy's implicit full sync).
//
// It is created as a *blocking* stream (cudaStreamCreate, NOT cudaStreamNonBlocking)
// on purpose: a blocking stream implicitly orders after prior work on the legacy
// default stream (0).  That reproduces exactly the ordering guarantee of the
// cudaMemcpy this replaces — a kernel that just wrote `src` on stream 0 is
// guaranteed to finish before the copy reads it — without an explicit event.
// On creation failure we fall back to the original blocking cudaMemcpy, so the
// path is always correct even on a driver that refuses the stream.
static cudaStream_t tensor_copy_stream() {
    static cudaStream_t stream = [] {
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            s = nullptr;
        }
        (void)cudaGetLastError();
        return s;
    }();
    return stream;
}

// (movida p/ escopo de namespace — ver apos Tensor::uninitialized)

#else
// (variante CPU fundida na definicao unica)

#endif

} // namespace

#ifdef USE_CUDA
namespace {

// ── GPU caching allocator (PyTorch-style caching allocator) ───────────────
// Per-step training allocates/frees many managed tensors with RECURRING exact
// sizes (same shapes every iteration).  cudaMallocManaged/cudaFree are
// heavyweight, partially-synchronizing driver calls; doing dozens per step adds
// avoidable overhead AND makes CUDA Graph capture impossible (allocation is
// illegal during capture).  This pool keeps freed blocks on per-exact-size free
// lists and hands them back on the next request -> with static shapes, reuse is
// perfect (zero fragmentation) and addresses are stable across steps (the
// precondition for graph replay; warm the pool with one step, then capture
// hits only the free list -> no driver alloc inside the captured region).
// Disable with NSOS_GPU_POOL=0 (falls back to raw cudaMallocManaged/cudaFree).
class ManagedPool {
public:
    static ManagedPool& instance() {
        static ManagedPool pool;
        return pool;
    }

    // Size-class binning (estilo PyTorch caching allocator).  Cachear por
    // tamanho EXATO fragmenta patologicamente sob shapes dependentes de dados
    // (contagem de tokens por expert no MoE, buckets de comprimento variável):
    // cada tamanho inédito vira uma free-list própria que nunca recicla, o
    // cache incha até o cap e o footprint UM chega a ~96% da GPU -> thrash
    // (medido na T4: mem 11.4->15.4GB em 3 steps, step-time crescendo junto,
    // sem throttle térmico).  Arredondar para classes faz tamanhos vizinhos
    // reutilizarem o mesmo bloco: <1MB -> múltiplo de 64KB; >=1MB -> de 2MB.
    static size_t bin_bytes(size_t bytes) {
        constexpr size_t k64 = 64ull << 10;
        constexpr size_t k2m = 2ull << 20;
        if (bytes < (1ull << 20)) {
            return ((bytes + k64 - 1) / k64) * k64;
        }
        return ((bytes + k2m - 1) / k2m) * k2m;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;
        if (!enabled_) return raw_alloc(bytes);
        bytes = bin_bytes(bytes);
        std::lock_guard<std::mutex> lk(mtx_);
        live_bytes_ += bytes;
        auto& bin = free_[bytes];
        if (!bin.empty()) {
            void* p = bin.back();
            bin.pop_back();
            cached_bytes_ -= bytes;
            return p;
        }
        void* p = raw_alloc(bytes);
        if (!p) {
            // Out of memory: return every cached free block to the driver and
            // retry once (mirrors a caching allocator's empty-cache-on-OOM).
            trim_locked();
            p = raw_alloc(bytes);
        }
        if (p) live_[p] = bytes;
        return p;
    }

    void deallocate(void* p) {
        if (!p) return;
        if (!enabled_) {
            cudaFree(p);
            return;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = live_.find(p);
        if (it == live_.end()) {
            cudaFree(p);  // not pool-owned (shouldn't happen) — be safe
            return;
        }
        const size_t sz = it->second;
        if (cap_bytes_ != 0 && cached_bytes_ + sz > cap_bytes_) {
            // Cache is full: return this block to the driver instead of caching
            // it, so total managed footprint stays bounded.
            cudaFree(p);
            live_.erase(it);
            live_bytes_ -= (live_bytes_ >= sz ? sz : live_bytes_);
            return;
        }
        free_[sz].push_back(p);
        cached_bytes_ += sz;
        live_bytes_ -= (live_bytes_ >= sz ? sz : live_bytes_);
        // Guarda de oversubscription UM (auditoria #27): checagem barata por
        // cadência — UM não dá OOM, degrada paginando (T4 a 96% custou steps
        // 6->8s antes do binning).  Acima de 88% de uso do device: poda o
        // cache e avisa uma vez por episódio.
        if (((++dealloc_probe_) & 511u) == 0) {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0) {
                const double used = 1.0 - static_cast<double>(free_b) /
                                              static_cast<double>(total_b);
                if (used > 0.88) {
                    trim_locked();
                    if (!um_pressure_warned_) {
                        std::fprintf(stderr,
                                     "[pool] WARN: device %.0f%% cheio — cache "
                                     "podado (evitando thrash UM)\n",
                                     used * 100.0);
                        um_pressure_warned_ = true;
                    }
                } else if (used < 0.80) {
                    um_pressure_warned_ = false;
                }
            }
            (void)cudaGetLastError();
        }
    }

public:
    PoolStats stats() {
        std::lock_guard<std::mutex> lk(mtx_);
        PoolStats s;
        s.cached_bytes = cached_bytes_;
        s.live_bytes = live_bytes_;
        s.bins = free_.size();
        return s;
    }
    void trim() {
        std::lock_guard<std::mutex> lk(mtx_);
        trim_locked();
    }

private:
    ManagedPool() {
        const char* env = std::getenv("NSOS_GPU_POOL");
        enabled_ = !(env && std::string(env) == "0");
        // Cap on CACHED (free-list) bytes.  Unified Memory oversubscribes
        // SILENTLY (no OOM — it just thrashes via page eviction), so an
        // unbounded cache (e.g. interleaving an inference and a training working
        // set in one process) could push a small card into thrash.  Default =
        // 60% of device memory; NSOS_GPU_POOL_MAX_MB overrides; "0" = unlimited.
        const char* cap_env = std::getenv("NSOS_GPU_POOL_MAX_MB");
        if (cap_env) {
            cap_bytes_ = static_cast<size_t>(std::strtoull(cap_env, nullptr, 10))
                         * 1024ull * 1024ull;
        } else {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0) {
                cap_bytes_ = static_cast<size_t>(static_cast<double>(total_b) * 0.6);
            }
            (void)cudaGetLastError();
        }
    }

    // Return all currently-cached (free) blocks to the driver.  Live blocks
    // still owned by a Tensor are untouched.
    void trim_locked() {
        for (auto& kv : free_) {
            for (void* p : kv.second) {
                cudaFree(p);
                live_.erase(p);
            }
            kv.second.clear();
        }
        cached_bytes_ = 0;
    }

    static void* raw_alloc(size_t bytes) {
        void* raw = nullptr;
        if (cudaMallocManaged(&raw, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        // Pascal+Windows hardening, applied ONCE per physical block (it then
        // persists across every pooled reuse): hint the driver that this UM
        // block is accessed by host and device so pages stay migratable rather
        // than faulting on host access (sm_61 + Windows lacks demand paging).
        int device_id = 0;
        if (cudaGetDevice(&device_id) == cudaSuccess) {
#if CUDART_VERSION >= 13000
            cudaMemLocation loc_dev;
            loc_dev.type = cudaMemLocationTypeDevice;
            loc_dev.id = device_id;
            cudaMemLocation loc_host;
            loc_host.type = cudaMemLocationTypeHost;
            loc_host.id = 0;
            cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, loc_dev);
            cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, loc_host);
#else
            cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, device_id);
            cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
#endif
        }
        (void)cudaGetLastError();
        return raw;
    }

    bool enabled_ = true;
    size_t cached_bytes_ = 0;  // current sum of free-list block sizes
    size_t live_bytes_ = 0;    // soma dos blocos atualmente entregues a Tensors
    unsigned dealloc_probe_ = 0;        // cadência da guarda de pressão UM
    bool um_pressure_warned_ = false;   // 1 aviso por episódio de pressão
    size_t cap_bytes_ = 0;     // max cached bytes (0 = unlimited)
    std::mutex mtx_;
    std::unordered_map<size_t, std::vector<void*>> free_;  // exact bytes -> free blocks
    std::unordered_map<void*, size_t> live_;               // ptr -> its byte size
};

}  // namespace
#endif  // USE_CUDA

void TensorDeleter::operator()(float* ptr) {
    if (!ptr) return;
    if (device == Device::GPU) {
#ifdef USE_CUDA
        ManagedPool::instance().deallocate(ptr);
#endif
    } else {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

// Definida adiante (junto de Tensor::uninitialized); lida pelo construtor.
bool tensor_skip_fill_flag();

// (auditoria #20) Aviso único por op quando um tensor GPU cai num caminho
// host-only — regressões de dispatch nunca mais passam em silêncio.
static void warn_host_fallback_once(const char* op, Device dev) {
#ifdef USE_CUDA
    if (dev != Device::GPU) return;
    static std::mutex m;
    static std::set<std::string> warned;
    std::lock_guard<std::mutex> lk(m);
    if (warned.insert(op).second) {
        std::fprintf(stderr,
                     "[tensor] WARN: op '%s' executando no HOST com tensor GPU "
                     "(fallback sem kernel) - perf degradada\n",
                     op);
    }
#else
    (void)op; (void)dev;
#endif
}


Tensor::Tensor() : size(0), device(Device::CPU) {
    shape = TensorShape(std::vector<int>{});
    data_ptr = nullptr;
}

Tensor::Tensor(std::vector<int> s, Device dev, float fill_value) : device(dev) {
    shape = TensorShape(s);
    size = checked_tensor_size(shape);
    if (size == 0) {
        data_ptr = nullptr;
        return;
    }

    float* raw_ptr = nullptr;
    if (device == Device::GPU) {
#ifdef USE_CUDA
        // Allocate from the managed caching pool.  cudaMallocManaged +
        // cudaMemAdvise (Pascal+Windows UM hardening) happen once per physical
        // block inside the pool; a pooled reuse returns a ready, advise-tagged
        // block with no driver call.  sync_host_access remains the safety net
        // for host access regardless.
        raw_ptr = static_cast<float*>(
            ManagedPool::instance().allocate(static_cast<size_t>(size) * sizeof(float)));
#endif
    } else {
#ifdef _WIN32
        raw_ptr = static_cast<float*>(_aligned_malloc(size * sizeof(float), 64));
#else
        if (posix_memalign(reinterpret_cast<void**>(&raw_ptr), 64, size * sizeof(float)) != 0) {
            raw_ptr = nullptr;
        }
#endif
    }

    if (!raw_ptr) {
        throw std::runtime_error("Tensor allocation failed");
    }

    if (tensor_skip_fill_flag()) {
        // Tensor::uninitialized: produtor garante sobrescrita total; pular o
        // memset economiza um kernel por alocação no hot path de treino.
        data_ptr = std::shared_ptr<float>(raw_ptr, TensorDeleter(device));
        return;
    }

    if (fill_value == 0.0f) {
        if (device == Device::GPU) {
#ifdef USE_CUDA
            // Zero on the default stream and return WITHOUT a full-device drain.
            // Kernels that consume this tensor run on the same (default) stream
            // and are ordered after the memset automatically; host access is
            // gated by sync_host_access()/data().  The previous
            // cudaDeviceSynchronize() drained the WHOLE device on every
            // zero-filled GPU allocation -> dozens of pipeline stalls per train
            // step, a dominant cause of low GPU utilization.
            cudaMemsetAsync(raw_ptr, 0, size * sizeof(float), 0);
            (void)cudaGetLastError();
#endif
        } else {
            std::memset(raw_ptr, 0, size * sizeof(float));
        }
    } else {
        if (device == Device::GPU) {
#ifdef USE_CUDA
            // Synchronous cudaMemcpy from pageable host memory blocks the host
            // until the copy completes, so host_values is safe to free on return
            // and no separate cudaDeviceSynchronize is required.
            std::vector<float> host_values(static_cast<size_t>(size), fill_value);
            cudaMemcpy(raw_ptr,
                       host_values.data(),
                       static_cast<size_t>(size) * sizeof(float),
                       cudaMemcpyHostToDevice);
            (void)cudaGetLastError();
#endif
        } else {
            std::fill_n(raw_ptr, size, fill_value);
        }
    }

    data_ptr = std::shared_ptr<float>(raw_ptr, TensorDeleter(device));
}

// Flag thread-local lida pelo construtor: quando armada (via
// Tensor::uninitialized), o fill inicial é pulado.  Guard RAII garante reset
// mesmo se o construtor lançar (falha de alocação).
namespace {
thread_local bool g_tensor_skip_fill = false;
struct SkipFillGuard {
    SkipFillGuard() { g_tensor_skip_fill = true; }
    ~SkipFillGuard() { g_tensor_skip_fill = false; }
};
}  // namespace

bool tensor_skip_fill_flag() { return g_tensor_skip_fill; }

Tensor Tensor::uninitialized(const std::vector<int>& s, Device dev) {
    // TRIAGEM 2026-06-11 (T4): a bissecção por kill-switch isolou ESTA elisão
    // como causa de grads zero/NaN/explosão no harness de paridade (pisos
    // voltaram a ~0,2 com uninit=0; fused=0 e async=0 continuaram doentes).
    // Algum produtor da lista "provadamente 100% sobrescrito" não cobre tudo
    // em alguma condição.  Default invertido para SEGURO: a elisão agora é
    // OPT-IN (NSOS_UNINIT=1) até a prova de cobertura ser fechada site a site
    // com o instrumento de paridade.  Custo de manter zero-fill: ~1 memset
    // assíncrono por alocação — nunca foi medido como gargalo isolado.
    static const bool uninit_enabled = [] {
        const char* e = std::getenv("NSOS_UNINIT");
        return e && e[0] == '1';
    }();
    if (!uninit_enabled) {
        return Tensor(s, dev);
    }
    SkipFillGuard guard;
    return Tensor(s, dev);
}


// (auditoria #8) Definição ÚNICA em escopo de namespace — casa com a decl do
// tensor.h e elimina as cópias divergentes de trainer.cpp/jamba.cpp.
void copy_tensor_bytes(float* dst, Device dst_device, const float* src,
                       Device src_device, size_t bytes, bool async_d2d) {
#ifdef USE_CUDA

    if (bytes == 0) {
        return;
    }
    if (dst_device == Device::GPU || src_device == Device::GPU) {
        static const bool async_d2d_enabled = [] {
            const char* e = std::getenv("NSOS_ASYNC_D2D");
            return !(e && e[0] == '0');
        }();
        if (async_d2d_enabled && async_d2d && dst_device == Device::GPU &&
            src_device == Device::GPU) {
            // D2D entre buffers do pool: enfileira no stream legacy (0) e NÃO
            // bloqueia o host.  A ordenação é total — produtores de `src` e
            // consumidores de `dst` rodam no stream 0, e o reuso de buffers
            // do pool também é stream-ordered; leituras host posteriores
            // passam por sync_host_access (que drena se houver pendência).
            // O sync de host aqui era um dreno de pipeline COMPLETO por
            // clone/save (dezenas por step no forward de treino) — medido
            // como o overhead uniforme por camada na T4.  Opt-in por call
            // site: cópias de/para ponteiros EXTERNOS (__cuda_array_interface__)
            // e H2D/D2H mantêm o sync (lifetime do buffer de origem).
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, nullptr);
            const cudaError_t launch = cudaGetLastError();
            if (launch != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA async D2D memcpy failed: ") +
                                         cudaGetErrorString(launch));
            }
            return;
        }
        cudaStream_t stream = tensor_copy_stream();
        if (stream) {
            // Async copy on the dedicated stream, then wait on JUST this stream.
            // Equivalent ordering to the legacy blocking cudaMemcpy (blocking
            // stream serializes with stream 0), but does not stall unrelated
            // device work the way cudaDeviceSynchronize would.
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
            const cudaError_t launch = cudaGetLastError();
            if (launch != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA async memcpy failed: ") +
                                         cudaGetErrorString(launch));
            }
            const cudaError_t sync = cudaStreamSynchronize(stream);
            if (sync != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA copy-stream sync failed: ") +
                                         cudaGetErrorString(sync));
            }
        } else {
            cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memcpy failed: ") +
                                         cudaGetErrorString(status));
            }
        }
        return;
    }
    std::memcpy(dst, src, bytes);

#else
    (void)dst_device; (void)src_device; (void)async_d2d;
    if (bytes > 0) {
        std::memcpy(dst, src, bytes);
    }
#endif
}

PoolStats pool_stats() {
#ifdef USE_CUDA
    return ManagedPool::instance().stats();
#else
    return PoolStats{};
#endif
}

void release_cached_memory() {
#ifdef USE_CUDA
    ManagedPool::instance().trim();
#endif
}

Tensor Tensor::random(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return t;
}

Tensor Tensor::kaiming_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    float fan_in = s.empty() ? 1.0f : static_cast<float>(s.back());
    float bound = std::sqrt(6.0f / std::max(fan_in, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return t;
}

Tensor Tensor::kaiming_uniform(const std::vector<int>& s, Device dev,
                               uint64_t seed) {
    if (seed == 0) {
        // Preserve historical behavior for seed=0 (un-seeded path).
        return kaiming_uniform(s, dev);
    }
    Tensor t(s, dev);
    float fan_in = s.empty() ? 1.0f : static_cast<float>(s.back());
    float bound = std::sqrt(6.0f / std::max(fan_in, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::mt19937 local_rng(static_cast<uint32_t>(seed));
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(local_rng);
    }
    return t;
}

Tensor Tensor::xavier_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    float fan_in = s.empty() ? 1.0f : static_cast<float>(s.front());
    float fan_out = s.size() > 1 ? static_cast<float>(s.back()) : fan_in;
    float bound = std::sqrt(6.0f / std::max(fan_in + fan_out, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return t;
}

Tensor Tensor::eye(int n, Device dev) {
    Tensor t = zeros({n, n}, dev);
    float* dst = t.data();
    for (int i = 0; i < n; ++i) {
        dst[i * n + i] = 1.0f;
    }
    return t;
}

Tensor Tensor::from_scalar(float val, Device dev) {
    Tensor t({1}, dev);
    t.data()[0] = val;
    return t;
}

void Tensor::clip_grad_norm_(std::vector<Tensor>& params, float max_norm) {
    float total_norm_sq = 0.0f;
    for (auto& p : params) {
        float n = p.norm();
        total_norm_sq += n * n;
    }

    float total_norm = std::sqrt(total_norm_sq);
    if (total_norm <= max_norm || total_norm <= 1e-8f) {
        return;
    }

    float scale = max_norm / (total_norm + 1e-6f);
    for (auto& p : params) {
        float* dst = p.data();
        for (int i = 0; i < p.size; ++i) {
            dst[i] *= scale;
        }
    }
}

Tensor Tensor::add(const Tensor& other) const {
    TensorShape out_shape;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_shape);
    Tensor result(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported()) {
        if (shape == other.shape) {
            launch_add_kernel(result.raw_data(), raw_data(), other.raw_data(),
                              size);
            sync_cuda();
            return result;
        }
        if (!shape.empty() && other.shape.size() == 1 &&
            other.shape.back() == shape.back() && size == result.size) {
            launch_add_broadcast_kernel(result.raw_data(), raw_data(),
                                         other.raw_data(), size, shape.back());
            sync_cuda();
            return result;
        }
    }
#endif
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a + b; });
    return result;
}

Tensor Tensor::sub(const Tensor& other) const {
    TensorShape out_shape;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_shape);
    Tensor result(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported() &&
        shape == other.shape) {
        launch_sub_kernel(result.raw_data(), raw_data(), other.raw_data(),
                          size);
        sync_cuda();
        return result;
    }
#endif
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a - b; });
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    TensorShape out_shape;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_shape);
    Tensor result(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported()) {
        if (shape == other.shape) {
            launch_mul_tensor_kernel(result.raw_data(), raw_data(),
                                     other.raw_data(), size);
            sync_cuda();
            return result;
        }
        if (!shape.empty() && other.shape.size() == 1 &&
            other.shape.back() == shape.back() && size == result.size) {
            const int cols = shape.back();
            const int rows = size / std::max(cols, 1);
            launch_mul_vector_broadcast_kernel(result.raw_data(), raw_data(),
                                                other.raw_data(), rows, cols);
            sync_cuda();
            return result;
        }
    }
#endif
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a * b; });
    return result;
}

Tensor Tensor::mul(float scalar) const {
    Tensor result(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_mul_scalar_kernel(result.raw_data(), raw_data(), scalar, size);
        sync_cuda();
        return result;
    }
#endif
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i] * scalar;
    }
    return result;
}

// ── Mixed-precision (BF16/FP16 Tensor-Core) GEMM control ─────────────────────
// 0 = FP32 (default, bit-parity with CPU); 1 = BF16; 2 = FP16.  Master weights
// and optimizer state stay FP32; inputs are cast to the low precision only at
// the GEMM call site (standard AMP recipe).  Runtime-settable so a model/config
// or the Python plane can flip it on a modern (sm_75+) GPU for 2-4x training
// throughput, while default-OFF preserves FP32 parity for existing checkpoints.
// Initialized from NSOS_MIXED_PRECISION for back-compat.  Atomic: the control
// thread writes, matmul reads per call (relaxed — exactness of the switch step
// is irrelevant, only that it is observed).
static std::atomic<int>& matmul_precision_mode_storage() {
    static std::atomic<int> mode{[] {
        const char* env = std::getenv("NSOS_MIXED_PRECISION");
        if (!env || *env == '\0') return 0;
        const std::string v(env);
        if (v == "bf16" || v == "BF16") return 1;
        if (v == "fp16" || v == "FP16") return 2;
        return 0;
    }()};
    return mode;
}

void set_matmul_precision_mode(int mode) {
    if (mode < 0 || mode > 2) mode = 0;
    matmul_precision_mode_storage().store(mode, std::memory_order_relaxed);
}

int matmul_precision_mode() {
    return matmul_precision_mode_storage().load(std::memory_order_relaxed);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() < 2 || other.shape.size() < 2) {
        throw std::runtime_error("matmul requires rank >= 2 tensors");
    }

    int rank_a = static_cast<int>(shape.size());
    int rank_b = static_cast<int>(other.shape.size());
    int m = shape[rank_a - 2];
    int k = shape[rank_a - 1];
    int k_other = other.shape[rank_b - 2];
    int n = other.shape[rank_b - 1];

    if (k != k_other) {
        throw std::runtime_error("matmul shape mismatch");
    }

    std::vector<int> out_dims = shape.dims;
    out_dims.back() = n;
    if (rank_b > 2) {
        out_dims.assign(other.shape.dims.begin(), other.shape.dims.end());
        out_dims[rank_b - 2] = m;
    }

#ifdef USE_CUDA
    const bool matmul_gpu_path =
        use_gpu_fast_path(*this, other) && gpu_blas_supported();
#else
    const bool matmul_gpu_path = false;
#endif
    // GPU: GEMM beta=0 sobrescreve 100% de C -> alocação sem zero-fill.
    Tensor result = matmul_gpu_path ? Tensor::uninitialized(out_dims, device)
                                    : Tensor(out_dims, device);
    int batch = size / (m * k);
    int other_batch = other.size / (k * n);
    if (other_batch != 1 && other_batch != batch) {
        throw std::runtime_error("Unsupported batched matmul shape mismatch");
    }

#ifdef USE_CUDA
    if (matmul_gpu_path) {
        // raw_data(): SEM sync_host_access.  O fetch antigo via data() nos 3
        // tensores ANTES deste branch drenava o device a cada matmul (a op
        // mais chamada do treino, ~100-200x/step) — o passo inteiro rodava
        // efetivamente síncrono op-a-op.  cuBLAS roda no stream legacy,
        // ordenado com produtores/consumidores; leitura host posterior passa
        // por sync_host_access.
        const float* a_ptr = raw_data();
        const float* b_ptr = other.raw_data();
        float* out_ptr = result.raw_data();
        cublasHandle_t handle = cublas_handle();
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // AUDIT #6 + LEARN B3 (2026-05-16): mixed-precision matmul
        // dispatch via cublasGemmEx + Tensor Cores.  T4 (sm_75) and
        // newer have hardware Tensor Cores for BF16 GEMM that runs
        // ~4-8x faster than FP32 cublasSgemm.  When NSOS_MIXED_PRECISION
        // is set to "bf16" or "fp16", we cast inputs to the lower
        // precision, do the GEMM in Tensor Cores, and write FP32
        // accumulated output.
        //
        // Why BF16 over FP16:
        //   BF16 has the same 8-bit exponent as FP32 (range ~1e-38 to
        //   ~3e38) — no underflow/overflow risk vs FP32 baselines.
        //   FP16 has only 5-bit exponent (range ~6e-5 to ~6e4) which
        //   bites BitNet 1.58 because the per-row activation scaling
        //   factors can land outside that range.  BF16 is the right
        //   default for any model with BitNet-style quantization.
        //
        // Why this is gated behind an env var:
        //   Existing checkpoints were trained in FP32.  Switching to
        //   BF16 mid-run produces visible loss spikes (the optimizer
        //   has to re-calibrate to the new noise floor).  Users opt
        //   in explicitly for the speedup; default stays FP32 for
        //   correctness parity with prior runs.
        //
        // Adam state stays FP32 always.  Weights stay FP32 in storage;
        // they're cast to BF16 only at the GEMM call site.  This is
        // the standard "mixed precision" recipe (TF / PyTorch AMP).
        // Runtime-controllable (set_matmul_precision_mode / ModelConfig /
        // Python binding); initialized from NSOS_MIXED_PRECISION for back-compat.
        const int mixed_mode = matmul_precision_mode();

        if (mixed_mode != 0) {
            // cublasGemmEx with mixed precision: A and B in lower
            // precision, accumulation and C in FP32.  We allocate
            // workspace tensors for the BF16/FP16 copies of A and B
            // and use cublasGemmStridedBatchedEx to handle the batch.
            //
            // Per-call cost of the cast: O(rows × cols) elementwise.
            // For a forward pass this is dominated by the GEMM cost
            // (O(m*n*k)) so amortizes well.  In a tight loop the
            // alloc+cast could be hoisted as a workspace; current
            // implementation favors clarity.
            cudaDataType_t compute_dtype =
                (mixed_mode == 1) ? CUDA_R_16BF : CUDA_R_16F;
            cublasComputeType_t accumulate_type = CUBLAS_COMPUTE_32F;

            const size_t bytes_per_lp = 2;  // both bf16 and fp16 are 2 bytes
            const size_t total_a_elems =
                static_cast<size_t>(batch) * static_cast<size_t>(m) * static_cast<size_t>(k);
            const size_t total_b_elems =
                (other_batch == 1
                     ? static_cast<size_t>(k) * static_cast<size_t>(n)
                     : static_cast<size_t>(batch) * static_cast<size_t>(k) *
                           static_cast<size_t>(n));

            // Get cached staging buffers from the workspace (resized
            // on demand, reused across calls).  Replaces the per-
            // matmul cudaMalloc/cudaFree pair that was burning 2-8 s
            // per training step on a 12-layer model.
            GemmLowpWorkspace& ws = gemm_lowp_workspace();
            if (ws.ensure(total_a_elems * bytes_per_lp,
                          total_b_elems * bytes_per_lp)) {
                void* a_low_ptr = ws.a_ptr;
                void* b_low_ptr = ws.b_ptr;

                // Cast A and B to lower precision.  Reuse our
                // existing float->bf16 / float->fp16 cast kernel if
                // available; else use cuBLAS's BLAS-internal cast via
                // cublasSgemmEx auto-conversion through CUDA_R_32F
                // inputs is also valid but slower.  We launch the
                // dedicated cast kernel for both A and B.
                launch_cast_f32_to_lowp_kernel(
                    a_low_ptr, a_ptr, total_a_elems, mixed_mode);
                launch_cast_f32_to_lowp_kernel(
                    b_low_ptr, b_ptr, total_b_elems, mixed_mode);

                const long long stride_a = static_cast<long long>(m) * k;
                const long long stride_b = (other_batch == 1)
                                                ? 0LL
                                                : static_cast<long long>(k) * n;
                const long long stride_c = static_cast<long long>(m) * n;
                cublas_check(
                    cublasGemmStridedBatchedEx(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        b_low_ptr,
                        compute_dtype,
                        n,
                        stride_b,
                        a_low_ptr,
                        compute_dtype,
                        k,
                        stride_a,
                        &beta,
                        out_ptr,
                        CUDA_R_32F,
                        n,
                        stride_c,
                        batch,
                        accumulate_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "cublasGemmStridedBatchedEx");
                // No cudaFree — buffers are owned by the workspace and
                // outlive this call.  sync_cuda() still runs to surface
                // any kernel-launch errors via the LastError API.
                sync_cuda();
                return result;
            }
            // ws.ensure() failed (rare — only on cudaMalloc OOM).
            // Fall through to the FP32 cublasSgemmStridedBatched path
            // below; correctness preserved, just slower for this call.
        }

        // AUDIT #3 (2026-05-16): use cublasSgemmStridedBatched to fuse
        // all batch_idx iterations into ONE kernel launch.  The
        // original loop did `batch` separate cublasSgemm calls, each
        // costing ~10-30 us of launch overhead plus a barrier between
        // adjacent calls (CUDA serializes kernels on the default
        // stream).  For batch=32 that was 32x the launch overhead and
        // no parallelism across batch dimension within the SM array.
        //
        // cublasSgemmStridedBatched does a single launch that runs
        // all batches in parallel across SMs.  The strides are the
        // gap between consecutive batch matrices in memory:
        //   strideA = m*k       (always)
        //   strideB = 0 if other has no batch dim (broadcast),
        //             k*n if other has batch dim (no broadcast)
        //   strideC = m*n       (always)
        //
        // The math is IDENTICAL to the loop above.  Speedup is 1.2-2x
        // for matmul-heavy paths; combined with sync removal, more.
        const long long stride_a = static_cast<long long>(m) * k;
        const long long stride_b = (other_batch == 1)
                                       ? 0LL
                                       : static_cast<long long>(k) * n;
        const long long stride_c = static_cast<long long>(m) * n;
        cublas_check(
            cublasSgemmStridedBatched(handle,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       n,
                                       m,
                                       k,
                                       &alpha,
                                       b_ptr,
                                       n,
                                       stride_b,
                                       a_ptr,
                                       k,
                                       stride_a,
                                       &beta,
                                       out_ptr,
                                       n,
                                       stride_c,
                                       batch),
            "cublasSgemmStridedBatched");
        sync_cuda();
        return result;
    }
#endif

    // Fallback host: o fetch via data() (com sync_host_access) vive SÓ aqui —
    // o branch GPU acima usa raw_data() sem drenos.
    const float* a_ptr = data();
    const float* b_ptr = other.data();
    float* out_ptr = result.data();

    // CPU: BLIS-style blocked GEMM (MathOps::gemm) per batch -- same contiguous
    // row-major layout the naive loop below assumes, but an order of magnitude
    // faster.  beta=0 overwrites the freshly-allocated result.  The naive loop is
    // kept as a fallback for any non-CPU data that reaches here.
    if (device == Device::CPU) {
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const float* a_batch = a_ptr + static_cast<size_t>(batch_idx) * m * k;
            const float* b_batch =
                b_ptr + (other_batch == 1 ? 0 : static_cast<size_t>(batch_idx) * k * n);
            float* out_batch = out_ptr + static_cast<size_t>(batch_idx) * m * n;
            MathOps::gemm(m, n, k, 1.0f, a_batch, k, b_batch, n, 0.0f, out_batch, n);
        }
        return result;
    }

#pragma omp parallel for
    for (int row_index = 0; row_index < batch * m; ++row_index) {
        const int batch_idx = row_index / m;
        const int row = row_index % m;
        const float* a_batch = a_ptr + batch_idx * m * k;
        const float* b_batch = b_ptr + (other_batch == 1 ? 0 : batch_idx * k * n);
        float* out_row = out_ptr + row_index * n;
        std::fill_n(out_row, n, 0.0f);

        const float* a_row = a_batch + row * k;
        for (int kk = 0; kk < k; ++kk) {
            const float a_value = a_row[kk];
            const float* b_row = b_batch + kk * n;
            for (int col = 0; col < n; ++col) {
                out_row[col] += a_value * b_row[col];
            }
        }
    }

    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (shape.empty()) {
        return clone();
    }

    int rank = static_cast<int>(shape.size());
    dim0 = normalize_dim(dim0, rank);
    dim1 = normalize_dim(dim1, rank);
    if (dim0 == dim1) {
        return clone();
    }

    std::vector<int> out_dims = shape.dims;
    std::swap(out_dims[dim0], out_dims[dim1]);
    Tensor result(out_dims, device);

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported() &&
        rank == 2 && dim0 == 0 && dim1 == 1) {
        launch_transpose2d_kernel(result.data(), data(), shape[0], shape[1]);
        sync_cuda();
        return result;
    }
#endif

    // Direct stride math: decompose the output flat index into coordinates,
    // swap dim0/dim1 to recover the source coordinates, and index the source
    // via its precomputed strides.  The previous version allocated a
    // std::vector (src_idx) AND called get() (which recomputes a flat index)
    // for every element.
    warn_host_fallback_once("transpose(rank>2)", device);
    std::vector<int> idx(rank);
    const float* src = data();
    float* dst = result.data();
    for (int flat = 0; flat < result.size; ++flat) {
        int remaining = flat;
        for (int d = rank - 1; d >= 0; --d) {
            idx[d] = remaining % out_dims[d];
            remaining /= out_dims[d];
        }
        std::swap(idx[dim0], idx[dim1]);
        size_t src_flat = 0;
        for (int d = 0; d < rank; ++d) {
            src_flat += static_cast<size_t>(idx[d]) * shape.strides[d];
        }
        dst[flat] = src[src_flat];
        std::swap(idx[dim0], idx[dim1]);
    }

    return result;
}

Tensor Tensor::transpose() const {
    if (shape.size() < 2) {
        return clone();
    }
    return transpose(-2, -1);
}

Tensor Tensor::relu() const {
    Tensor result(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_relu_kernel(result.raw_data(), raw_data(), size);
        sync_cuda();
        return result;
    }
#endif
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = std::max(src[i], 0.0f);
    }
    return result;
}

// LEARN S1 (BitNet b1.58 2B4T technical report 2026): squared ReLU is
// the activation BitNet ships with because SwiGLU in low-precision
// regimes (ternary BitLinear, FP8) suffers occasional activation spikes
// that overflow the dynamic range and diverge loss after extended
// training.  Squared ReLU is numerically stable in quantized regimes
// and gives comparable expressive power to SwiGLU at moderate scale.
Tensor Tensor::squared_relu() const {
    Tensor result(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_squared_relu_kernel(result.raw_data(), raw_data(), size);
        sync_cuda();
        return result;
    }
#endif
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        const float v = std::max(src[i], 0.0f);
        dst[i] = v * v;
    }
    return result;
}

Tensor Tensor::squared_relu_backward(const Tensor& dy,
                                      const Tensor& pre_activation) {
    if (dy.shape != pre_activation.shape) {
        throw std::runtime_error(
            "squared_relu_backward: dy and pre_activation must have same shape");
    }
    if (dy.get_device() != pre_activation.get_device()) {
        throw std::runtime_error(
            "squared_relu_backward: dy and pre_activation must be on same device");
    }
    Tensor result(dy.shape.dims, dy.get_device());
#ifdef USE_CUDA
    if (use_gpu_fast_path(dy, pre_activation) && gpu_custom_kernels_supported()) {
        launch_squared_relu_backward_kernel(result.raw_data(),
                                              dy.raw_data(),
                                              pre_activation.raw_data(),
                                              dy.size);
        sync_cuda();
        return result;
    }
#endif
    const float* dy_ptr = dy.data();
    const float* pre_ptr = pre_activation.data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < dy.size; ++i) {
        const float relu_x = std::max(pre_ptr[i], 0.0f);
        dst[i] = dy_ptr[i] * 2.0f * relu_x;
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_sigmoid_kernel(result.raw_data(), raw_data(), size);
        sync_cuda();
        return result;
    }
#endif
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    int rank = static_cast<int>(shape.size());
    dim = normalize_dim(dim, rank);

    Tensor result(shape.dims, device);
    int axis = shape[dim];
    int inner = 1;
    for (int i = dim + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    int outer = size / (axis * inner);

#ifdef USE_CUDA
    if (dim == rank - 1 && use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_softmax_kernel(result.raw_data(), raw_data(), outer, axis);
        sync_cuda();
        return result;
    }
#endif

    warn_host_fallback_once("softmax(dim!=last)", device);
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int row_index = 0; row_index < outer * inner; ++row_index) {
        const int outer_idx = row_index / inner;
        const int inner_idx = row_index % inner;
        const int base = outer_idx * axis * inner + inner_idx;
        float max_val = src[base];
        for (int axis_idx = 1; axis_idx < axis; ++axis_idx) {
            max_val = std::max(max_val, src[base + axis_idx * inner]);
        }

        float sum = 0.0f;
        for (int axis_idx = 0; axis_idx < axis; ++axis_idx) {
            const int offset = base + axis_idx * inner;
            dst[offset] = std::exp(src[offset] - max_val);
            sum += dst[offset];
        }

        const float inv_sum = 1.0f / std::max(sum, 1e-8f);
        for (int axis_idx = 0; axis_idx < axis; ++axis_idx) {
            dst[base + axis_idx * inner] *= inv_sum;
        }
    }

    return result;
}

Tensor Tensor::rmsnorm(float eps) const {
    Tensor result(shape.dims, device);
    int inner = shape.back();
    int outer = size / inner;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_rmsnorm_kernel(result.raw_data(), raw_data(), outer, inner,
                              0, 0);
        sync_cuda();
        return result;
    }
#endif

    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < outer; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < inner; ++j) {
            float value = src[i * inner + j];
            sum_sq += value * value;
        }

        float inv = 1.0f / std::sqrt(sum_sq / std::max(inner, 1) + eps);
        for (int j = 0; j < inner; ++j) {
            dst[i * inner + j] = src[i * inner + j] * inv;
        }
    }

    return result;
}

Tensor Tensor::clamp(float min_val, float max_val) const {
    Tensor result(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_clamp_kernel(result.raw_data(), raw_data(), min_val, max_val,
                            size);
        sync_cuda();
        return result;
    }
#endif
    const float* src = data();
    float* dst = result.data();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = std::clamp(src[i], min_val, max_val);
    }
    return result;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    int rank = static_cast<int>(shape.size());
    dim = normalize_dim(dim, rank);

    if (rank == 1) {
        Tensor result({1}, device);
        float total = 0.0f;
        for (int i = 0; i < size; ++i) {
            total += data()[i];
        }
        result.data()[0] = total;
        return result;
    }

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported() && rank == 2) {
        const int rows = shape[0];
        const int cols = shape[1];
        std::vector<int> gpu_out_dims = shape.dims;
        if (keepdim) {
            gpu_out_dims[dim] = 1;
        } else {
            gpu_out_dims.erase(gpu_out_dims.begin() + dim);
            if (gpu_out_dims.empty()) {
                gpu_out_dims.push_back(1);
            }
        }
        Tensor result(gpu_out_dims, device);
        if (dim == 0) {
            launch_mean_kernel(result.raw_data(), raw_data(), 1, rows, cols);
            launch_scale_inplace_kernel(result.raw_data(),
                                         static_cast<float>(rows), cols);
            sync_cuda();
            return result;
        }
        if (dim == 1) {
            launch_mean_kernel(result.raw_data(), raw_data(), rows, cols, 1);
            launch_scale_inplace_kernel(result.raw_data(),
                                         static_cast<float>(cols), rows);
            sync_cuda();
            return result;
        }
    }
#endif

    std::vector<int> out_dims = shape.dims;
    if (keepdim) {
        out_dims[dim] = 1;
    } else {
        out_dims.erase(out_dims.begin() + dim);
    }
    if (out_dims.empty()) {
        out_dims.push_back(1);
    }

    Tensor result(out_dims, device);
    std::vector<int> idx = make_indices(rank);
    for (int flat = 0; flat < size; ++flat) {
        int remaining = flat;
        for (int d = rank - 1; d >= 0; --d) {
            idx[d] = remaining % shape[d];
            remaining /= shape[d];
        }

        std::vector<int> out_idx = idx;
        if (keepdim) {
            out_idx[dim] = 0;
        } else {
            out_idx.erase(out_idx.begin() + dim);
            if (out_idx.empty()) {
                out_idx.push_back(0);
            }
        }

        result.at(out_idx) += data()[flat];
    }

    return result;
}

float Tensor::norm() const {
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        CudaBuffer<float> d_sum_sq(1);
        cudaMemset(d_sum_sq.get(), 0, sizeof(float));
        launch_norm_kernel(d_sum_sq.get(), raw_data(), size);
        sync_cuda();
        const float sum_sq = copy_scalar_from_device(d_sum_sq.get());
        return std::sqrt(sum_sq);
    }
#endif
    float sum_sq = 0.0f;
    const float* src = data();
    for (int i = 0; i < size; ++i) {
        sum_sq += src[i] * src[i];
    }
    return std::sqrt(sum_sq);
}

Tensor Tensor::clone() const {
    // Cópia integral sobrescreve tudo — sem zero-fill.
    Tensor result = Tensor::uninitialized(shape.dims, device);
    if (size > 0) {
#ifdef USE_CUDA
        if (device == Device::GPU) {
            // raw_data() + cópia async no stream 0: o caminho antigo chamava
            // data() nos DOIS lados (cada um podendo drenar o device via
            // sync_host_access) e ainda sincronizava o copy-stream — três
            // barreiras por clone, pagas em cada save de treino por camada.
            copy_tensor_bytes(result.raw_data(), result.device, raw_data(), device,
                              size * sizeof(float), /*async_d2d=*/true);
            return result;
        }
#endif
        copy_tensor_bytes(result.data(), result.device, data(), device, size * sizeof(float));
    }
    return result;
}

Tensor Tensor::to(Device dev) const {
    if (device == dev) {
        return *this;
    }

    // Cópia integral sobrescreve tudo — sem zero-fill.
    Tensor result = Tensor::uninitialized(shape.dims, dev);
    if (size > 0) {
        copy_tensor_bytes(result.data(), result.device, data(), device, size * sizeof(float));
    }
    return result;
}

Tensor Tensor::cpu() const {
    return to(Device::CPU);
}

void Tensor::sync_host_access() const {
#ifdef USE_CUDA
    if (device == Device::GPU && size > 0) {
        // Cheap-when-idle barrier:
        //   * cudaStreamQuery(0) returns cudaSuccess in microseconds when
        //     the default stream has no in-flight work — the common case
        //     for back-to-back data() reads after the first sync.
        //   * Only when there IS pending work (cudaErrorNotReady) do we
        //     pay the cost of cudaDeviceSynchronize.
        // This lets eager-sync mode stay viable for production: a chain
        // of data() reads after a single kernel launch syncs once and
        // then no-ops, instead of bottlenecking on N cudaDeviceSync.
        const cudaError_t pending = cudaStreamQuery(0);
        if (pending != cudaSuccess) {
            // Drain the default stream and reset any sticky error state
            // so a subsequent kernel launch isn't poisoned by the
            // ErrorNotReady we just observed.
            cudaDeviceSynchronize();
            (void)cudaGetLastError();
        }
    }
#endif
}

bool Tensor::eager_gpu_sync_enabled() {
    // Retained as a public predicate for tests / diagnostics.  No
    // longer consulted by data() — auto-sync is now mandatory because
    // Pascal+Windows UM cannot be relied upon for safe host access.
    return true;
}

float* Tensor::data() {
    // ALWAYS synchronize when device == GPU.  This is the safety net
    // for the Pascal+Windows UM bug class: any host code that obtains
    // a pointer through data() can rely on the underlying memory being
    // up-to-date and host-accessible by the time the call returns.
    //
    // Cost: on CPU, no-op.  On GPU, one cudaStreamQuery (microseconds
    // when idle) plus an unconditional cudaDeviceSynchronize when the
    // default stream has pending work.  Hot kernel-launch paths that
    // do NOT need this safety should call raw_data() instead, which
    // returns the pointer with zero sync overhead — see callers in
    // jamba.cpp / mamba2.cpp / trainer.cpp etc.
    sync_host_access();
    return data_ptr.get();
}

const float* Tensor::data() const {
    sync_host_access();
    return data_ptr.get();
}

void Tensor::copy_from(const Tensor& other) {
    if (size != other.size) {
        throw std::runtime_error("Tensor size mismatch in copy_from");
    }
    if (shape != other.shape) {
        throw std::runtime_error("Tensor shape mismatch in copy_from");
    }
    if (size > 0) {
#ifdef USE_CUDA
        if (device == Device::GPU && other.device == Device::GPU) {
            copy_tensor_bytes(raw_data(), device, other.raw_data(), other.device,
                              size * sizeof(float), /*async_d2d=*/true);
            return;
        }
#endif
        copy_tensor_bytes(data(), device, other.data(), other.device, size * sizeof(float));
    }
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    int inferred_index = -1;
    long long known_product = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (inferred_index != -1) {
                throw std::runtime_error("Only one inferred dimension is supported");
            }
            inferred_index = static_cast<int>(i);
            continue;
        }
        if (new_shape[i] <= 0) {
            throw std::runtime_error("Reshape dimensions must be positive or -1");
        }
        known_product *= new_shape[i];
    }

    if (inferred_index != -1) {
        if (known_product <= 0 || size % known_product != 0) {
            throw std::runtime_error("Reshape size mismatch");
        }
        new_shape[inferred_index] = size / static_cast<int>(known_product);
    }

    // reshape/squeeze/unsqueeze return a VIEW: the result shares the same
    // underlying data buffer (O(1), no copy).  We deliberately drop the
    // parent's gradient here -- it carries the parent's shape and would
    // otherwise be a grad aliased to a mismatched shape (a real footgun).
    // Use clone() when an independent, copy-backed tensor is required.
    Tensor reshaped = *this;
    TensorShape new_tensor_shape(new_shape);
    if (static_cast<int>(new_tensor_shape.numel()) != size) {
        throw std::runtime_error("Reshape size mismatch");
    }
    reshaped.shape = new_tensor_shape;
    reshaped.grad = nullptr;
    return reshaped;
}

Tensor Tensor::squeeze(int dim) const {
    if (shape.empty()) {
        return clone();
    }

    std::vector<int> out_dims = shape.dims;
    if (dim < 0) {
        out_dims.erase(
            std::remove(out_dims.begin(), out_dims.end(), 1),
            out_dims.end());
    } else {
        dim = normalize_dim(dim, static_cast<int>(shape.size()));
        if (out_dims[dim] == 1) {
            out_dims.erase(out_dims.begin() + dim);
        }
    }

    if (out_dims.empty()) {
        out_dims.push_back(1);
    }

    return reshape(out_dims);
}

Tensor Tensor::unsqueeze(int dim) const {
    int rank = static_cast<int>(shape.size());
    if (dim < 0) {
        dim += rank + 1;
    }
    if (dim < 0 || dim > rank) {
        throw std::out_of_range("Unsqueeze dimension out of range");
    }

    std::vector<int> out_dims = shape.dims;
    out_dims.insert(out_dims.begin() + dim, 1);
    return reshape(out_dims);
}

Tensor Tensor::slice(int dim, int start, int end) const {
    int rank = static_cast<int>(shape.size());
    dim = normalize_dim(dim, rank);
    start = std::max(start, 0);
    end = std::min(end, shape[dim]);
    if (start >= end) {
        throw std::runtime_error("Invalid slice bounds");
    }

    std::vector<int> out_dims = shape.dims;
    out_dims[dim] = end - start;
    // Ambos os caminhos (memcpy dim-0 e loop genérico) escrevem todo elemento.
    Tensor result = Tensor::uninitialized(out_dims, device);

    if (dim == 0 && rank >= 1) {
        const int inner = size / shape[0];
        const size_t bytes =
            static_cast<size_t>((end - start) * inner) * sizeof(float);
#ifdef USE_CUDA
        if (device == Device::GPU) {
            copy_tensor_bytes(result.raw_data(), result.device,
                              raw_data() + static_cast<size_t>(start * inner), device,
                              bytes, /*async_d2d=*/true);
            return result;
        }
#endif
        const float* src_ptr = data() + static_cast<size_t>(start * inner);
        copy_tensor_bytes(result.data(), result.device, src_ptr, device, bytes);
        return result;
    }

    warn_host_fallback_once("slice(dim!=0)", device);
    std::vector<int> idx = make_indices(rank);
    for (int flat = 0; flat < result.size; ++flat) {
        int remaining = flat;
        for (int d = rank - 1; d >= 0; --d) {
            idx[d] = remaining % out_dims[d];
            remaining /= out_dims[d];
        }
        std::vector<int> src_idx = idx;
        src_idx[dim] += start;
        result.data()[flat] = get(src_idx);
    }

    return result;
}

int Tensor::get_flat_index(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Tensor index rank mismatch");
    }

    int flat = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape.dims[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        flat += static_cast<int>(indices[i] * shape.strides[i]);
    }
    return flat;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data()[get_flat_index(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    return data()[get_flat_index(indices)];
}

float Tensor::get(const std::vector<int>& indices) const {
    return at(indices);
}

Tensor Tensor::rmsnorm_backward(const Tensor& grad, const Tensor& x_norm) const {
    Tensor dx(shape.dims, device);
    int inner = shape.back();
    int outer = size / inner;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, grad) && x_norm.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        launch_rmsnorm_backward_kernel(dx.raw_data(), grad.raw_data(),
                                        x_norm.raw_data(), raw_data(),
                                        outer, inner);
        sync_cuda();
        return dx;
    }
#endif

    const float* g = grad.data();
    const float* y = x_norm.data();
    const float* x = data();  // *this is the original pre-norm input
    float* dx_ptr = dx.data();
    const float eps = 1e-6f;  // matches the rmsnorm() forward default
#pragma omp parallel for
    for (int i = 0; i < outer; ++i) {
        float sum_sq = 0.0f;
        float dot = 0.0f;
        for (int j = 0; j < inner; ++j) {
            const float xv = x[i * inner + j];
            sum_sq += xv * xv;
            dot += g[i * inner + j] * y[i * inner + j];
        }
        dot /= std::max(inner, 1);
        // Exact RMSNorm Jacobian: dx = (1/rms) * (g - y * mean(g.y)), with
        // rms = sqrt(mean(x^2) + eps).  The prior version omitted the 1/rms
        // factor -- scaling EVERY gradient flowing back through an RMSNorm by
        // rms, a real magnitude error in the training backward.
        const float inv_rms = 1.0f / std::sqrt(
            sum_sq / static_cast<float>(std::max(inner, 1)) + eps);
        for (int j = 0; j < inner; ++j) {
            // Projeção ortogonal removendo a variância (Jacobiano Aproximado RMS)
            dx_ptr[i * inner + j] =
                (g[i * inner + j] - y[i * inner + j] * dot) * inv_rms;
        }
    }
    return dx;
}

#ifdef USE_CUDA
// Persistent device scratch for the cross_entropy GPU fast path.  The trainer
// calls cross_entropy ~batch_size times per training step; the previous per-call
// CudaBuffer<float>(1) + CudaBuffer<int>(rows) did 2 cudaMalloc + 2 cudaFree EACH
// (all synchronizing) -> ~4*batch alloc syncs/step.  Reused buffers remove that.
// Single-threaded training use (the only caller).
static float* ce_loss_scratch() {
    // thread_local: per-thread device scratch so concurrent callers never share
    // the same buffer (matches gemm_lowp_workspace's replica-safety rationale).
    thread_local float* p = [] {
        float* q = nullptr;
        if (cudaMalloc(&q, sizeof(float)) != cudaSuccess) {
            q = nullptr;
            (void)cudaGetLastError();
        }
        return q;
    }();
    return p;
}
static int* ce_target_scratch(int rows) {
    thread_local int* p = nullptr;
    thread_local int cap = 0;
    if (rows <= 0) return nullptr;
    if (rows > cap) {
        if (p) cudaFree(p);
        p = nullptr;
        if (cudaMalloc(&p, static_cast<size_t>(rows) * sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            cap = 0;
            return nullptr;
        }
        cap = rows;
    }
    return p;
}
#endif

std::pair<float, Tensor> Tensor::cross_entropy(const std::vector<int>& target) const {
    int rank = static_cast<int>(shape.size());
    if (rank < 2) {
        throw std::runtime_error("cross_entropy expects rank >= 2 logits");
    }

    int classes = shape.back();
    int rows = size / classes;
    if (static_cast<int>(target.size()) != rows) {
        throw std::runtime_error("cross_entropy target size mismatch");
    }
    for (int row = 0; row < rows; ++row) {
        if (target[row] < 0 || target[row] >= classes) {
            throw std::out_of_range("cross_entropy target out of range");
        }
    }

    Tensor grad(shape.dims, device);
    float loss = 0.0f;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        // Reused device scratch instead of per-call cudaMalloc/cudaFree.
        float* d_loss = ce_loss_scratch();
        int* d_target = ce_target_scratch(rows);
        if (d_loss != nullptr && d_target != nullptr) {
            cudaMemset(d_loss, 0, sizeof(float));
            cudaMemcpy(d_target, target.data(),
                       static_cast<size_t>(rows) * sizeof(int),
                       cudaMemcpyHostToDevice);
            launch_fused_cross_entropy(d_loss, grad.raw_data(), raw_data(),
                                        d_target, rows, classes);
            sync_cuda();
            const float inv_rows = 1.0f / std::max(rows, 1);
            launch_scale_inplace_kernel(grad.raw_data(), inv_rows, grad.size);
            sync_cuda();
            loss = copy_scalar_from_device(d_loss) * inv_rows;
            return {loss, grad};
        }
        // scratch alloc failed (rare) — fall through to the host CE path.
    }
#endif

    const float* logits = data();
    float* grad_ptr = grad.data();

    for (int row = 0; row < rows; ++row) {
        const float* row_ptr = logits + row * classes;
        float max_logit = row_ptr[0];
        for (int c = 1; c < classes; ++c) {
            max_logit = std::max(max_logit, row_ptr[c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; ++c) {
            grad_ptr[row * classes + c] = std::exp(row_ptr[c] - max_logit);
            sum_exp += grad_ptr[row * classes + c];
        }

        float inv_sum = 1.0f / std::max(sum_exp, 1e-8f);
        int target_class = target[row];
        if (target_class < 0 || target_class >= classes) {
            throw std::out_of_range("cross_entropy target out of range");
        }

        for (int c = 0; c < classes; ++c) {
            grad_ptr[row * classes + c] *= inv_sum;
        }

        float prob = std::max(grad_ptr[row * classes + target_class], 1e-8f);
        loss += -std::log(prob);
        grad_ptr[row * classes + target_class] -= 1.0f;
    }

    float inv_rows = 1.0f / std::max(rows, 1);
    for (int i = 0; i < grad.size; ++i) {
        grad_ptr[i] *= inv_rows;
    }

    return {loss * inv_rows, grad};
}

std::pair<float, Tensor> Tensor::mse_loss(const Tensor& target) const {
    if (size != target.size) {
        throw std::runtime_error("mse_loss shape mismatch");
    }

    Tensor grad(shape.dims, device);
    const float* src = data();
    const float* tgt = target.data();
    float* grad_ptr = grad.data();
    float loss = 0.0f;

    float inv_size = 1.0f / static_cast<float>(std::max<int64_t>(size, 1));
    for (int i = 0; i < size; ++i) {
        float diff = src[i] - tgt[i];
        loss += diff * diff;
        grad_ptr[i] = 2.0f * diff * inv_size;
    }

    return {loss * inv_size, grad};
}

void Tensor::print(const std::string& name, int max_elements) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << shape.dims[i];
    }
    std::cout << "] Device: " << (device == Device::CPU ? "CPU" : "GPU");
    std::cout << " Values: ";
    int limit = static_cast<int>(std::min<int64_t>(size, max_elements));
    for (int i = 0; i < limit; ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << data()[i];
    }
    if (limit < size) {
        std::cout << ", ...";
    }
    std::cout << std::endl;
}

Tensor Tensor::from_blob(void* ptr, std::vector<int> s, Device d, bool take_ownership) {
    Tensor t;
    t.shape = TensorShape(s);
    t.size = checked_tensor_size(t.shape);
    t.device = d;
    if (t.size > 0 && ptr == nullptr) {
        throw std::runtime_error("from_blob received null pointer for non-empty tensor");
    }
    if (take_ownership) {
        t.data_ptr = std::shared_ptr<float>(static_cast<float*>(ptr), TensorDeleter(d));
    } else {
        t = Tensor(s, d);
        if (t.size > 0) {
            copy_tensor_bytes(t.data(), d, static_cast<const float*>(ptr), d,
                              static_cast<size_t>(t.size) * sizeof(float));
        }
    }
    return t;
}

} // namespace nsos
