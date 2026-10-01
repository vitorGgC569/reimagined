#include "../include/tensor.h"
#include "../include/nsos_arena.h"
#include "../include/nsos_math.h"
#include "../include/nsos/determinism.h"
#include "../include/runtime_execution_identity.h"
#include "../include/gpu_gemm_provider.h"
#include "../include/tensor_iterator.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/cuda/device_buffer.h"
#include "../include/cuda/pinned_buffer.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <set>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#endif

namespace nsos {

namespace {

struct AtomicGpuTransferStats {
    std::atomic<uint64_t> h2d_calls{0};
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_calls{0};
    std::atomic<uint64_t> d2h_bytes{0};
    std::atomic<uint64_t> d2d_calls{0};
    std::atomic<uint64_t> d2d_bytes{0};
    std::atomic<uint64_t> h2h_calls{0};
    std::atomic<uint64_t> h2h_bytes{0};
    std::atomic<uint64_t> device_synchronizations{0};
    std::atomic<uint64_t> stream_synchronizations{0};
};

AtomicGpuTransferStats& gpu_transfer_counters() {
    static AtomicGpuTransferStats counters;
    return counters;
}

int checked_tensor_size(const TensorShape& shape) {
    // Ponto ÚNICO de validação de shape (chamado por todo construtor de Tensor
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
        const size_t dim = static_cast<size_t>(d);
        if (dim != 0 && numel > kMaxElements / dim) {
            throw std::overflow_error(
                "Tensor element count exceeds NSOS v1 int storage limit");
        }
        numel *= dim;
    }
    return static_cast<int>(numel);
}

int checked_int_product(int lhs, int rhs, const char* label) {
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument(
            std::string(label) + " received a negative factor");
    }
    const int64_t product =
        static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    if (product > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            std::string(label) + " exceeds the tensor index range");
    }
    return static_cast<int>(product);
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

}  // namespace

TensorRandomState capture_tensor_random_state() {
    return tensor_rng();
}

void restore_tensor_random_state(const TensorRandomState& state) {
    tensor_rng() = state;
}

namespace {

std::atomic<uint64_t>& matmul_precision_policy_epoch_storage() {
    static std::atomic<uint64_t> epoch{1};
    return epoch;
}

#ifdef USE_CUDA
void cublas_check(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS failure in ") + op);
    }
}

// thread_local (K6 replica-safety class): a cublasHandle_t must not be used
// concurrently from multiple threads (cuBLAS docs) — the HTTP server drives
// concurrent inference replicas that all call Tensor::matmul.  Per-thread
// handles are the vendored-recommended pattern; all still launch on the legacy
// stream 0, so ordering semantics are unchanged for the single-threaded
// training path (one handle, same behavior as before).
struct ThreadBlasState {
    int availability = -1;
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    bool stream_bound = false;

    ~ThreadBlasState() noexcept {
        if (handle != nullptr) {
            (void)cublasDestroy(handle);
            handle = nullptr;
        }
    }
};

ThreadBlasState& gpu_blas_state() {
    thread_local ThreadBlasState state;
    return state;
}

bool gpu_blas_supported() {
    ThreadBlasState& state = gpu_blas_state();
    if (state.availability != -1) {
        return state.availability == 1;
    }

    const cudaError_t context_status = cudaFree(nullptr);
    if (context_status != cudaSuccess) {
        state.availability = 0;
        return false;
    }

    if (cublasCreate(&state.handle) != CUBLAS_STATUS_SUCCESS) {
        state.handle = nullptr;
        state.availability = 0;
        return false;
    }

#ifdef NSOS_CUDA_PTDS
    // Per-thread-default-stream build: our __global__ launches go to
    // cudaStreamPerThread (nvcc --default-stream=per-thread), but cuBLAS
    // interprets a null stream as the LEGACY stream regardless of that flag.
    // Pin the handle to the per-thread stream explicitly so GEMMs stay
    // ordered with the surrounding kernels — and get RECORDED when a decode
    // CUDA graph captures that stream.
    if (cublasSetStream(state.handle, cudaStreamPerThread) !=
        CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(state.handle);
        state.handle = nullptr;
        state.availability = 0;
        return false;
    }
#endif

    state.availability = 1;
    return true;
}

cublasHandle_t cublas_handle() {
    gpu::require_classic_gemm_provider();
    if (!gpu_blas_supported()) {
        return nullptr;
    }
    auto& state = gpu_blas_state();
    const auto stream = gpu::current_stream();
    if (!state.stream_bound || state.stream != stream) {
        if (cublasSetStream(state.handle, stream) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Cannot bind BLAS to the execution stream");
        state.stream = stream;
        state.stream_bound = true;
    }
    return state.handle;
}

// AUDIT (post BATCH 4): the mixed-precision GEMM path used to do
//   cudaMalloc(a_low); cudaMalloc(b_low); <gemm>; cudaFree; cudaFree;
// per matmul call.  Each cudaMalloc/cudaFree pair costs ~1-10 ms AND
// forces an implicit stream synchronization — which on a 12-layer
// model with ~400 matmuls per training step burned 2-8 s/step of pure
// allocator overhead, completely masking the Tensor Core speedup that
// the user enabled with NSOS_MIXED_PRECISION=bf16.
//
// This workspace caches the BF16/FP16 staging buffers across matmul calls and
// only reallocates when a larger tensor shape comes through. Each host thread
// owns one instance because inference replicas execute concurrently.
struct GemmLowpWorkspace {
    cuda_detail::DeviceBuffer<unsigned char> a;
    cuda_detail::DeviceBuffer<unsigned char> b;

    // Ensure both buffers hold at least the requested bytes. The caller treats
    // false as a hard mixed-precision contract failure.
    bool ensure(size_t a_bytes, size_t b_bytes) {
        return a.ensure(a_bytes) != nullptr &&
               b.ensure(b_bytes) != nullptr;
    }

    bool ensure_a(size_t bytes) { return a.ensure(bytes) != nullptr; }
    bool ensure_b(size_t bytes) { return b.ensure(bytes) != nullptr; }

    void* a_pointer() noexcept { return a.get(); }
    void* b_pointer() noexcept { return b.get(); }
};
// thread_local (not a single static): the HTTP server runs concurrent
// inference replicas, each on its own worker thread.  A shared static would let
// two threads cudaFree/cudaMalloc/cast into the same staging buffers at once
// (use-after-free / wrong results).  Per-thread instances make the mixed-
// precision GEMM path replica-safe; single-threaded training is unaffected.
GemmLowpWorkspace& gemm_lowp_workspace() {
    thread_local GemmLowpWorkspace ws;
    return ws;
}

struct AtomicLowpWeightCacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> version_refreshes{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> budget_bypasses{0};
    std::atomic<uint64_t> allocation_failures{0};
    std::atomic<size_t> resident_bytes{0};
    std::atomic<size_t> peak_resident_bytes{0};
    std::atomic<uint64_t> access_clock{1};
};

AtomicLowpWeightCacheStats& lowp_weight_cache_counters() {
    static AtomicLowpWeightCacheStats counters;
    return counters;
}

size_t configured_lowp_weight_cache_budget_bytes() {
    static const size_t budget = [] {
        // 192 MiB covers the audited ~135 MiB converted training set while
        // remaining explicitly bounded. Operators may lower it to zero or
        // raise it deliberately; malformed/overflowing values fail closed.
        constexpr uint64_t kDefaultMiB = 192;
        constexpr uint64_t kMaximumMiB = 16ULL * 1024ULL;
        const char* value = std::getenv("NSOS_LOWP_WEIGHT_CACHE_MIB");
        uint64_t mib = kDefaultMiB;
        if (value != nullptr && *value != '\0') {
            const char* end = value + std::strlen(value);
            const auto parsed = std::from_chars(value, end, mib);
            if (parsed.ec != std::errc() || parsed.ptr != end ||
                mib > kMaximumMiB) {
                throw std::invalid_argument(
                    "NSOS_LOWP_WEIGHT_CACHE_MIB must be an integer in "
                    "[0, 16384]");
            }
        }
        constexpr uint64_t kMiB = 1024ULL * 1024ULL;
        return static_cast<size_t>(mib * kMiB);
    }();
    return budget;
}

bool same_storage_owner(const std::weak_ptr<float>& lhs,
                        const std::shared_ptr<float>& rhs) {
    return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

class LowpWeightCache {
 public:
    ~LowpWeightCache() noexcept {
        while (!entries_.empty()) {
            release_entry(entries_.size() - 1, false);
        }
    }

    void* find_or_convert(const Tensor& weight, int mixed_mode,
                          uint64_t content_version) {
        auto& counters = lowp_weight_cache_counters();
        const size_t bytes = static_cast<size_t>(weight.size) * 2u;
        const size_t budget = configured_lowp_weight_cache_budget_bytes();
        if (bytes == 0 || budget == 0 || bytes > budget) {
            counters.budget_bypasses.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        int device_id = -1;
        if (cudaGetDevice(&device_id) != cudaSuccess) {
            throw std::runtime_error(
                "Cannot resolve active device for low-precision weight cache");
        }
        const uint64_t policy_epoch =
            matmul_precision_policy_epoch_storage().load(
                std::memory_order_relaxed);
        const uint64_t now = counters.access_clock.fetch_add(
            1, std::memory_order_relaxed);

        for (size_t index = entries_.size(); index-- > 0;) {
            if (entries_[index]->storage.expired()) {
                release_entry(index, true);
            }
        }
        for (const auto& candidate : entries_) {
            Entry& entry = *candidate;
            if (!same_storage_owner(entry.storage, weight.data_ptr) ||
                entry.view_start != weight.raw_data() ||
                entry.elements != static_cast<size_t>(weight.size) ||
                entry.shape != weight.shape.dims ||
                entry.device_id != device_id ||
                entry.mixed_mode != mixed_mode ||
                entry.policy_epoch != policy_epoch) {
                continue;
            }
            entry.last_use = now;
            if (entry.content_version == content_version) {
                counters.hits.fetch_add(1, std::memory_order_relaxed);
                return entry.buffer.get();
            }
            launch_cast_f32_to_lowp_kernel(
                entry.buffer.get(), weight.raw_data(), entry.elements,
                mixed_mode);
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Low-precision weight cache refresh failed: ") +
                    cudaGetErrorString(status));
            }
            entry.content_version = content_version;
            counters.version_refreshes.fetch_add(
                1, std::memory_order_relaxed);
            return entry.buffer.get();
        }

        counters.misses.fetch_add(1, std::memory_order_relaxed);
        while (!reserve(bytes, budget)) {
            if (entries_.empty()) {
                counters.budget_bypasses.fetch_add(
                    1, std::memory_order_relaxed);
                return nullptr;
            }
            size_t oldest = 0;
            for (size_t index = 1; index < entries_.size(); ++index) {
                if (entries_[index]->last_use < entries_[oldest]->last_use) {
                    oldest = index;
                }
            }
            release_entry(oldest, true);
        }

        // reserve() publishes the byte budget before any C++ metadata or
        // device allocation. Keep that publication transactional as well:
        // bad_alloc while constructing the entry/vector must not permanently
        // inflate resident_bytes and disable future cache fills.
        bool owns_budget_reservation = true;
        std::unique_ptr<Entry> entry;
        const auto release_uncommitted = [&]() noexcept {
            if (!owns_budget_reservation) return;
            bool storage_released = true;
            if (entry && entry->buffer.get() != nullptr) {
                storage_released = entry->buffer.release();
                if (!storage_released) {
                    // Ambiguous in-flight storage is intentionally leaked; its
                    // reservation remains charged so the configured budget is
                    // still a hard upper bound on reusable cache allocations.
                    entry->buffer.abandon();
                }
            }
            if (storage_released) {
                counters.resident_bytes.fetch_sub(
                    bytes, std::memory_order_relaxed);
            }
            owns_budget_reservation = false;
        };
        try {
            entry = std::make_unique<Entry>();
            entry->storage = weight.data_ptr;
            entry->view_start = weight.raw_data();
            entry->elements = static_cast<size_t>(weight.size);
            entry->shape = weight.shape.dims;
            entry->device_id = device_id;
            entry->mixed_mode = mixed_mode;
            entry->policy_epoch = policy_epoch;
            entry->content_version = content_version;
            entry->last_use = now;
            if (entry->buffer.ensure(bytes) == nullptr) {
                counters.allocation_failures.fetch_add(
                    1, std::memory_order_relaxed);
                release_uncommitted();
                return nullptr;
            }
            launch_cast_f32_to_lowp_kernel(
                entry->buffer.get(), weight.raw_data(), entry->elements,
                mixed_mode);
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Low-precision weight cache fill failed: ") +
                    cudaGetErrorString(status));
            }
            void* pointer = entry->buffer.get();
            entries_.push_back(std::move(entry));
            owns_budget_reservation = false;
            return pointer;
        } catch (...) {
            release_uncommitted();
            throw;
        }
    }

 private:
    struct Entry {
        std::weak_ptr<float> storage;
        const float* view_start = nullptr;
        size_t elements = 0;
        std::vector<int> shape;
        int device_id = -1;
        int mixed_mode = 0;
        uint64_t policy_epoch = 0;
        uint64_t content_version = 0;
        uint64_t last_use = 0;
        cuda_detail::DeviceBuffer<unsigned char> buffer;
    };

    bool reserve(size_t bytes, size_t budget) {
        auto& resident = lowp_weight_cache_counters().resident_bytes;
        size_t current = resident.load(std::memory_order_relaxed);
        while (current <= budget && bytes <= budget - current) {
            if (resident.compare_exchange_weak(
                    current, current + bytes,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                auto& peak =
                    lowp_weight_cache_counters().peak_resident_bytes;
                size_t observed = peak.load(std::memory_order_relaxed);
                while (observed < current + bytes &&
                       !peak.compare_exchange_weak(
                           observed, current + bytes,
                           std::memory_order_relaxed,
                           std::memory_order_relaxed)) {
                }
                return true;
            }
        }
        return false;
    }

    void release_entry(size_t index, bool count_eviction) noexcept {
        auto& entry = entries_[index];
        const size_t bytes = entry->elements * 2u;
        if (entry->buffer.release()) {
            lowp_weight_cache_counters().resident_bytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        } else {
            // A failed runtime release means the allocation may still be in
            // flight. Relinquish it and retain its budget reservation forever
            // rather than reusing/freeing ambiguous storage.
            entry->buffer.abandon();
        }
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
        if (count_eviction) {
            lowp_weight_cache_counters().evictions.fetch_add(
                1, std::memory_order_relaxed);
        }
    }

    std::vector<std::unique_ptr<Entry>> entries_;
};

LowpWeightCache& lowp_weight_cache() {
    thread_local LowpWeightCache cache;
    return cache;
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
        record_gpu_device_synchronization();
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
    const cudaError_t status =
        cudaMemcpy(&value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("GPU scalar download failed on ") +
            NSOS_GPU_BACKEND_NAME + ": " +
            cudaGetErrorString(status));
    }
    record_gpu_transfer(Device::CPU, Device::GPU, sizeof(T));
    return value;
}

float* scalar_reduction_scratch() {
    thread_local cuda_detail::DeviceBuffer<float> buffer;
    return buffer.ensure(1);
}
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
std::atomic<int>& pool_release_failure_countdown() {
    static std::atomic<int> countdown{-1};
    return countdown;
}

bool inject_pool_release_failure() noexcept {
#ifdef NSOS_ENABLE_TEST_HOOKS
    auto& countdown = pool_release_failure_countdown();
    int current = countdown.load(std::memory_order_relaxed);
    while (current >= 0) {
        const int next = current == 0 ? -1 : current - 1;
        if (countdown.compare_exchange_weak(
                current, next, std::memory_order_relaxed)) {
            return current == 0;
        }
    }
#endif
    return false;
}

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
struct TensorCopyStream {
    cudaStream_t stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    int device_id = -1;
    std::mutex mutex;

    TensorCopyStream() {
        std::string selection_error;
        if (!gpu::select_preferred_device(
                &device_id, &selection_error)) {
            std::fprintf(
                stderr,
                "[GPU] tensor copy stream unavailable; using checked "
                "blocking copies: %s\n",
                selection_error.c_str());
            stream = nullptr;
            device_id = -1;
            return;
        }
        const cudaError_t stream_status = cudaStreamCreate(&stream);
        if (stream_status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[GPU] tensor copy stream creation failed; using checked "
                "blocking copies on %s: %s\n",
                NSOS_GPU_BACKEND_NAME, cudaGetErrorString(stream_status));
            stream = nullptr;
            device_id = -1;
            (void)cudaGetLastError();
            return;
        }
        const cudaError_t event_status = cudaEventCreate(&producer_ready);
        if (event_status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[GPU] tensor copy producer event creation failed; using "
                "checked blocking copies on %s: %s\n",
                NSOS_GPU_BACKEND_NAME, cudaGetErrorString(event_status));
            gpu::report_cleanup_status(
                cudaStreamDestroy(stream),
                "tensor copy stream rollback destruction");
            stream = nullptr;
            producer_ready = nullptr;
            device_id = -1;
        }
    }

    ~TensorCopyStream() noexcept {
        const cudaError_t selection_status =
            device_id >= 0 ? cudaSetDevice(device_id) : cudaSuccess;
        if (selection_status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[GPU] tensor copy stream cleanup could not select device "
                "%d on %s: %s\n",
                device_id, NSOS_GPU_BACKEND_NAME,
                cudaGetErrorString(selection_status));
            (void)cudaGetLastError();
            return;
        }
        if (producer_ready != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(producer_ready),
                "tensor copy producer event destruction");
            producer_ready = nullptr;
        }
        if (stream != nullptr) {
            gpu::report_cleanup_status(
                cudaStreamDestroy(stream),
                "tensor copy stream destruction");
            stream = nullptr;
        }
    }
};

static TensorCopyStream& tensor_copy_stream() {
    static TensorCopyStream owned_stream;
    return owned_stream;
}

// (movida p/ escopo de namespace — ver apos Tensor::uninitialized)

#else
// (variante CPU fundida na definicao unica)

#endif

} // namespace

#ifdef USE_CUDA
namespace {

// ── GPU caching allocator (PyTorch-style caching allocator) ───────────────
// Per-step training allocates/frees many device tensors with recurring exact
// sizes. Driver allocation/free calls are
// heavyweight, partially-synchronizing driver calls; doing dozens per step adds
// avoidable overhead AND makes CUDA Graph capture impossible (allocation is
// illegal during capture).  This pool keeps freed blocks on per-exact-size free
// lists and hands them back on the next request -> with static shapes, reuse is
// perfect (zero fragmentation) and addresses are stable across steps (the
// precondition for graph replay; warm the pool with one step, then capture
// hits only the free list -> no driver alloc inside the captured region).
// Disable with NSOS_GPU_POOL=0 (falls back to raw driver allocation/free).
class ManagedPool {
private:
    enum class MemoryPolicy {
        Automatic,
        Device,
        Managed,
    };

    struct Allocation {
        size_t bytes = 0;
        bool managed = false;
        uint64_t stream_domain = 0;
        bool quarantined = false;
        bool retained_release = false;
        bool retryable_release = false;
        bool cached = false;
    };

    struct CachedBlock {
        void* pointer = nullptr;
        uint64_t stream_domain = 0;
    };

    static size_t cache_key(size_t bytes, bool managed) {
        // bin_bytes() always returns a 64 KiB multiple, so the low bit is free
        // to distinguish allocation provenance without changing capacity math.
        return bytes | (managed ? size_t{1} : size_t{0});
    }

    static uint64_t current_stream_domain() noexcept {
        const auto stream = gpu::current_stream();
        if (stream != nullptr && stream != cudaStreamPerThread)
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
#ifdef NSOS_CUDA_PTDS
        // Under per-thread default-stream semantics, stream 0 work from two
        // host threads is not mutually ordered. A cached address may therefore
        // be reused only by the thread/domain that last checked it out.
        static std::atomic<uint64_t> next_domain{1};
        thread_local const uint64_t domain =
            next_domain.fetch_add(1, std::memory_order_relaxed);
        return (domain << 1) | 1;
#else
        // The legacy default stream globally orders all users.
        return 0;
#endif
    }

    void update_peaks_locked() noexcept {
        peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
        const size_t reserved =
            cached_bytes_ > std::numeric_limits<size_t>::max() - live_bytes_
                ? std::numeric_limits<size_t>::max()
                : cached_bytes_ + live_bytes_;
        peak_reserved_bytes_ = std::max(peak_reserved_bytes_, reserved);
    }

    bool release_owned_locked(
        std::unordered_map<void*, Allocation>::iterator allocation,
        bool was_cached) noexcept {
        void* pointer = allocation->first;
        const size_t bytes = allocation->second.bytes;
        const bool injected_failure = inject_pool_release_failure();
        cudaError_t status = static_cast<cudaError_t>(1);
        if (!injected_failure) {
            // Attribute the following status to the release itself instead of
            // a stale launch error left in the calling thread's runtime slot.
            // A prior error is itself a hard failure: retain the pointer
            // without calling cudaFree because its ownership would otherwise
            // become ambiguous.
            status = cudaGetLastError();
            if (status == cudaSuccess) {
                status = cudaFree(pointer);
            }
        }
        if (status != cudaSuccess) {
            if (!injected_failure) {
                (void)cudaGetLastError();
            }
            ++release_failures_;
            allocation->second.retained_release = true;
            allocation->second.retryable_release = injected_failure;
            if (was_cached) {
                cached_bytes_ -=
                    cached_bytes_ >= bytes ? bytes : cached_bytes_;
                if (bytes <=
                    std::numeric_limits<size_t>::max() - live_bytes_) {
                    live_bytes_ += bytes;
                    update_peaks_locked();
                }
                allocation->second.cached = false;
            }
            return false;
        }
        if (was_cached) {
            cached_bytes_ -=
                cached_bytes_ >= bytes ? bytes : cached_bytes_;
        } else {
            live_bytes_ -=
                live_bytes_ >= bytes ? bytes : live_bytes_;
        }
        live_.erase(allocation);
        return true;
    }

public:
    static ManagedPool& instance() {
        static ManagedPool pool;
        return pool;
    }

    ~ManagedPool() noexcept {
        bool device_bound = device_id_ < 0;
        if (device_id_ >= 0) {
            const cudaError_t select_status =
                cudaSetDevice(device_id_);
            if (select_status != cudaSuccess) {
                (void)cudaGetLastError();
            } else {
                device_bound = true;
            }
        }
        // live_ is the authoritative ownership registry and contains cached,
        // quarantined, and checked-out blocks exactly once each.
        if (device_bound) {
            for (const auto& entry : live_) {
                if (entry.first != nullptr) {
                    const cudaError_t status = cudaFree(entry.first);
                    if (status != cudaSuccess) (void)cudaGetLastError();
                }
            }
        }
        live_.clear();
        free_.clear();
        captured_.clear();
        quarantined_blocks_ = 0;
        cached_bytes_ = 0;
        live_bytes_ = 0;
    }

    bool use_managed_memory() const noexcept {
        // Strict execution is a hard safety/performance contract: an
        // environment preference must never silently turn a production model
        // back into page-faulting Unified Memory.
        if (strict_gpu_execution()) return false;
        if (memory_policy_ == MemoryPolicy::Device) return false;
        return true;
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
        const size_t quantum =
            bytes < (1ull << 20) ? k64 : k2m;
        if (bytes >
            std::numeric_limits<size_t>::max() - (quantum - 1)) {
            return 0;
        }
        return ((bytes + quantum - 1) / quantum) * quantum;
    }

    // ── CUDA-graph capture guard ─────────────────────────────────────────
    // Between begin_capture() and release_capture() the pool is
    // capture-safe: (1) it NEVER calls cudaFree/trim/cudaMemGetInfo (all
    // synchronize the device, which is illegal during stream capture), and
    // (2) every buffer allocated during capture is QUARANTINED on free
    // instead of returning to the free-list, so no address the captured
    // graph writes on replay is ever handed to another tensor.  The graph's
    // decode owner calls release_capture() only after the graph is
    // destroyed, at which point the quarantine is returned to the driver.
    void begin_capture() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (capturing_ || !captured_.empty() ||
            quarantined_blocks_ != 0) {
            ++capture_contract_violations_;
            throw std::logic_error(
                "GPU pool capture cannot be nested or replaced before "
                "the prior capture is released");
        }
        capturing_ = true;
    }
    void end_capture() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!capturing_) {
            ++capture_contract_violations_;
            throw std::logic_error(
                "GPU pool capture end has no matching begin");
        }
        capturing_ = false;  // captured addresses persist until release
    }
    void release_capture() noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        if (capturing_) {
            ++capture_contract_violations_;
            return;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            (void)cudaGetLastError();
            for (auto& entry : live_) {
                if (!entry.second.quarantined) {
                    continue;
                }
                ++release_failures_;
                entry.second.quarantined = false;
                entry.second.retained_release = true;
                entry.second.retryable_release = false;
            }
            quarantined_blocks_ = 0;
            captured_.clear();
            return;
        }
        capturing_ = false;
        for (auto allocation = live_.begin();
             allocation != live_.end();) {
            auto current = allocation++;
            if (current->second.quarantined) {
                current->second.quarantined = false;
                if (quarantined_blocks_ > 0) {
                    --quarantined_blocks_;
                }
                (void)release_owned_locked(current, false);
            }
        }
        quarantined_blocks_ = 0;
        captured_.clear();
    }

    void* allocate(size_t bytes, bool managed_memory) {
        if (bytes == 0) return nullptr;
        bind_selected_device();
        if (!enabled_) {
            void* pointer = raw_alloc(bytes, managed_memory);
            if (pointer != nullptr) {
                std::lock_guard<std::mutex> lk(mtx_);
                if (bytes >
                    std::numeric_limits<size_t>::max() - live_bytes_) {
                    const cudaError_t status = cudaFree(pointer);
                    if (status != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                    return nullptr;
                }
                try {
                    const auto inserted = live_.emplace(
                        pointer,
                        Allocation{
                            bytes, managed_memory,
                            current_stream_domain()});
                    if (!inserted.second) {
                        throw std::logic_error(
                            "GPU driver returned an already-owned address");
                    }
                    live_bytes_ += bytes;
                    update_peaks_locked();
                } catch (...) {
                    const cudaError_t status = cudaFree(pointer);
                    if (status != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                    throw;
                }
            }
            return pointer;
        }
        bytes = bin_bytes(bytes);
        if (bytes == 0) return nullptr;
        std::lock_guard<std::mutex> lk(mtx_);
        if (bytes >
            std::numeric_limits<size_t>::max() - live_bytes_) {
            return nullptr;
        }
        auto& bin = free_[cache_key(bytes, managed_memory)];
        live_bytes_ += bytes;
        const uint64_t stream_domain = current_stream_domain();
        auto cached = std::find_if(
            bin.rbegin(), bin.rend(),
            [stream_domain](const CachedBlock& block) {
                return block.stream_domain == stream_domain;
            });
        if (cached != bin.rend()) {
            const size_t index =
                static_cast<size_t>(
                    std::distance(cached, bin.rend()) - 1);
            void* p = bin[index].pointer;
            auto live_allocation = live_.find(p);
            if (live_allocation == live_.end()) {
                if (index + 1 != bin.size()) {
                    bin[index] = bin.back();
                }
                bin.pop_back();
                cached_bytes_ -=
                    cached_bytes_ >= bytes ? bytes : cached_bytes_;
                ++unknown_deallocation_attempts_;
                live_bytes_ -=
                    live_bytes_ >= bytes ? bytes : live_bytes_;
                throw std::logic_error(
                    "GPU pool free-list contains an unowned block");
            }
            if (capturing_) {
                try {
                    const auto inserted = captured_.insert(p);
                    if (!inserted.second) {
                        throw std::logic_error(
                            "GPU pool attempted to reuse an address still "
                            "referenced by a captured graph");
                    }
                } catch (...) {
                    live_bytes_ -=
                        live_bytes_ >= bytes ? bytes : live_bytes_;
                    throw;
                }
            }
            if (index + 1 != bin.size()) {
                bin[index] = bin.back();
            }
            bin.pop_back();
            cached_bytes_ -= bytes;
            live_allocation->second.stream_domain = stream_domain;
            live_allocation->second.cached = false;
            update_peaks_locked();
            return p;
        }
        void* p = raw_alloc(bytes, managed_memory);
        if (!p && !capturing_) {
            // Out of memory: return every cached free block to the driver and
            // retry once (mirrors a caching allocator's empty-cache-on-OOM).
            // Skipped during capture — trim_locked calls cudaFree (illegal
            // mid-capture); a capture-time OOM fails the capture cleanly and
            // the decode falls back to eager.
            trim_locked();
            p = raw_alloc(bytes, managed_memory);
        }
        if (p) {
            auto registered = live_.end();
            bool owns_new_registration = false;
            bool owns_new_capture_record = false;
            try {
                const auto inserted = live_.emplace(
                    p,
                    Allocation{
                        bytes, managed_memory, stream_domain});
                registered = inserted.first;
                owns_new_registration = inserted.second;
                if (!inserted.second) {
                    throw std::logic_error(
                        "GPU driver returned an already-owned address");
                }
                if (capturing_) {
                    const auto captured = captured_.insert(p);
                    owns_new_capture_record = captured.second;
                    if (!captured.second) {
                        throw std::logic_error(
                            "GPU driver returned an address still referenced "
                            "by a captured graph");
                    }
                }
                update_peaks_locked();
            } catch (...) {
                if (owns_new_capture_record) {
                    captured_.erase(p);
                }
                if (owns_new_registration) {
                    (void)release_owned_locked(registered, false);
                } else if (registered != live_.end()) {
                    live_bytes_ -=
                        live_bytes_ >= bytes ? bytes : live_bytes_;
                    ++unknown_deallocation_attempts_;
                } else {
                    live_bytes_ -=
                        live_bytes_ >= bytes ? bytes : live_bytes_;
                    const cudaError_t status = cudaFree(p);
                    if (status != cudaSuccess) {
                        (void)cudaGetLastError();
                    }
                }
                throw;
            }
        } else {
            live_bytes_ -= (live_bytes_ >= bytes ? bytes : live_bytes_);
        }
        return p;
    }

    void deallocate(void* p) noexcept {
        if (!p) return;
        // Shared-pointer deleters must never throw. Rebind this host thread to
        // the owning pool device directly; on an unrecoverable runtime failure
        // retain the allocation for process teardown rather than terminating
        // during stack unwinding.
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            (void)cudaGetLastError();
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = live_.find(p);
            if (it == live_.end()) {
                ++unknown_deallocation_attempts_;
            } else {
                ++release_failures_;
                it->second.retained_release = true;
                it->second.retryable_release = false;
            }
            return;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = live_.find(p);
        if (it == live_.end()) {
            ++unknown_deallocation_attempts_;
            return;
        }
        // A cached/quarantined/retained entry is still present in live_
        // because that map is the pool's ownership registry, but it is no
        // longer checked out to a Tensor.  Treat a second deleter invocation
        // as an invalid deallocation instead of inserting the same address in
        // the free list twice (which would corrupt byte/block accounting and
        // could later hand one physical allocation to two tensors).
        if (it->second.cached || it->second.quarantined ||
            it->second.retained_release) {
            ++unknown_deallocation_attempts_;
            return;
        }
        if (!enabled_) {
            (void)release_owned_locked(it, false);
            return;
        }
        const Allocation allocation = it->second;
        const size_t sz = allocation.bytes;
        // Captured buffer: the graph still references this address on every
        // replay.  Quarantine it (keep live_[p] so release_capture can size
        // the free) — never cache/free it until the graph is destroyed.
        auto cap_it = captured_.find(p);
        if (cap_it != captured_.end()) {
            captured_.erase(cap_it);
            it->second.quarantined = true;
            ++quarantined_blocks_;
            return;
        }
#ifdef NSOS_CUDA_PTDS
        if (allocation.stream_domain != current_stream_domain()) {
            // Cross-thread destruction has no stream-ordering relationship
            // with the allocation's per-thread default stream. cudaFree is the
            // conservative synchronization boundary; caching here would permit
            // a use-after-free/reuse race.
            ++cross_stream_domain_frees_;
            (void)release_owned_locked(it, false);
            return;
        }
#endif
        const bool cache_size_overflow =
            sz > std::numeric_limits<size_t>::max() - cached_bytes_;
        if (cache_size_overflow ||
            (cap_bytes_ != 0 &&
             (sz > cap_bytes_ ||
              cached_bytes_ > cap_bytes_ - sz))) {
            // Cache is full: return this block to the driver instead of caching
            // it, so total managed footprint stays bounded.  During capture
            // cudaFree is illegal, so cache it instead (bounded growth for the
            // duration of a single capture is acceptable).
            if (!capturing_) {
                (void)release_owned_locked(it, false);
                return;
            }
        }
        try {
            free_[cache_key(sz, allocation.managed)].push_back(
                CachedBlock{p, allocation.stream_domain});
        } catch (...) {
            // A deleter cannot propagate allocation failure. Returning the
            // block to the driver is safe and preserves the original
            // exception, if any, that triggered stack unwinding.
            (void)release_owned_locked(it, false);
            return;
        }
        it->second.cached = true;
        cached_bytes_ += sz;
        live_bytes_ -= (live_bytes_ >= sz ? sz : live_bytes_);
        // Guarda de oversubscription UM (auditoria #27): checagem barata por
        // cadência — UM não dá OOM, degrada paginando (T4 a 96% custou steps
        // 6->8s antes do binning).  Acima de 88% de uso do device: poda o
        // cache e avisa uma vez por episódio.  Pulada durante captura
        // (cudaMemGetInfo/cudaFree sincronizam o device -> ilegal).
        // Device-only allocations already fail explicitly and allocate()
        // performs trim+retry. Probe pressure only for Unified Memory; calling
        // cudaMemGetInfo for ordinary device blocks periodically synchronized
        // the strict-GPU training hot path without adding a safety guarantee.
        if (allocation.managed && !capturing_ &&
            ((++dealloc_probe_) & 511u) == 0) {
            ++managed_pressure_probes_;
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
        size_t owned_device_bytes = 0;
        size_t owned_managed_bytes = 0;
        for (const auto& entry : live_) {
            if (entry.second.managed) {
                owned_managed_bytes += entry.second.bytes;
            } else {
                owned_device_bytes += entry.second.bytes;
            }
            if (entry.second.quarantined) {
                s.quarantined_bytes += entry.second.bytes;
                ++s.quarantined_blocks;
            }
            if (entry.second.retained_release) {
                s.retained_release_bytes += entry.second.bytes;
                ++s.retained_release_blocks;
            }
            if (entry.second.cached) {
                s.largest_cached_block_bytes =
                    std::max(s.largest_cached_block_bytes,
                             entry.second.bytes);
            } else {
                s.largest_live_block_bytes =
                    std::max(s.largest_live_block_bytes,
                             entry.second.bytes);
                ++s.live_blocks;
            }
        }
        for (const auto& bin : free_) {
            if (bin.second.empty()) continue;
            const size_t bytes = bin.first & ~size_t{1};
            const size_t aggregate = bytes * bin.second.size();
            if ((bin.first & size_t{1}) != 0) {
                s.managed_cached_bytes += aggregate;
            } else {
                s.device_cached_bytes += aggregate;
            }
            s.cached_blocks += bin.second.size();
            ++s.bins;
        }
        s.cached_bytes =
            s.device_cached_bytes + s.managed_cached_bytes;
        s.device_live_bytes =
            owned_device_bytes >= s.device_cached_bytes
                ? owned_device_bytes - s.device_cached_bytes
                : 0;
        s.managed_live_bytes =
            owned_managed_bytes >= s.managed_cached_bytes
                ? owned_managed_bytes - s.managed_cached_bytes
                : 0;
        s.live_bytes = s.device_live_bytes + s.managed_live_bytes;
        s.allocated_bytes = s.live_bytes;
        if (s.cached_bytes >
            std::numeric_limits<size_t>::max() - s.live_bytes) {
            throw std::overflow_error(
                "GPU pool reserved-byte telemetry overflow");
        }
        s.reserved_bytes = s.cached_bytes + s.live_bytes;
        s.peak_allocated_bytes = peak_live_bytes_;
        s.peak_reserved_bytes = peak_reserved_bytes_;
        s.cached_fragmentation_ratio =
            s.cached_bytes == 0
                ? 0.0
                : 1.0 - static_cast<double>(
                            s.largest_cached_block_bytes) /
                            static_cast<double>(s.cached_bytes);
        s.managed_pressure_probes = managed_pressure_probes_;
        s.managed_advice_failures = managed_advice_failures_;
        s.cross_stream_domain_frees = cross_stream_domain_frees_;
        s.release_failures = release_failures_;
        s.unknown_deallocation_attempts =
            unknown_deallocation_attempts_;
        s.capture_contract_violations =
            capture_contract_violations_;
        s.pool_enabled = enabled_;
        s.capture_active = capturing_;

        // Keep allocator-accounting regressions fail-closed: telemetry must
        // never report a plausible but internally inconsistent footprint.
        if (s.cached_bytes != cached_bytes_ || s.live_bytes != live_bytes_ ||
            owned_device_bytes < s.device_cached_bytes ||
            owned_managed_bytes < s.managed_cached_bytes ||
            s.retained_release_bytes > s.live_bytes ||
            s.quarantined_bytes > s.live_bytes ||
            s.quarantined_blocks != quarantined_blocks_ ||
            s.cached_blocks + s.live_blocks != live_.size()) {
            throw std::logic_error(
                "GPU pool accounting invariant violated");
        }
        return s;
    }
    void trim() {
        bind_selected_device();
        std::lock_guard<std::mutex> lk(mtx_);
        trim_locked();
    }

private:
    ManagedPool() {
        std::string selection_error;
        if (!gpu::select_preferred_device(
                &device_id_, &selection_error)) {
            throw std::runtime_error(
                "GPU memory pool cannot select a device: " +
                selection_error);
        }
        const char* env = std::getenv("NSOS_GPU_POOL");
        enabled_ = !(env && std::string(env) == "0");
        // Explicit GPU models enable strict execution before their first device
        // allocation and always receive true device memory. This policy only
        // controls non-strict legacy/interop callers.
        const char* memory_env = std::getenv("NSOS_GPU_MEMORY");
        if (memory_env != nullptr) {
            const std::string mode(memory_env);
            if (mode == "device") {
                memory_policy_ = MemoryPolicy::Device;
            } else if (mode == "managed") {
                memory_policy_ = MemoryPolicy::Managed;
            } else {
                throw std::runtime_error(
                    "NSOS_GPU_MEMORY must be 'device' or 'managed'");
            }
        }
        // Cap on CACHED (free-list) bytes.  Unified Memory oversubscribes
        // SILENTLY (no OOM — it just thrashes via page eviction), so an
        // unbounded cache (e.g. interleaving an inference and a training working
        // set in one process) could push a small card into thrash.  Default =
        // 60% of device memory; NSOS_GPU_POOL_MAX_MB overrides; "0" = unlimited.
        const char* cap_env = std::getenv("NSOS_GPU_POOL_MAX_MB");
        if (cap_env) {
            uint64_t cap_megabytes = 0;
            const char* cap_end = cap_env + std::strlen(cap_env);
            const auto parsed = std::from_chars(
                cap_env, cap_end, cap_megabytes);
            constexpr uint64_t kBytesPerMegabyte = 1024ull * 1024ull;
            if (cap_env == cap_end ||
                parsed.ec != std::errc{} ||
                parsed.ptr != cap_end ||
                cap_megabytes >
                    static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()) /
                        kBytesPerMegabyte) {
                throw std::runtime_error(
                    "NSOS_GPU_POOL_MAX_MB must be a non-negative integer "
                    "that fits the host address space");
            }
            cap_bytes_ =
                static_cast<size_t>(
                    cap_megabytes * kBytesPerMegabyte);
        } else {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0) {
                cap_bytes_ = static_cast<size_t>(static_cast<double>(total_b) * 0.6);
            }
            (void)cudaGetLastError();
        }
    }

    void bind_selected_device() const {
        int selected = -1;
        std::string selection_error;
        if (!gpu::select_preferred_device(
                &selected, &selection_error) ||
            selected != device_id_) {
            throw std::runtime_error(
                "GPU memory pool device binding failed: " +
                selection_error);
        }
    }

    // Return all currently-cached (free) blocks to the driver.  Live blocks
    // still owned by a Tensor are untouched.
    void trim_locked() {
        for (auto allocation = live_.begin();
             allocation != live_.end();) {
            auto current = allocation++;
            if (current->second.retained_release &&
                current->second.retryable_release) {
                current->second.retained_release = false;
                current->second.retryable_release = false;
                (void)release_owned_locked(current, false);
            }
        }
        for (auto& kv : free_) {
            for (const CachedBlock& block : kv.second) {
                auto allocation = live_.find(block.pointer);
                if (allocation == live_.end()) {
                    ++unknown_deallocation_attempts_;
                    const size_t bytes = kv.first & ~size_t{1};
                    cached_bytes_ -=
                        cached_bytes_ >= bytes
                            ? bytes
                            : cached_bytes_;
                    continue;
                }
                (void)release_owned_locked(allocation, true);
            }
            kv.second.clear();
        }
    }

    void* raw_alloc(size_t bytes, bool managed_memory) {
        void* raw = nullptr;
        const cudaError_t allocation_status = managed_memory
                                                  ? cudaMallocManaged(&raw, bytes)
                                                  : cudaMalloc(&raw, bytes);
        if (allocation_status != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        if (!managed_memory) return raw;
        // Pascal+Windows hardening, applied ONCE per physical block (it then
        // persists across every pooled reuse): hint the driver that this UM
        // block is accessed by host and device so pages stay migratable rather
        // than faulting on host access (sm_61 + Windows lacks demand paging).
        // NSOS_NO_MEMADVISE opts out: some environments (notably running under
        // compute-sanitizer) reject cudaMemAdvise with "invalid device
        // ordinal", flooding the log — the advise is a perf hint, never a
        // correctness requirement, so skipping it is safe.
        static const bool skip_advise = std::getenv("NSOS_NO_MEMADVISE") != nullptr;
        int device_id = 0;
        const cudaError_t device_status =
            skip_advise ? cudaSuccess : cudaGetDevice(&device_id);
        if (!skip_advise && device_status == cudaSuccess) {
            cudaError_t device_advice_status = cudaSuccess;
            cudaError_t host_advice_status = cudaSuccess;
#if CUDART_VERSION >= 13000
            cudaMemLocation loc_dev;
            loc_dev.type = cudaMemLocationTypeDevice;
            loc_dev.id = device_id;
            cudaMemLocation loc_host;
            loc_host.type = cudaMemLocationTypeHost;
            loc_host.id = 0;
            device_advice_status =
                cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, loc_dev);
            host_advice_status =
                cudaMemAdvise(raw, bytes, cudaMemAdviseSetAccessedBy, loc_host);
#else
            device_advice_status = cudaMemAdvise(
                raw, bytes, cudaMemAdviseSetAccessedBy, device_id);
            host_advice_status = cudaMemAdvise(
                raw, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
#endif
            if (device_advice_status != cudaSuccess) {
                ++managed_advice_failures_;
            }
            if (host_advice_status != cudaSuccess) {
                ++managed_advice_failures_;
            }
            if ((device_advice_status != cudaSuccess ||
                 host_advice_status != cudaSuccess) &&
                !um_advice_warned_) {
                std::fprintf(
                    stderr,
                    "[pool] WARN: managed-memory advice rejected; "
                    "continuing without the optional residency hint\n");
                um_advice_warned_ = true;
            }
        } else if (!skip_advise) {
            ++managed_advice_failures_;
            if (!um_advice_warned_) {
                std::fprintf(
                    stderr,
                    "[pool] WARN: active device query failed before "
                    "managed-memory advice\n");
                um_advice_warned_ = true;
            }
        }
        (void)cudaGetLastError();
        return raw;
    }

    bool enabled_ = true;
    MemoryPolicy memory_policy_ = MemoryPolicy::Automatic;
    size_t cached_bytes_ = 0;  // current sum of free-list block sizes
    size_t live_bytes_ = 0;    // soma dos blocos atualmente entregues a Tensors
    size_t peak_live_bytes_ = 0;
    size_t peak_reserved_bytes_ = 0;
    unsigned dealloc_probe_ = 0;        // cadência da guarda de pressão UM
    uint64_t managed_pressure_probes_ = 0;
    uint64_t managed_advice_failures_ = 0;
    uint64_t cross_stream_domain_frees_ = 0;
    uint64_t release_failures_ = 0;
    uint64_t unknown_deallocation_attempts_ = 0;
    uint64_t capture_contract_violations_ = 0;
    bool um_pressure_warned_ = false;   // 1 aviso por episódio de pressão
    bool um_advice_warned_ = false;
    size_t cap_bytes_ = 0;     // max cached bytes (0 = unlimited)
    int device_id_ = -1;
    std::mutex mtx_;
    std::unordered_map<size_t, std::vector<CachedBlock>> free_;
    std::unordered_map<void*, Allocation> live_;
    bool capturing_ = false;                 // inside a CUDA-graph capture
    std::unordered_set<void*> captured_;     // alive buffers touched by the capture
    size_t quarantined_blocks_ = 0;
};

}  // namespace
#endif  // USE_CUDA

void TensorDeleter::operator()(float* ptr) noexcept {
    if (!ptr) return;
    if (device == Device::GPU) {
#ifdef USE_CUDA
        if (pool_owned) {
            ManagedPool::instance().deallocate(ptr);
        } else {
            int owner = gpu_device;
            if (owner < 0) {
                try {
                    if (!gpu::select_preferred_device(
                            &owner, nullptr)) {
                        std::fprintf(
                            stderr,
                            "[nsos][ERROR] external GPU tensor release could "
                            "not select an owning device; allocation retained\n");
                        return;
                    }
                } catch (...) {
                    std::fprintf(
                        stderr,
                        "[nsos][ERROR] external GPU tensor release failed "
                        "while selecting its owning device; allocation retained\n");
                    return;
                }
            }
            const cudaError_t select_status = cudaSetDevice(owner);
            if (select_status != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "[nsos][ERROR] external GPU tensor release could not "
                    "select device %d: %s; allocation retained\n",
                    owner, cudaGetErrorString(select_status));
                (void)cudaGetLastError();
                return;
            }
            const cudaError_t status = cudaFree(ptr);
            if (status != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "[nsos][ERROR] external GPU tensor release failed on "
                    "device %d: %s; allocation retained\n",
                    owner, cudaGetErrorString(status));
                (void)cudaGetLastError();
            }
        }
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
    if (strict_gpu_execution()) {
        throw std::runtime_error(
            std::string("Strict GPU execution forbids host fallback in '") +
            op + "'");
    }
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

static std::atomic<bool>& strict_gpu_execution_storage() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

void set_strict_gpu_execution(bool enabled) {
    RuntimeExecutionPolicyMutationGuard mutation;
    strict_gpu_execution_storage().store(enabled, std::memory_order_relaxed);
}

bool strict_gpu_execution() {
    return strict_gpu_execution_storage().load(std::memory_order_relaxed);
}

LowpWeightCacheStats lowp_weight_cache_stats() {
    LowpWeightCacheStats snapshot;
#ifdef USE_CUDA
    auto& counters = lowp_weight_cache_counters();
    snapshot.hits = counters.hits.load(std::memory_order_relaxed);
    snapshot.misses = counters.misses.load(std::memory_order_relaxed);
    snapshot.version_refreshes =
        counters.version_refreshes.load(std::memory_order_relaxed);
    snapshot.evictions = counters.evictions.load(std::memory_order_relaxed);
    snapshot.budget_bypasses =
        counters.budget_bypasses.load(std::memory_order_relaxed);
    snapshot.allocation_failures =
        counters.allocation_failures.load(std::memory_order_relaxed);
    snapshot.resident_bytes =
        counters.resident_bytes.load(std::memory_order_relaxed);
    snapshot.peak_resident_bytes =
        counters.peak_resident_bytes.load(std::memory_order_relaxed);
    snapshot.budget_bytes = configured_lowp_weight_cache_budget_bytes();
#endif
    return snapshot;
}

void reset_lowp_weight_cache_stats() {
#ifdef USE_CUDA
    auto& counters = lowp_weight_cache_counters();
    counters.hits.store(0, std::memory_order_relaxed);
    counters.misses.store(0, std::memory_order_relaxed);
    counters.version_refreshes.store(0, std::memory_order_relaxed);
    counters.evictions.store(0, std::memory_order_relaxed);
    counters.budget_bypasses.store(0, std::memory_order_relaxed);
    counters.allocation_failures.store(0, std::memory_order_relaxed);
    counters.peak_resident_bytes.store(
        counters.resident_bytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
#endif
}

size_t lowp_weight_cache_budget_bytes() {
#ifdef USE_CUDA
    return configured_lowp_weight_cache_budget_bytes();
#else
    return 0;
#endif
}


Tensor::Tensor()
    : size(0), device(Device::CPU), host_accessible_storage_(true) {
    shape = TensorShape(std::vector<int>{});
    data_ptr = nullptr;
}

Tensor::Tensor(std::vector<int> s, Device dev, float fill_value)
    : device(dev), host_accessible_storage_(dev == Device::CPU) {
    shape = TensorShape(s);
    size = checked_tensor_size(shape);
    if (size == 0) {
        data_ptr = nullptr;
        return;
    }

    float* raw_ptr = nullptr;
    if (device == Device::GPU) {
#ifdef USE_CUDA
        // Strict GPU models use native device memory; explicitly fallback-
        // compatible callers may select managed memory. Both share the same
        // size-classed caching allocator and stable-address graph semantics.
        ManagedPool& pool = ManagedPool::instance();
        host_accessible_storage_ = pool.use_managed_memory();
        raw_ptr = static_cast<float*>(
            pool.allocate(static_cast<size_t>(size) * sizeof(float),
                          host_accessible_storage_));
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
    // Install ownership before initialization so every CUDA initialization
    // failure below releases the allocation during stack unwinding.
    data_ptr =
        std::shared_ptr<float>(raw_ptr, TensorDeleter(device));

    if (tensor_skip_fill_flag()) {
        // Tensor::uninitialized: produtor garante sobrescrita total; pular o
        // memset economiza um kernel por alocação no hot path de treino.
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
            const cudaError_t status =
                cudaMemsetAsync(raw_ptr, 0, size * sizeof(float), nsos::gpu::current_stream());
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA tensor memset failed: ") +
                    cudaGetErrorString(status));
            }
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
            const cudaError_t status =
                cudaMemcpy(raw_ptr,
                           host_values.data(),
                           static_cast<size_t>(size) * sizeof(float),
                           cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA tensor fill copy failed: ") +
                    cudaGetErrorString(status));
            }
            record_gpu_transfer(
                Device::GPU, Device::CPU,
                static_cast<size_t>(size) * sizeof(float));
#endif
        } else {
            std::fill_n(raw_ptr, size, fill_value);
        }
    }
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
    // Static coverage audit: every current caller is a whole-buffer producer
    // (GEMM beta=0, memcpy, a full CUDA kernel, or an exhaustive CPU loop).
    // The earlier safety default had remained enabled after the incomplete
    // producers were corrected and inserted a redundant memset in hot paths.
    static const bool uninit_enabled = [] {
        const char* e = std::getenv("NSOS_UNINIT");
        // All callers are whole-buffer producers. Keep =0 as an explicit
        // sanitizer/parity bisection arm without paying an unconditional
        // pre-write memset in production.
        return e == nullptr || e[0] != '0';
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
        std::string selection_error;
        if (!gpu::select_preferred_device(
                nullptr, &selection_error)) {
            throw std::runtime_error(
                "GPU tensor copy cannot bind the selected device: " +
                selection_error);
        }
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
            const cudaError_t launch =
                cudaMemcpyAsync(
                    dst, src, bytes, cudaMemcpyDefault, nsos::gpu::current_stream());
            if (launch != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA async D2D memcpy failed: ") +
                                         cudaGetErrorString(launch));
            }
            record_gpu_transfer(dst_device, src_device, bytes);
            return;
        }
        TensorCopyStream& copy_state = tensor_copy_stream();
        std::lock_guard<std::mutex> copy_lock(copy_state.mutex);
        cudaStream_t stream = copy_state.stream;
        if (stream) {
            // A blocking stream has no implicit ordering relationship with a
            // per-thread default stream. Fence the producer explicitly before
            // the shared copy stream reads device storage.
            cudaError_t fence_status =
                cudaEventRecord(
                    copy_state.producer_ready,
                    gpu::current_stream());
            if (fence_status == cudaSuccess) {
                fence_status =
                    cudaStreamWaitEvent(
                        stream, copy_state.producer_ready, 0);
            }
            if (fence_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("GPU copy-stream producer fence failed: ") +
                    cudaGetErrorString(fence_status));
            }
            // Async copy on the dedicated stream, then wait on JUST this stream.
            // Equivalent ordering to the legacy blocking cudaMemcpy (blocking
            // stream serializes with stream 0), but does not stall unrelated
            // device work the way cudaDeviceSynchronize would.
            const cudaError_t launch =
                cudaMemcpyAsync(
                    dst, src, bytes, cudaMemcpyDefault, stream);
            if (launch != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA async memcpy failed: ") +
                                         cudaGetErrorString(launch));
            }
            const cudaError_t sync = cudaStreamSynchronize(stream);
            if (sync != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA copy-stream sync failed: ") +
                                         cudaGetErrorString(sync));
            }
            record_gpu_transfer(dst_device, src_device, bytes);
            record_gpu_stream_synchronization();
        } else {
            const cudaError_t status =
                cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memcpy failed: ") +
                                         cudaGetErrorString(status));
            }
            record_gpu_transfer(dst_device, src_device, bytes);
        }
        return;
    }
    std::memcpy(dst, src, bytes);
    record_gpu_transfer(dst_device, src_device, bytes);

#else
    (void)dst_device; (void)src_device; (void)async_d2d;
    if (bytes > 0) {
        std::memcpy(dst, src, bytes);
        record_gpu_transfer(dst_device, src_device, bytes);
    }
#endif
}

GpuTransferStats gpu_transfer_stats() {
    const auto& counters = gpu_transfer_counters();
    GpuTransferStats snapshot;
    snapshot.h2d_calls =
        counters.h2d_calls.load(std::memory_order_relaxed);
    snapshot.h2d_bytes =
        counters.h2d_bytes.load(std::memory_order_relaxed);
    snapshot.d2h_calls =
        counters.d2h_calls.load(std::memory_order_relaxed);
    snapshot.d2h_bytes =
        counters.d2h_bytes.load(std::memory_order_relaxed);
    snapshot.d2d_calls =
        counters.d2d_calls.load(std::memory_order_relaxed);
    snapshot.d2d_bytes =
        counters.d2d_bytes.load(std::memory_order_relaxed);
    snapshot.h2h_calls =
        counters.h2h_calls.load(std::memory_order_relaxed);
    snapshot.h2h_bytes =
        counters.h2h_bytes.load(std::memory_order_relaxed);
    snapshot.device_synchronizations =
        counters.device_synchronizations.load(std::memory_order_relaxed);
    snapshot.stream_synchronizations =
        counters.stream_synchronizations.load(std::memory_order_relaxed);
    return snapshot;
}

void reset_gpu_transfer_stats() {
    auto& counters = gpu_transfer_counters();
    counters.h2d_calls.store(0, std::memory_order_relaxed);
    counters.h2d_bytes.store(0, std::memory_order_relaxed);
    counters.d2h_calls.store(0, std::memory_order_relaxed);
    counters.d2h_bytes.store(0, std::memory_order_relaxed);
    counters.d2d_calls.store(0, std::memory_order_relaxed);
    counters.d2d_bytes.store(0, std::memory_order_relaxed);
    counters.h2h_calls.store(0, std::memory_order_relaxed);
    counters.h2h_bytes.store(0, std::memory_order_relaxed);
    counters.device_synchronizations.store(0, std::memory_order_relaxed);
    counters.stream_synchronizations.store(0, std::memory_order_relaxed);
}

void record_gpu_transfer(Device dst_device, Device src_device, size_t bytes) {
    if (bytes == 0) return;
    auto& counters = gpu_transfer_counters();
    if (dst_device == Device::GPU && src_device == Device::CPU) {
        counters.h2d_calls.fetch_add(1, std::memory_order_relaxed);
        counters.h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else if (dst_device == Device::CPU && src_device == Device::GPU) {
        counters.d2h_calls.fetch_add(1, std::memory_order_relaxed);
        counters.d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else if (dst_device == Device::GPU &&
               src_device == Device::GPU) {
        counters.d2d_calls.fetch_add(1, std::memory_order_relaxed);
        counters.d2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        counters.h2h_calls.fetch_add(1, std::memory_order_relaxed);
        counters.h2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void record_gpu_device_synchronization() {
    gpu_transfer_counters().device_synchronizations.fetch_add(
        1, std::memory_order_relaxed);
}

void record_gpu_stream_synchronization() {
    gpu_transfer_counters().stream_synchronizations.fetch_add(
        1, std::memory_order_relaxed);
}

#ifdef USE_CUDA
Tensor matmul_nt_mixed_gpu(const Tensor& a,
                           const Tensor& b_rowmajor,
                           const uint64_t* content_version);
#endif

bool matmul_nt_mixed_materialization_enabled() {
    static const bool enabled = [] {
        const char* value =
            std::getenv("NSOS_MATMUL_NT_MIXED_MATERIALIZE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool matmul_tn_materialization_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NSOS_MATMUL_TN_MATERIALIZE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

Tensor matmul_nt_impl(const Tensor& a, const Tensor& b_rowmajor,
                      const uint64_t* content_version) {
    // Contrato estrito: b é uma matriz de pesos rank-2 [n, k] (o caso BitLinear/
    // KAN); a é [..., m, k].  Resultado [..., m, n].
    if (b_rowmajor.shape.size() != 2 || a.shape.size() < 2 ||
        a.shape.back() != b_rowmajor.shape[1]) {
        throw std::runtime_error("matmul_nt expects a[..,m,k] and b[n,k]");
    }
    if (a.get_device() != b_rowmajor.get_device()) {
        throw std::invalid_argument(
            "matmul_nt requires tensors on the same device");
    }
#ifdef USE_CUDA
    if (a.get_device() == Device::GPU &&
        b_rowmajor.get_device() == Device::GPU && a.size > 0 &&
        b_rowmajor.size > 0 && gpu_blas_supported() &&
        matmul_precision_mode() != 0 &&
        !matmul_nt_mixed_materialization_enabled()) {
        return matmul_nt_mixed_gpu(a, b_rowmajor, content_version);
    }
    // FP32 GPU uses native OP_T below. Mixed GPU dispatches above to GemmEx
    // with the same transpose flag; only the CPU reference materializes B^T.
    if (a.get_device() == Device::GPU && b_rowmajor.get_device() == Device::GPU &&
        a.size > 0 && b_rowmajor.size > 0 && gpu_blas_supported() &&
        matmul_precision_mode() == 0) {
        const int rank_a = static_cast<int>(a.shape.size());
        const int m = a.shape[rank_a - 2];
        const int k = a.shape[rank_a - 1];
        const int n = b_rowmajor.shape[0];
        std::vector<int> out_dims = a.shape.dims;
        out_dims.back() = n;
        // GEMM beta=0 sobrescreve 100% de C.
        Tensor result = Tensor::uninitialized(out_dims, Device::GPU);
        const int batch = a.size / (m * k);
        // Row-major C[m,n] = A[m,k] · B[n,k]ᵀ.  Em termos column-major do
        // cuBLAS (mesma memória): C_cm[n,m] = OP_T(B_mem, ld=k)[n,k] ·
        // OP_N(A_mem, ld=k)[k,m].  B é compartilhado entre batches (stride 0).
        cublasHandle_t handle = cublas_handle();
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublas_check(
            cublasSgemmStridedBatched(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      n,
                                      m,
                                      k,
                                      &alpha,
                                      b_rowmajor.raw_data(),
                                      k,
                                      0LL,
                                      a.raw_data(),
                                      k,
                                      static_cast<long long>(m) * k,
                                      &beta,
                                      result.raw_data(),
                                      n,
                                      static_cast<long long>(m) * n,
                                      batch),
            "cublasSgemmStridedBatched(NT)");
        sync_cuda();
        return result;
    }
#endif
    return a.matmul(b_rowmajor.transpose());
}

Tensor matmul_nt(const Tensor& a, const Tensor& b_rowmajor) {
    return matmul_nt_impl(a, b_rowmajor, nullptr);
}

Tensor matmul_nt_cached_weight(const Tensor& a,
                               const Tensor& b_rowmajor,
                               uint64_t content_version) {
    if (content_version == 0) {
        throw std::invalid_argument(
            "matmul_nt_cached_weight requires a non-zero content version");
    }
    return matmul_nt_impl(a, b_rowmajor, &content_version);
}

float tensor_abs_mean(const Tensor& t) {
    if (t.size <= 0) {
        return 0.0f;
    }
#ifdef USE_CUDA
    if (t.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
        float* d_sum = scalar_reduction_scratch();
        if (d_sum == nullptr) {
            throw std::runtime_error(
                "tensor_abs_mean could not allocate GPU reduction scratch");
        }
        const cudaError_t clear_status =
            cudaMemsetAsync(d_sum, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("tensor_abs_mean GPU clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
        launch_abs_sum_kernel(d_sum, t.raw_data(), t.size);
        sync_cuda();
        const float total = copy_scalar_from_device(d_sum);
        return total / static_cast<float>(t.size);
    }
#endif
    const float* src = t.data();
    double total = 0.0;
    for (int i = 0; i < t.size; ++i) {
        total += std::fabs(static_cast<double>(src[i]));
    }
    return static_cast<float>(total / static_cast<double>(t.size));
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

void gpu_pool_begin_capture() {
#ifdef USE_CUDA
    ManagedPool::instance().begin_capture();
#endif
}

void gpu_pool_end_capture() {
#ifdef USE_CUDA
    ManagedPool::instance().end_capture();
#endif
}

void gpu_pool_release_capture() noexcept {
#ifdef USE_CUDA
    ManagedPool::instance().release_capture();
#endif
}

#ifdef NSOS_ENABLE_TEST_HOOKS
void set_gpu_pool_release_failure_countdown(int countdown) {
#ifdef USE_CUDA
    pool_release_failure_countdown().store(
        countdown < 0 ? -1 : countdown,
        std::memory_order_relaxed);
#else
    (void)countdown;
#endif
}
#endif

Tensor Tensor::random(const std::vector<int>& s, Device dev) {
    Tensor t(s, Device::CPU);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return dev == Device::CPU ? t : t.to(dev);
}

Tensor Tensor::uniform(const std::vector<int>& s, float low, float high,
                       Device dev) {
    if (!std::isfinite(low) || !std::isfinite(high) || low > high) {
        throw std::invalid_argument("Tensor::uniform invalid bounds");
    }
    Tensor host(s, Device::CPU);
    std::uniform_real_distribution<float> dist(low, high);
    float* dst = host.data();
    for (int64_t i = 0; i < host.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return dev == Device::CPU ? host : host.to(dev);
}

Tensor Tensor::kaiming_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, Device::CPU);
    float fan_in = s.empty() ? 1.0f : static_cast<float>(s.back());
    float bound = std::sqrt(6.0f / std::max(fan_in, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return dev == Device::CPU ? t : t.to(dev);
}

Tensor Tensor::kaiming_uniform(const std::vector<int>& s, Device dev,
                               uint64_t seed) {
    if (seed == 0) {
        // Preserve historical behavior for seed=0 (un-seeded path).
        return kaiming_uniform(s, dev);
    }
    Tensor t(s, Device::CPU);
    float fan_in = s.empty() ? 1.0f : static_cast<float>(s.back());
    float bound = std::sqrt(6.0f / std::max(fan_in, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::mt19937 local_rng(static_cast<uint32_t>(seed));
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(local_rng);
    }
    return dev == Device::CPU ? t : t.to(dev);
}

Tensor Tensor::xavier_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, Device::CPU);
    // Pesos são [out, in] row-major: fan_out = s.front(), fan_in = s.back().
    // (Os rótulos estavam trocados; sem efeito numérico — a fórmula de Xavier
    // usa fan_in + fan_out simetricamente — corrigido por clareza.)
    float fan_out = s.empty() ? 1.0f : static_cast<float>(s.front());
    float fan_in = s.size() > 1 ? static_cast<float>(s.back()) : fan_out;
    float bound = std::sqrt(6.0f / std::max(fan_in + fan_out, 1.0f));
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* dst = t.data();
    for (int i = 0; i < t.size; ++i) {
        dst[i] = dist(tensor_rng());
    }
    return dev == Device::CPU ? t : t.to(dev);
}

Tensor Tensor::eye(int n, Device dev) {
    Tensor t = zeros({n, n}, Device::CPU);
    float* dst = t.data();
    for (int i = 0; i < n; ++i) {
        dst[i * n + i] = 1.0f;
    }
    return dev == Device::CPU ? t : t.to(dev);
}

Tensor Tensor::from_scalar(float val, Device dev) {
    Tensor t({1}, Device::CPU);
    t.data()[0] = val;
    return dev == Device::CPU ? t : t.to(dev);
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
#ifdef USE_CUDA
        if (p.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
            launch_scale_inplace_kernel(p.raw_data(), scale, p.size);
            sync_cuda();
            continue;
        }
#endif
        float* dst = p.data();
        for (int i = 0; i < p.size; ++i) {
            dst[i] *= scale;
        }
    }
}

Tensor Tensor::add(const Tensor& other) const {
    if (device != other.device) {
        throw std::invalid_argument(
            "Tensor::add requires tensors on the same device");
    }
    TensorShape out_shape;
    if (!TensorIterator::compute_broadcast_shape(
            shape, other.shape, out_shape)) {
        throw std::invalid_argument(
            "Tensor::add received non-broadcastable shapes");
    }
    Tensor result = Tensor::uninitialized(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported()) {
        if (shape == other.shape) {
            launch_add_kernel(result.raw_data(), raw_data(), other.raw_data(),
                              size);
            sync_cuda();
            return result;
        }
        if (size == result.size && other.shape.size() == 1 &&
            other.size > 0 &&
            (other.size == 1 || shape.back() == other.size)) {
            launch_add_broadcast_kernel(result.raw_data(), raw_data(),
                                         other.raw_data(), size, other.size);
            sync_cuda();
            return result;
        }
        if (other.size == result.size && shape.size() == 1 &&
            size > 0 &&
            (size == 1 || other.shape.back() == size)) {
            launch_add_broadcast_kernel(result.raw_data(), other.raw_data(),
                                         raw_data(), other.size, size);
            sync_cuda();
            return result;
        }
        if (size == result.size && shape.size() == other.shape.size() &&
            other.shape.back() == 1 &&
            other.size == size / shape.back()) {
            if (!launch_add_trailing_broadcast_kernel(
                result.raw_data(), raw_data(), other.raw_data(),
                size, shape.back())) {
                throw std::runtime_error(
                    "Tensor::add trailing GPU broadcast rejected invalid "
                    "arguments");
            }
            sync_cuda();
            return result;
        }
        if (other.size == result.size &&
            shape.size() == other.shape.size() && shape.back() == 1 &&
            size == other.size / other.shape.back()) {
            if (!launch_add_trailing_broadcast_kernel(
                result.raw_data(), other.raw_data(), raw_data(),
                other.size, other.shape.back())) {
                throw std::runtime_error(
                    "Tensor::add reverse trailing GPU broadcast rejected "
                    "invalid arguments");
            }
            sync_cuda();
            return result;
        }
    }
#endif
    warn_host_fallback_once("add(broadcast nao-padrao)", device);
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a + b; });
    return result;
}

void Tensor::add_inplace_(const Tensor& other) {
    if (shape != other.shape || size != other.size) {
        throw std::invalid_argument(
            "Tensor::add_inplace_ requires identical shapes");
    }
    if (device != other.device) {
        throw std::invalid_argument(
            "Tensor::add_inplace_ requires identical devices");
    }
    if (size == 0) {
        return;
    }
#ifdef USE_CUDA
    if (device == Device::GPU) {
        if (!gpu_custom_kernels_supported()) {
            throw std::runtime_error(
                "Tensor::add_inplace_ requires supported GPU kernels for a "
                "GPU tensor");
        }
        launch_add_kernel(raw_data(), raw_data(), other.raw_data(), size);
        sync_cuda();
        return;
    }
#endif
    float* dst = data();
    const float* src = other.data();
    for (int i = 0; i < size; ++i) {
        dst[i] += src[i];
    }
}

Tensor Tensor::sub(const Tensor& other) const {
    if (device != other.device) {
        throw std::invalid_argument(
            "Tensor::sub requires tensors on the same device");
    }
    TensorShape out_shape;
    if (!TensorIterator::compute_broadcast_shape(
            shape, other.shape, out_shape)) {
        throw std::invalid_argument(
            "Tensor::sub received non-broadcastable shapes");
    }
    Tensor result = Tensor::uninitialized(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported() &&
        size > 0 && other.size > 0) {
        if (shape == other.shape) {
            launch_sub_kernel(result.raw_data(), raw_data(), other.raw_data(),
                              size);
            sync_cuda();
            return result;
        }
        if (size == result.size && other.shape.size() == 1 &&
            (other.size == 1 || shape.back() == other.size)) {
            if (!launch_sub_broadcast_kernel(
                result.raw_data(), raw_data(), other.raw_data(),
                size, other.size, false)) {
                throw std::runtime_error(
                    "Tensor::sub GPU broadcast rejected invalid arguments");
            }
            sync_cuda();
            return result;
        }
        if (other.size == result.size && shape.size() == 1 &&
            (size == 1 || other.shape.back() == size)) {
            if (!launch_sub_broadcast_kernel(
                result.raw_data(), other.raw_data(), raw_data(),
                other.size, size, true)) {
                throw std::runtime_error(
                    "Tensor::sub reverse GPU broadcast rejected invalid "
                    "arguments");
            }
            sync_cuda();
            return result;
        }
        if (size == result.size && shape.size() == other.shape.size() &&
            other.shape.back() == 1 &&
            other.size == size / shape.back()) {
            if (!launch_sub_trailing_broadcast_kernel(
                result.raw_data(), raw_data(), other.raw_data(),
                size, shape.back(), false)) {
                throw std::runtime_error(
                    "Tensor::sub trailing GPU broadcast rejected invalid "
                    "arguments");
            }
            sync_cuda();
            return result;
        }
        if (other.size == result.size &&
            shape.size() == other.shape.size() && shape.back() == 1 &&
            size == other.size / other.shape.back()) {
            if (!launch_sub_trailing_broadcast_kernel(
                result.raw_data(), other.raw_data(), raw_data(),
                other.size, other.shape.back(), true)) {
                throw std::runtime_error(
                    "Tensor::sub reverse trailing GPU broadcast rejected "
                    "invalid arguments");
            }
            sync_cuda();
            return result;
        }
    }
#endif
    warn_host_fallback_once("sub(broadcast nao-padrao)", device);
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a - b; });
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    if (device != other.device) {
        throw std::invalid_argument(
            "Tensor::mul requires tensors on the same device");
    }
    TensorShape out_shape;
    if (!TensorIterator::compute_broadcast_shape(
            shape, other.shape, out_shape)) {
        throw std::invalid_argument(
            "Tensor::mul received non-broadcastable shapes");
    }
    Tensor result = Tensor::uninitialized(out_shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_custom_kernels_supported()) {
        if (shape == other.shape) {
            launch_mul_tensor_kernel(result.raw_data(), raw_data(),
                                     other.raw_data(), size);
            sync_cuda();
            return result;
        }
        if (size == result.size && other.shape.size() == 1 &&
            other.size > 0 &&
            (other.size == 1 || shape.back() == other.size)) {
            const int cols = other.size;
            const int rows = size / std::max(cols, 1);
            launch_mul_vector_broadcast_kernel(result.raw_data(), raw_data(),
                                                other.raw_data(), rows, cols);
            sync_cuda();
            return result;
        }
        if (other.size == result.size && shape.size() == 1 &&
            size > 0 &&
            (size == 1 || other.shape.back() == size)) {
            const int cols = size;
            const int rows = other.size / cols;
            launch_mul_vector_broadcast_kernel(
                result.raw_data(), other.raw_data(), raw_data(), rows, cols);
            sync_cuda();
            return result;
        }
        if (size == result.size && shape.size() == other.shape.size() &&
            other.shape.back() == 1 &&
            other.size == size / shape.back()) {
            launch_mul_broadcast_kernel(
                result.raw_data(), raw_data(), other.raw_data(),
                size, shape.back());
            sync_cuda();
            return result;
        }
        if (other.size == result.size &&
            shape.size() == other.shape.size() && shape.back() == 1 &&
            size == other.size / other.shape.back()) {
            launch_mul_broadcast_kernel(
                result.raw_data(), other.raw_data(), raw_data(),
                other.size, other.shape.back());
            sync_cuda();
            return result;
        }
    }
#endif
    warn_host_fallback_once("mul(broadcast nao-padrao)", device);
    TensorIterator iter(result, *this, other);
    iter.parallel_for_each([](float a, float b) { return a * b; });
    return result;
}

Tensor Tensor::mul(float scalar) const {
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
        if (v == "fp32" || v == "FP32") return 0;
        throw std::invalid_argument(
            "NSOS_MIXED_PRECISION must be fp32, fp16, or bf16");
    }()};
    return mode;
}

void set_matmul_precision_mode(int mode) {
    if (mode < 0 || mode > 2) {
        throw std::invalid_argument(
            "matmul precision mode must be 0 (FP32), 1 (BF16), or 2 (FP16)");
    }
    RuntimeExecutionPolicyMutationGuard mutation;
    const int current =
        matmul_precision_mode_storage().load(std::memory_order_relaxed);
    if (current != mode) {
        auto& epoch = matmul_precision_policy_epoch_storage();
        if (epoch.load(std::memory_order_relaxed) ==
            std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(
                "matmul precision policy epoch is exhausted");
        }
        matmul_precision_mode_storage().store(mode, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_relaxed);
    }
}

int matmul_precision_mode() {
    return matmul_precision_mode_storage().load(std::memory_order_relaxed);
}

#ifdef USE_CUDA
void validate_matmul_precision_for_active_device(int mode) {
    if (mode == 0) return;
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        throw std::runtime_error(
            std::string("Cannot resolve active ") +
            NSOS_GPU_BACKEND_NAME +
            " device for mixed precision");
    }
#if defined(NSOS_GPU_BACKEND_HIP)
    const std::vector<gpu::DeviceInfo> devices =
        gpu::enumerate_devices();
    const auto active = std::find_if(
        devices.begin(), devices.end(),
        [device_id](const gpu::DeviceInfo& info) {
            return info.index == device_id;
        });
    if (active == devices.end()) {
        throw std::runtime_error(
            "Cannot query active HIP device capabilities for mixed "
            "precision");
    }
    if (mode == 1 && !active->bf16) {
        throw std::runtime_error(
            "BF16 GEMM is not validated for active AMD architecture " +
            active->architecture + "; select FP16 or FP32");
    }
    if (mode == 2 && !active->fp16) {
        throw std::runtime_error(
            "FP16 GEMM is not validated for active AMD architecture " +
            active->architecture + "; select FP32");
    }
#else
    thread_local int cached_device = -1;
    thread_local int cached_major = -1;
    thread_local int cached_minor = -1;
    if (cached_device != device_id) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device_id) != cudaSuccess) {
            throw std::runtime_error(
                "Cannot query CUDA compute capability for mixed precision");
        }
        cached_device = device_id;
        cached_major = properties.major;
        cached_minor = properties.minor;
    }
    if (mode == 1 && cached_major < 8) {
        throw std::runtime_error(
            "BF16 Tensor-Core GEMM requires sm_80 or newer; active device is "
            "sm_" + std::to_string(cached_major) +
            std::to_string(cached_minor) +
            ". Select FP16 on Turing/T4 (sm_75) or FP32.");
    }
    if (mode == 2 && cached_major < 7) {
        throw std::runtime_error(
            "FP16 Tensor-Core GEMM requires sm_70 or newer");
    }
#endif
}
#endif

#ifdef USE_CUDA
Tensor matmul_nt_mixed_gpu(const Tensor& a,
                           const Tensor& b_rowmajor,
                           const uint64_t* content_version) {
    const int rank = static_cast<int>(a.shape.size());
    const int m = a.shape[rank - 2];
    const int k = a.shape[rank - 1];
    const int n = b_rowmajor.shape[0];
    const int64_t matrix_elements =
        static_cast<int64_t>(m) * k;
    if (m <= 0 || k <= 0 || n <= 0 || matrix_elements <= 0 ||
        a.size % matrix_elements != 0 ||
        a.size / matrix_elements >
            std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "matmul_nt mixed shape exceeds the supported range");
    }
    const int batch = static_cast<int>(a.size / matrix_elements);
    std::vector<int> output_shape = a.shape.dims;
    output_shape.back() = n;
    Tensor result =
        Tensor::uninitialized(output_shape, Device::GPU);

    const int mixed_mode = matmul_precision_mode();
    validate_matmul_precision_for_active_device(mixed_mode);
    const cudaDataType_t input_type =
        mixed_mode == 1 ? CUDA_R_16BF : CUDA_R_16F;
    const size_t a_elements = static_cast<size_t>(a.size);
    const size_t b_elements =
        static_cast<size_t>(b_rowmajor.size);
    GemmLowpWorkspace& workspace = gemm_lowp_workspace();
    if (!workspace.ensure_a(a_elements * 2u)) {
        throw std::runtime_error(
            "matmul_nt mixed-precision workspace allocation failed");
    }
    void* a_low = workspace.a_pointer();
    launch_cast_f32_to_lowp_kernel(
        a_low, a.raw_data(), a_elements, mixed_mode);
    void* b_low = nullptr;
    if (content_version != nullptr) {
        b_low = lowp_weight_cache().find_or_convert(
            b_rowmajor, mixed_mode, *content_version);
    }
    if (b_low == nullptr) {
        if (!workspace.ensure_b(b_elements * 2u)) {
            throw std::runtime_error(
                "matmul_nt mixed-precision weight workspace allocation "
                "failed");
        }
        b_low = workspace.b_pointer();
        launch_cast_f32_to_lowp_kernel(
            b_low, b_rowmajor.raw_data(), b_elements, mixed_mode);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmStridedBatchedEx(
            cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, &alpha, b_low, input_type, k, 0LL,
            a_low, input_type, k,
            static_cast<long long>(m) * k, &beta,
            result.raw_data(), CUDA_R_32F, n,
            static_cast<long long>(m) * n, batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx(NT)");
    sync_cuda();
    return result;
}
#endif

Tensor matmul_tn(const Tensor& a_rowmajor, const Tensor& b) {
    if (a_rowmajor.shape.size() != 2 || b.shape.size() != 2 ||
        a_rowmajor.shape[0] != b.shape[0]) {
        throw std::runtime_error(
            "matmul_tn expects a[m,k] and b[m,n]");
    }
    if (a_rowmajor.get_device() != b.get_device()) {
        throw std::invalid_argument(
            "matmul_tn requires tensors on the same device");
    }
    const int m = a_rowmajor.shape[0];
    const int k = a_rowmajor.shape[1];
    const int n = b.shape[1];
    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument(
            "matmul_tn requires non-empty matrix dimensions");
    }
    if (matmul_tn_materialization_enabled()) {
        return a_rowmajor.transpose().matmul(b);
    }
#ifdef USE_CUDA
    if (a_rowmajor.get_device() == Device::GPU &&
        gpu_blas_supported()) {
        Tensor result =
            Tensor::uninitialized({k, n}, Device::GPU);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int mixed_mode = matmul_precision_mode();
        if (mixed_mode != 0) {
            validate_matmul_precision_for_active_device(mixed_mode);
            const cudaDataType_t input_type =
                mixed_mode == 1 ? CUDA_R_16BF : CUDA_R_16F;
            const size_t a_elements =
                static_cast<size_t>(a_rowmajor.size);
            const size_t b_elements = static_cast<size_t>(b.size);
            GemmLowpWorkspace& workspace = gemm_lowp_workspace();
            if (!workspace.ensure(a_elements * 2u, b_elements * 2u)) {
                throw std::runtime_error(
                    "matmul_tn mixed-precision workspace allocation failed");
            }
            void* a_low = workspace.a_pointer();
            void* b_low = workspace.b_pointer();
#if defined(NSOS_GPU_BACKEND_HIP)
            // rocBLAS lowp OP_T can select an invalid RDNA3 code object.
            // Fuse the required transpose with the FP32->lowp conversion so
            // no FP32 transpose, host fallback, or false-stride view exists.
            launch_cast_transpose_f32_to_lowp_kernel(
                a_low, a_rowmajor.raw_data(), m, k, mixed_mode);
#else
            launch_cast_f32_to_lowp_kernel(
                a_low, a_rowmajor.raw_data(), a_elements, mixed_mode);
#endif
            launch_cast_f32_to_lowp_kernel(
                b_low, b.raw_data(), b_elements, mixed_mode);
#if defined(NSOS_GPU_BACKEND_HIP)
            // A_transposed is [k,m] row-major and therefore [m,k]
            // column-major. NN computes C_cm[n,k] = B_cm[n,m] * A_t_cm[m,k].
            cublas_check(
                cublasGemmStridedBatchedEx(
                    cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                    n, k, m, &alpha,
                    b_low, input_type, n, 0LL,
                    a_low, input_type, m, 0LL,
                    &beta, result.raw_data(), CUDA_R_32F, n, 0LL,
                    1, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "hipblasGemmStridedBatchedEx(TN-cast-transpose)");
#else
            cublas_check(
                cublasGemmStridedBatchedEx(
                    cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                    n, k, m, &alpha,
                    b_low, input_type, n, 0LL,
                    a_low, input_type, k, 0LL,
                    &beta, result.raw_data(), CUDA_R_32F, n, 0LL,
                    1, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmStridedBatchedEx(TN)");
#endif
            sync_cuda();
            return result;
        }
        cublas_check(
            cublasSgemm(
                cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                n, k, m, &alpha, b.raw_data(), n,
                a_rowmajor.raw_data(), k, &beta,
                result.raw_data(), n),
            "cublasSgemm(TN)");
        sync_cuda();
        return result;
    }
#endif
    return a_rowmajor.transpose().matmul(b);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() < 2 || other.shape.size() < 2) {
        throw std::runtime_error("matmul requires rank >= 2 tensors");
    }
    if (device != other.device) {
        throw std::invalid_argument(
            "Tensor::matmul requires tensors on the same device");
    }

    int rank_a = static_cast<int>(shape.size());
    int rank_b = static_cast<int>(other.shape.size());
    int m = shape[rank_a - 2];
    int k = shape[rank_a - 1];
    int k_other = other.shape[rank_b - 2];
    int n = other.shape[rank_b - 1];

    if (m <= 0 || k <= 0 || k_other <= 0 || n <= 0) {
        throw std::invalid_argument(
            "Tensor::matmul requires non-empty matrix dimensions");
    }
    if (k != k_other) {
        throw std::runtime_error("matmul shape mismatch");
    }

    const int batch_rank_a = rank_a - 2;
    const int batch_rank_b = rank_b - 2;
    const int batch_rank = std::max(batch_rank_a, batch_rank_b);
    std::vector<int> out_dims(static_cast<size_t>(batch_rank), 1);
    for (int index = 0; index < batch_rank; ++index) {
        const int a_index = index - (batch_rank - batch_rank_a);
        const int b_index = index - (batch_rank - batch_rank_b);
        const int a_dim =
            a_index >= 0 ? shape[static_cast<size_t>(a_index)] : 1;
        const int b_dim =
            b_index >= 0 ? other.shape[static_cast<size_t>(b_index)] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument(
                "Tensor::matmul received non-broadcastable batch dimensions");
        }
        out_dims[static_cast<size_t>(index)] = std::max(a_dim, b_dim);
    }
    out_dims.push_back(m);
    out_dims.push_back(n);

    const int64_t matrix_a =
        static_cast<int64_t>(m) * static_cast<int64_t>(k);
    const int64_t matrix_b =
        static_cast<int64_t>(k) * static_cast<int64_t>(n);
    if (size % matrix_a != 0 || other.size % matrix_b != 0) {
        throw std::logic_error(
            "Tensor::matmul storage size is inconsistent with its shape");
    }
    const int64_t batch_a_64 = size / matrix_a;
    const int64_t batch_b_64 = other.size / matrix_b;
    int64_t batch_64 = 1;
    for (int index = 0; index < batch_rank; ++index) {
        const int dim = out_dims[static_cast<size_t>(index)];
        if (dim <= 0 ||
            batch_64 > std::numeric_limits<int>::max() / dim) {
            throw std::overflow_error(
                "Tensor::matmul batch element count exceeds the supported "
                "integer range");
        }
        batch_64 *= dim;
    }
    const int batch_a = static_cast<int>(batch_a_64);
    const int batch_b = static_cast<int>(batch_b_64);
    const int batch = static_cast<int>(batch_64);
    // The strided implementation can broadcast a whole matrix batch (stride
    // zero), but not an arbitrary partially-broadcast batch grid.
    if ((batch_a != 1 && batch_a != batch) ||
        (batch_b != 1 && batch_b != batch)) {
        throw std::invalid_argument(
            "Tensor::matmul only supports identical batch grids or a "
            "single-matrix batch broadcast");
    }

#ifdef USE_CUDA
    const bool matmul_gpu_path =
        use_gpu_fast_path(*this, other) && gpu_blas_supported();
#else
    const bool matmul_gpu_path = false;
#endif
    // GPU: GEMM beta=0 sobrescreve 100% de C -> alocação sem zero-fill.
    Tensor result = Tensor::uninitialized(out_dims, device);

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
        // dispatch via cublasGemmEx + Tensor Cores. T4 (sm_75) supports
        // Tensor-Core FP16; native BF16 requires Ampere (sm_80) or newer.
        // When NSOS_MIXED_PRECISION is set to "bf16" or "fp16", the active
        // device capability is checked before inputs are cast and the GEMM
        // writes an FP32-accumulated output.
        //
        // Why BF16 over FP16 on sm_80+:
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
            validate_matmul_precision_for_active_device(mixed_mode);
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
                static_cast<size_t>(batch_a) * static_cast<size_t>(m) *
                static_cast<size_t>(k);
            const size_t total_b_elems =
                static_cast<size_t>(batch_b) * static_cast<size_t>(k) *
                static_cast<size_t>(n);

            // Get cached staging buffers from the workspace (resized
            // on demand, reused across calls).  Replaces the per-
            // matmul cudaMalloc/cudaFree pair that was burning 2-8 s
            // per training step on a 12-layer model.
            GemmLowpWorkspace& ws = gemm_lowp_workspace();
            if (ws.ensure(total_a_elems * bytes_per_lp,
                          total_b_elems * bytes_per_lp)) {
                void* a_low_ptr = ws.a_pointer();
                void* b_low_ptr = ws.b_pointer();

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

                const long long stride_a = (batch_a == 1)
                                                ? 0LL
                                                : static_cast<long long>(m) * k;
                const long long stride_b = (batch_b == 1)
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
            // An explicitly requested precision mode is a compute contract,
            // not a hint.  Falling through here used to execute FP32 after a
            // low-precision workspace OOM while reporting a successful mixed
            // precision step.  Fail closed so telemetry, loss-scaler tests and
            // production policy cannot mistake a degraded run for FP16/BF16.
            throw std::runtime_error(
                "Mixed-precision CUDA workspace allocation failed; refusing "
                "silent FP32 fallback");
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
        //   strideA = 0 for a single lhs matrix, otherwise m*k
        //   strideB = 0 for a single rhs matrix, otherwise k*n
        //   strideC = m*n       (always)
        //
        // The math is IDENTICAL to the loop above.  Speedup is 1.2-2x
        // for matmul-heavy paths; combined with sync removal, more.
        const long long stride_a = (batch_a == 1)
                                       ? 0LL
                                       : static_cast<long long>(m) * k;
        const long long stride_b = (batch_b == 1)
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
    warn_host_fallback_once("matmul", device);
    const float* a_ptr = data();
    const float* b_ptr = other.data();
    float* out_ptr = result.data();

    // CPU: BLIS-style blocked GEMM (MathOps::gemm) per batch -- same contiguous
    // row-major layout the naive loop below assumes, but an order of magnitude
    // faster.  beta=0 overwrites the freshly-allocated result.  The naive loop is
    // kept as a fallback for any non-CPU data that reaches here.
    if (device == Device::CPU) {
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const float* a_batch =
                a_ptr + (batch_a == 1
                             ? 0
                             : static_cast<size_t>(batch_idx) * m * k);
            const float* b_batch =
                b_ptr + (batch_b == 1
                             ? 0
                             : static_cast<size_t>(batch_idx) * k * n);
            float* out_batch = out_ptr + static_cast<size_t>(batch_idx) * m * n;
            MathOps::gemm(m, n, k, 1.0f, a_batch, k, b_batch, n, 0.0f, out_batch, n);
        }
        return result;
    }

#pragma omp parallel for
    for (int row_index = 0; row_index < batch * m; ++row_index) {
        const int batch_idx = row_index / m;
        const int row = row_index % m;
        const float* a_batch =
            a_ptr + (batch_a == 1 ? 0 : batch_idx * m * k);
        const float* b_batch =
            b_ptr + (batch_b == 1 ? 0 : batch_idx * k * n);
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
    Tensor result = Tensor::uninitialized(out_dims, device);

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported() &&
        rank == 2 && dim0 == 0 && dim1 == 1) {
        launch_transpose2d_kernel(result.raw_data(), raw_data(), shape[0], shape[1]);
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
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
    Tensor result =
        Tensor::uninitialized(dy.shape.dims, dy.get_device());
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

Tensor Tensor::silu() const {
    Tensor result = Tensor::uninitialized(shape.dims, device);
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_silu_kernel(result.raw_data(), raw_data(), size);
        sync_cuda();
        return result;
    }
#endif
    const float* source = data();
    float* destination = result.data();
#pragma omp parallel for if(size > 1024)
    for (int index = 0; index < size; ++index) {
        const float value = source[index];
        const float sigmoid =
            value >= 0.0f
                ? 1.0f / (1.0f + std::exp(-value))
                : std::exp(value) / (1.0f + std::exp(value));
        destination[index] = value * sigmoid;
    }
    return result;
}

Tensor Tensor::silu_backward(const Tensor& dy,
                             const Tensor& pre_activation) {
    if (dy.shape != pre_activation.shape) {
        throw std::invalid_argument(
            "silu_backward: dy and pre_activation must have same shape");
    }
    if (dy.get_device() != pre_activation.get_device()) {
        throw std::invalid_argument(
            "silu_backward: dy and pre_activation must be on same device");
    }
    Tensor result =
        Tensor::uninitialized(dy.shape.dims, dy.get_device());
#ifdef USE_CUDA
    if (use_gpu_fast_path(dy, pre_activation) &&
        gpu_custom_kernels_supported()) {
        launch_silu_backward_kernel(
            result.raw_data(), dy.raw_data(),
            pre_activation.raw_data(), dy.size);
        sync_cuda();
        return result;
    }
#endif
    const float* gradient = dy.data();
    const float* pre = pre_activation.data();
    float* destination = result.data();
#pragma omp parallel for if(dy.size > 1024)
    for (int index = 0; index < dy.size; ++index) {
        const float value = pre[index];
        const float sigmoid =
            value >= 0.0f
                ? 1.0f / (1.0f + std::exp(-value))
                : std::exp(value) / (1.0f + std::exp(value));
        destination[index] =
            gradient[index] * sigmoid *
            (1.0f + value * (1.0f - sigmoid));
    }
    return result;
}

Tensor Tensor::silu_gate(const Tensor& value, const Tensor& gate) {
    if (value.shape != gate.shape ||
        value.get_device() != gate.get_device()) {
        throw std::invalid_argument(
            "silu_gate requires identical shapes and devices");
    }
#ifdef USE_CUDA
    if (use_gpu_fast_path(value, gate) && gpu_custom_kernels_supported()) {
        Tensor result = Tensor::uninitialized(
            value.shape.dims, value.get_device());
        if (!launch_silu_gate_forward_kernel(
                result.raw_data(), value.raw_data(), gate.raw_data(),
                value.size)) {
            throw std::runtime_error(
                "silu_gate GPU launcher rejected invalid arguments");
        }
        sync_cuda();
        return result;
    }
#endif
    return value.mul(gate.silu());
}

std::pair<Tensor, Tensor> Tensor::silu_gate_backward(
    const Tensor& grad_out, const Tensor& value, const Tensor& gate) {
    if (grad_out.shape != value.shape || value.shape != gate.shape ||
        grad_out.get_device() != value.get_device() ||
        value.get_device() != gate.get_device()) {
        throw std::invalid_argument(
            "silu_gate_backward requires identical shapes and devices");
    }
#ifdef USE_CUDA
    if (use_gpu_fast_path(grad_out, value) &&
        gate.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        Tensor value_grad = Tensor::uninitialized(
            value.shape.dims, Device::GPU);
        Tensor gate_grad = Tensor::uninitialized(
            gate.shape.dims, Device::GPU);
        if (!launch_silu_gate_backward_kernel(
            value_grad.raw_data(), gate_grad.raw_data(),
            grad_out.raw_data(), value.raw_data(), gate.raw_data(),
            value.size)) {
            throw std::runtime_error(
                "silu_gate_backward GPU launcher rejected invalid arguments");
        }
        sync_cuda();
        return {std::move(value_grad), std::move(gate_grad)};
    }
#endif
    Tensor activated_gate = gate.silu();
    Tensor value_grad = grad_out.mul(activated_gate);
    Tensor gate_grad = Tensor::silu_backward(
        grad_out.mul(value), gate);
    return {std::move(value_grad), std::move(gate_grad)};
}

Tensor Tensor::sigmoid() const {
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
        const float value = src[i];
        dst[i] =
            value >= 0.0f
                ? 1.0f / (1.0f + std::exp(-value))
                : std::exp(value) / (1.0f + std::exp(value));
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    int rank = static_cast<int>(shape.size());
    dim = normalize_dim(dim, rank);

    int axis = shape[dim];
    if (axis <= 0 || size <= 0) {
        throw std::invalid_argument(
            "Tensor::softmax requires a non-empty reduction axis");
    }
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
    if (!std::isfinite(eps) || eps <= 0.0f ||
        shape.empty() || shape.back() <= 0 || size <= 0) {
        throw std::invalid_argument(
            "Tensor::rmsnorm requires a non-empty tensor and finite eps > 0");
    }
    int inner = shape.back();
    int outer = size / inner;
    Tensor result = Tensor::uninitialized(shape.dims, device);

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_rmsnorm_kernel(result.raw_data(), raw_data(), outer, inner, eps);
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
    if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
        min_val > max_val) {
        throw std::invalid_argument(
            "Tensor::clamp requires finite bounds with min <= max");
    }
    Tensor result = Tensor::uninitialized(shape.dims, device);
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
        Tensor result = Tensor::uninitialized({1}, device);
#ifdef USE_CUDA
        if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
            launch_mean_kernel(result.raw_data(), raw_data(), 1, size, 1);
            launch_scale_inplace_kernel(result.raw_data(),
                                         static_cast<float>(size), 1);
            sync_cuda();
            return result;
        }
#endif
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
        Tensor result = Tensor::uninitialized(gpu_out_dims, device);
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
        float* d_sum_sq = scalar_reduction_scratch();
        if (d_sum_sq == nullptr) {
            throw std::runtime_error(
                "Tensor::norm could not allocate GPU reduction scratch");
        }
        const cudaError_t clear_status =
            cudaMemsetAsync(d_sum_sq, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Tensor::norm GPU clear failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(clear_status));
        }
        launch_norm_kernel(d_sum_sq, raw_data(), size);
        sync_cuda();
        const float sum_sq = copy_scalar_from_device(d_sum_sq);
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
        copy_tensor_bytes(result.raw_data(), result.device, raw_data(), device,
                          size * sizeof(float));
    }
    return result;
}

Tensor Tensor::cpu() const {
    return to(Device::CPU);
}

Tensor Tensor::storage_view(
    size_t element_offset, const std::vector<int>& view_shape) const {
    Tensor view;
    view.shape = TensorShape(view_shape);
    view.size = checked_tensor_size(view.shape);
    view.device = device;
    const size_t source_size =
        size < 0 ? 0U : static_cast<size_t>(size);
    const size_t view_size = static_cast<size_t>(view.size);
    if (element_offset > source_size ||
        view_size > source_size - element_offset) {
        throw std::out_of_range(
            "Tensor::storage_view exceeds the source storage");
    }
    if (view_size > 0 && data_ptr == nullptr) {
        throw std::logic_error(
            "Tensor::storage_view cannot alias null storage");
    }
    if (data_ptr != nullptr) {
        view.data_ptr =
            std::shared_ptr<float>(
                data_ptr, data_ptr.get() + element_offset);
    }
    view.host_accessible_storage_ = host_accessible_storage_;
    return view;
}

void Tensor::sync_host_access() const {
#ifdef USE_CUDA
    if (device == Device::GPU && size > 0) {
        // Cheap-when-idle barrier:
        //   * cudaStreamQuery( nsos::gpu::current_stream()) returns cudaSuccess in microseconds when
        //     the default stream has no in-flight work — the common case
        //     for back-to-back data() reads after the first sync.
        //   * Only when there IS pending work (cudaErrorNotReady) do we
        //     pay the cost of cudaDeviceSynchronize.
        // This lets eager-sync mode stay viable for production: a chain
        // of data() reads after a single kernel launch syncs once and
        // then no-ops, instead of bottlenecking on N cudaDeviceSync.
        const cudaError_t pending = cudaStreamQuery( nsos::gpu::current_stream());
        if (pending == cudaErrorNotReady) {
            const cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA host-access synchronization failed: ") +
                    cudaGetErrorString(sync_status));
            }
            record_gpu_device_synchronization();
        } else if (pending != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA stream query failed before host access: ") +
                cudaGetErrorString(pending));
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
#ifdef USE_CUDA
    if (device == Device::GPU && size > 0 &&
        !host_accessible_storage_) {
        throw std::runtime_error(
            "Host access to a cudaMalloc-backed GPU Tensor is forbidden; "
            "copy with cpu() or pass raw_data() to a CUDA API");
    }
#endif
    sync_host_access();
    return data_ptr.get();
}

const float* Tensor::data() const {
#ifdef USE_CUDA
    if (device == Device::GPU && size > 0 &&
        !host_accessible_storage_) {
        throw std::runtime_error(
            "Host access to a cudaMalloc-backed GPU Tensor is forbidden; "
            "copy with cpu() or pass raw_data() to a CUDA API");
    }
#endif
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
        copy_tensor_bytes(raw_data(), device, other.raw_data(), other.device,
                          size * sizeof(float));
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

#ifdef USE_CUDA
    if (device == Device::GPU && gpu_custom_kernels_supported()) {
        const int inner =
            static_cast<int>(shape.strides[static_cast<size_t>(dim)]);
        const int axis_size = shape[dim];
        const int axis_span =
            checked_int_product(axis_size, inner, "slice axis span");
        if (axis_span <= 0 || size % axis_span != 0) {
            throw std::logic_error(
                "Tensor::slice source storage is inconsistent with shape");
        }
        const int outer = size / axis_span;
        const int slice_size = end - start;
        const int expected =
            checked_int_product(
                checked_int_product(outer, slice_size,
                                    "slice output rows"),
                inner, "slice output elements");
        if (expected != result.size) {
            throw std::logic_error(
                "Tensor::slice output storage is inconsistent with shape");
        }
        if (!launch_slice_contiguous_kernel(
            result.raw_data(), raw_data(), outer, axis_size, inner, start,
            slice_size, result.size)) {
            throw std::runtime_error(
                "Tensor::slice GPU launcher rejected invalid arguments");
        }
        sync_cuda();
        return result;
    }
    if (device == Device::GPU) {
        throw std::runtime_error(
            "Tensor::slice requires the compiled GPU kernel for a GPU tensor");
    }
#elif !defined(USE_CUDA)
    if (device == Device::GPU) {
        throw std::runtime_error(
            "Tensor::slice received a GPU tensor in a CPU-only build");
    }
#endif

    if (dim == 0 && rank >= 1) {
        const int inner = size / shape[0];
        const size_t bytes =
            static_cast<size_t>((end - start) * inner) * sizeof(float);
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

Tensor Tensor::rmsnorm_backward(const Tensor& grad, const Tensor& x_norm,
                                float eps) const {
    if (shape != grad.shape || shape != x_norm.shape ||
        size != grad.size || size != x_norm.size) {
        throw std::invalid_argument(
            "Tensor::rmsnorm_backward requires identical tensor shapes");
    }
    if (device != grad.device || device != x_norm.device) {
        throw std::invalid_argument(
            "Tensor::rmsnorm_backward requires identical devices");
    }
    if (!std::isfinite(eps) || eps <= 0.0f ||
        shape.empty() || shape.back() <= 0 || size <= 0) {
        throw std::invalid_argument(
            "Tensor::rmsnorm_backward requires a non-empty tensor and "
            "finite eps > 0");
    }
    int inner = shape.back();
    int outer = size / inner;
    Tensor dx = Tensor::uninitialized(shape.dims, device);

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, grad) && x_norm.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        launch_rmsnorm_backward_kernel(dx.raw_data(), grad.raw_data(),
                                        x_norm.raw_data(), raw_data(),
                                        outer, inner, eps);
        sync_cuda();
        return dx;
    }
#endif

    const float* g = grad.data();
    const float* y = x_norm.data();
    const float* x = data();  // *this is the original pre-norm input
    float* dx_ptr = dx.data();
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
    thread_local cuda_detail::DeviceBuffer<float> buffer;
    return buffer.ensure(1);
}

class CrossEntropyRowWorkspace {
public:
    struct DeviceRows {
        int* targets = nullptr;
        float* weights = nullptr;
    };

    ~CrossEntropyRowWorkspace() {
        if (!wait_for_consumer_noexcept() ||
            awaiting_completion_record_ || poisoned_) {
            // The runtime could not prove completion. Retain every allocation
            // rather than freeing pinned/device memory that an async copy may
            // still reference.
            device_targets_.abandon();
            device_weights_.abandon();
            host_targets_.abandon();
            host_weights_.abandon();
            copy_complete_ = nullptr;
            return;
        }
        if (copy_complete_ != nullptr) {
            const cudaError_t status =
                cudaEventDestroy(copy_complete_);
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
            }
        }
    }

    DeviceRows stage(const std::vector<int>& targets,
                     const std::vector<float>* weights) {
        if (targets.empty()) {
            throw std::invalid_argument(
                "cross-entropy GPU staging requires at least one row");
        }
        if (weights != nullptr && weights->size() != targets.size()) {
            throw std::invalid_argument(
                "cross-entropy target/weight staging size mismatch");
        }
        wait_for_consumer();
        ensure(targets.size());
        std::copy(
            targets.begin(), targets.end(), host_targets_.get());
        if (weights != nullptr) {
            std::copy(
                weights->begin(), weights->end(), host_weights_.get());
        }
        check(cudaMemcpyAsync(
                  device_targets_.get(), host_targets_.get(),
                  targets.size() * sizeof(int),
                  cudaMemcpyHostToDevice, nsos::gpu::current_stream()),
              "cross-entropy target upload");
        awaiting_completion_record_ = true;
        record_gpu_transfer(
            Device::GPU, Device::CPU,
            targets.size() * sizeof(int));
        if (weights != nullptr) {
            const cudaError_t weight_status =
                cudaMemcpyAsync(
                    device_weights_.get(), host_weights_.get(),
                    weights->size() * sizeof(float),
                    cudaMemcpyHostToDevice, nsos::gpu::current_stream());
            if (weight_status != cudaSuccess) {
                fail_after_unrecorded_work(
                    weight_status,
                    "cross-entropy weight upload");
            }
            record_gpu_transfer(
                Device::GPU, Device::CPU,
                weights->size() * sizeof(float));
        }
        const cudaError_t event_status =
            cudaEventRecord(copy_complete_, nsos::gpu::current_stream());
        if (event_status != cudaSuccess) {
            fail_after_unrecorded_work(
                event_status,
                "cross-entropy staging event record");
        }
        awaiting_completion_record_ = false;
        copy_pending_ = true;
        consumer_pending_ = true;
        return {
            device_targets_.get(),
            weights != nullptr ? device_weights_.get() : nullptr};
    }

    void complete_consumer() {
        if (!consumer_pending_) {
            throw std::logic_error(
                "cross-entropy GPU consumer completion was not pending");
        }
        // The caller reached a blocking default-stream D2H boundary after
        // every consumer kernel. That proves both upload and consumption.
        consumer_pending_ = false;
        copy_pending_ = false;
    }

    void record_consumer_completion() {
        if (!consumer_pending_) {
            throw std::logic_error(
                "cross-entropy GPU consumer completion was not pending");
        }
        const cudaError_t status = cudaEventRecord(copy_complete_, nsos::gpu::current_stream());
        if (status != cudaSuccess) {
            fail_consumer_completion(
                status, "cross-entropy consumer event record");
        }
        // The event now represents the last consumer, not merely the upload.
        // A subsequent stage waits/query-checks it before overwriting buffers.
        consumer_pending_ = false;
        copy_pending_ = true;
    }

    [[noreturn]] void fail_consumer_completion(
        cudaError_t original_status,
        const char* operation) {
        const cudaError_t sync_status =
            cudaStreamSynchronize( nsos::gpu::current_stream());
        record_gpu_stream_synchronization();
        if (sync_status != cudaSuccess) {
            poisoned_ = true;
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(original_status) +
                "; consumer completion recovery also failed: " +
                cudaGetErrorString(sync_status));
        }
        consumer_pending_ = false;
        copy_pending_ = false;
        check(original_status, operation);
        throw std::logic_error(
            "unreachable cross-entropy consumer failure path");
    }

private:
    static void check(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
    }

    void wait_for_consumer() {
        if (poisoned_) {
            throw std::runtime_error(
                "cross-entropy GPU staging is poisoned after an "
                "unrecoverable completion error");
        }
        if (consumer_pending_) {
            const cudaError_t status =
                cudaStreamSynchronize( nsos::gpu::current_stream());
            record_gpu_stream_synchronization();
            if (status != cudaSuccess) {
                poisoned_ = true;
                check(status,
                      "cross-entropy consumer synchronization");
            }
            consumer_pending_ = false;
            copy_pending_ = false;
            return;
        }
        if (!copy_pending_) {
            return;
        }
        cudaError_t status = cudaEventQuery(copy_complete_);
        if (status == cudaErrorNotReady) {
            status = cudaEventSynchronize(copy_complete_);
            record_gpu_stream_synchronization();
        }
        if (status != cudaSuccess) {
            poisoned_ = true;
            check(status,
                  "cross-entropy staging reuse synchronization");
        }
        copy_pending_ = false;
    }

    bool wait_for_consumer_noexcept() noexcept {
        if (consumer_pending_) {
            const cudaError_t status =
                cudaStreamSynchronize( nsos::gpu::current_stream());
            record_gpu_stream_synchronization();
            if (status != cudaSuccess) {
                poisoned_ = true;
                return false;
            }
            consumer_pending_ = false;
            copy_pending_ = false;
        }
        if (copy_pending_ && copy_complete_ != nullptr) {
            cudaError_t status =
                cudaEventQuery(copy_complete_);
            if (status == cudaErrorNotReady) {
                status =
                    cudaEventSynchronize(copy_complete_);
                record_gpu_stream_synchronization();
            }
            if (status != cudaSuccess) {
                poisoned_ = true;
                return false;
            }
            copy_pending_ = false;
        }
        return !poisoned_ && !awaiting_completion_record_;
    }

    [[noreturn]] void fail_after_unrecorded_work(
        cudaError_t original_status,
        const char* operation) {
        const cudaError_t sync_status =
            cudaStreamSynchronize( nsos::gpu::current_stream());
        record_gpu_stream_synchronization();
        if (sync_status != cudaSuccess) {
            poisoned_ = true;
            throw std::runtime_error(
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(original_status) +
                "; completion recovery also failed: " +
                cudaGetErrorString(sync_status));
        }
        awaiting_completion_record_ = false;
        check(original_status, operation);
        throw std::logic_error(
            "unreachable cross-entropy staging failure path");
    }

    void ensure(size_t requested) {
        const size_t minimum = std::max<size_t>(requested, 256);
        if (device_targets_.ensure(minimum) == nullptr) {
            throw std::runtime_error(
                "cross-entropy device target allocation failed");
        }
        if (device_weights_.ensure(minimum) == nullptr) {
            throw std::runtime_error(
                "cross-entropy device weight allocation failed");
        }
        if (host_targets_.ensure(minimum) == nullptr) {
            throw std::runtime_error(
                "cross-entropy pinned target allocation failed");
        }
        if (host_weights_.ensure(minimum) == nullptr) {
            throw std::runtime_error(
                "cross-entropy pinned weight allocation failed");
        }
        if (copy_complete_ == nullptr) {
            const cudaError_t status =
                cudaEventCreate(&copy_complete_);
            if (status != cudaSuccess) {
                check(status,
                      "cross-entropy staging event creation");
            }
        }
    }

    cuda_detail::DeviceBuffer<int> device_targets_;
    cuda_detail::DeviceBuffer<float> device_weights_;
    cuda_detail::PinnedHostBuffer<int> host_targets_;
    cuda_detail::PinnedHostBuffer<float> host_weights_;
    cudaEvent_t copy_complete_ = nullptr;
    bool copy_pending_ = false;
    bool consumer_pending_ = false;
    bool awaiting_completion_record_ = false;
    bool poisoned_ = false;
};

CrossEntropyRowWorkspace& ce_row_workspace() {
    thread_local CrossEntropyRowWorkspace workspace;
    return workspace;
}
#endif

std::pair<float, Tensor> Tensor::cross_entropy(const std::vector<int>& target) const {
    int rank = static_cast<int>(shape.size());
    if (rank < 2) {
        throw std::runtime_error("cross_entropy expects rank >= 2 logits");
    }

    const int classes = shape.back();
    if (classes <= 0 || size % classes != 0 ||
        size / classes > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "cross_entropy has an invalid logits shape");
    }
    const int rows = static_cast<int>(size / classes);
    if (static_cast<int>(target.size()) != rows) {
        throw std::runtime_error("cross_entropy target size mismatch");
    }
    for (int row = 0; row < rows; ++row) {
        if (target[row] < 0 || target[row] >= classes) {
            throw std::out_of_range("cross_entropy target out of range");
        }
    }

    // Every validated row is written exhaustively by both the GPU and CPU
    // producers below.
    Tensor grad = Tensor::uninitialized(shape.dims, device);
    float loss = 0.0f;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this)) {
        if (!gpu_custom_kernels_supported()) {
            throw std::runtime_error(
                "cross_entropy has no supported GPU kernel for this backend");
        }
        float* d_loss = ce_loss_scratch();
        if (d_loss == nullptr) {
            throw std::runtime_error(
                "cross_entropy could not allocate GPU loss scratch");
        }
        const cudaError_t clear_status =
            cudaMemsetAsync(d_loss, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cross_entropy GPU loss clear failed: ") +
                cudaGetErrorString(clear_status));
        }
        auto& row_workspace = ce_row_workspace();
        const auto staged = row_workspace.stage(target, nullptr);
        Tensor row_losses;
        if (determinism::deterministic_reductions_enabled()) {
            row_losses = Tensor::uninitialized({rows}, Device::GPU);
            if (!launch_fused_cross_entropy_deterministic(
                d_loss, row_losses.raw_data(), grad.raw_data(), raw_data(),
                staged.targets, nullptr, rows, classes)) {
                throw std::runtime_error(
                    "Deterministic cross-entropy launcher rejected invalid "
                    "arguments");
            }
        } else {
            launch_fused_cross_entropy(d_loss, grad.raw_data(), raw_data(),
                                       staged.targets, rows, classes);
        }
        sync_cuda();
        const float inv_rows = 1.0f / std::max(rows, 1);
        launch_scale_inplace_kernel(grad.raw_data(), inv_rows, grad.size);
        sync_cuda();
        float downloaded_loss = 0.0f;
        const cudaError_t download_status =
            cudaMemcpy(&downloaded_loss, d_loss, sizeof(float),
                       cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (download_status != cudaSuccess) {
            row_workspace.fail_consumer_completion(
                download_status, "cross_entropy GPU loss download");
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        row_workspace.complete_consumer();
        loss = downloaded_loss * inv_rows;
        return {loss, grad};
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

        const float inv_sum = 1.0f / sum_exp;
        int target_class = target[row];
        if (target_class < 0 || target_class >= classes) {
            throw std::out_of_range("cross_entropy target out of range");
        }

        for (int c = 0; c < classes; ++c) {
            grad_ptr[row * classes + c] *= inv_sum;
        }

        // Stable log-sum-exp NLL.  The former max(prob, 1e-8) cap made the
        // scalar flat in the tail while returning a non-zero softmax gradient.
        loss += max_logit + std::log(sum_exp) - row_ptr[target_class];
        grad_ptr[row * classes + target_class] -= 1.0f;
    }

    float inv_rows = 1.0f / std::max(rows, 1);
    for (int i = 0; i < grad.size; ++i) {
        grad_ptr[i] *= inv_rows;
    }

    return {loss * inv_rows, grad};
}

std::pair<Tensor, Tensor> Tensor::cross_entropy_device(
    const std::vector<int>& target) const {
    const int rank = static_cast<int>(shape.size());
    if (rank < 2) {
        throw std::runtime_error("cross_entropy_device expects rank >= 2 logits");
    }
    const int classes = shape.back();
    if (classes <= 0 || size % classes != 0 ||
        size / classes > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "cross_entropy_device has an invalid logits shape");
    }
    const int rows = static_cast<int>(size / classes);
    if (static_cast<int>(target.size()) != rows) {
        throw std::runtime_error("cross_entropy_device target size mismatch");
    }
    for (int row = 0; row < rows; ++row) {
        if (target[static_cast<size_t>(row)] < 0 ||
            target[static_cast<size_t>(row)] >= classes) {
            throw std::out_of_range(
                "cross_entropy_device target out of range");
        }
    }

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this)) {
        if (!gpu_custom_kernels_supported()) {
            throw std::runtime_error(
                "cross_entropy_device has no supported GPU kernel for this backend");
        }
        Tensor device_loss = Tensor::zeros({1}, Device::GPU);
        Tensor grad = Tensor::uninitialized(shape.dims, Device::GPU);
        auto& row_workspace = ce_row_workspace();
        const auto staged = row_workspace.stage(target, nullptr);
        Tensor row_losses;
        if (determinism::deterministic_reductions_enabled()) {
            row_losses = Tensor::uninitialized({rows}, Device::GPU);
            if (!launch_fused_cross_entropy_deterministic(
                device_loss.raw_data(), row_losses.raw_data(),
                grad.raw_data(), raw_data(), staged.targets, nullptr,
                rows, classes)) {
                throw std::runtime_error(
                    "Deterministic device cross-entropy launcher rejected "
                    "invalid arguments");
            }
        } else {
            launch_fused_cross_entropy(
                device_loss.raw_data(), grad.raw_data(), raw_data(),
                staged.targets, rows, classes);
        }
        sync_cuda();
        const float inv_rows = 1.0f / std::max(rows, 1);
        launch_scale_inplace_kernel(grad.raw_data(), inv_rows, grad.size);
        launch_scale_inplace_kernel(device_loss.raw_data(), inv_rows, 1);
        sync_cuda();
        row_workspace.record_consumer_completion();
        return {std::move(device_loss), std::move(grad)};
    }
#endif

    auto [loss, grad] = cross_entropy(target);
    Tensor loss_tensor({1}, Device::CPU);
    loss_tensor.data()[0] = loss;
    return {std::move(loss_tensor), std::move(grad)};
}

std::pair<float, Tensor> Tensor::cross_entropy_weighted(
    const std::vector<int>& target,
    const std::vector<float>& row_weights) const {
    const int rank = static_cast<int>(shape.size());
    if (rank < 2) {
        throw std::runtime_error("cross_entropy_weighted expects rank >= 2 logits");
    }
    const int classes = shape.back();
    if (classes <= 0 || size % classes != 0 ||
        size / classes > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "cross_entropy_weighted has an invalid logits shape");
    }
    const int rows = static_cast<int>(size / classes);
    if (static_cast<int>(target.size()) != rows ||
        static_cast<int>(row_weights.size()) != rows) {
        throw std::runtime_error("cross_entropy_weighted row count mismatch");
    }
    for (int row = 0; row < rows; ++row) {
        if (target[static_cast<size_t>(row)] < 0 ||
            target[static_cast<size_t>(row)] >= classes) {
            throw std::out_of_range("cross_entropy_weighted target out of range");
        }
        const float weight_value = row_weights[static_cast<size_t>(row)];
        if (!std::isfinite(weight_value) || weight_value < 0.0f) {
            throw std::invalid_argument(
                "cross_entropy_weighted requires finite non-negative row weights");
        }
    }

    // Targets were validated above, so every gradient row is overwritten.
    Tensor grad = Tensor::uninitialized(shape.dims, device);
    float loss = 0.0f;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this)) {
        if (!gpu_custom_kernels_supported()) {
            throw std::runtime_error(
                "cross_entropy_weighted has no supported GPU kernel for "
                "this backend");
        }
        float* d_loss = ce_loss_scratch();
        if (d_loss == nullptr) {
            throw std::runtime_error(
                "cross_entropy_weighted could not allocate GPU loss scratch");
        }
        const cudaError_t clear_status =
            cudaMemsetAsync(d_loss, 0, sizeof(float), nsos::gpu::current_stream());
        if (clear_status != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "cross_entropy_weighted GPU loss clear failed: ") +
                cudaGetErrorString(clear_status));
        }
        auto& row_workspace = ce_row_workspace();
        const auto staged =
            row_workspace.stage(target, &row_weights);
        Tensor row_losses;
        if (determinism::deterministic_reductions_enabled()) {
            row_losses = Tensor::uninitialized({rows}, Device::GPU);
            if (!launch_fused_cross_entropy_deterministic(
                d_loss, row_losses.raw_data(), grad.raw_data(), raw_data(),
                staged.targets, staged.weights, rows, classes)) {
                throw std::runtime_error(
                    "Deterministic weighted cross-entropy launcher rejected "
                    "invalid arguments");
            }
        } else {
            launch_fused_cross_entropy_weighted(
                d_loss, grad.raw_data(), raw_data(), staged.targets,
                staged.weights, rows, classes);
        }
        sync_cuda();
        const float inv_rows = 1.0f / std::max(rows, 1);
        launch_scale_inplace_kernel(grad.raw_data(), inv_rows, grad.size);
        sync_cuda();
        float downloaded_loss = 0.0f;
        const cudaError_t download_status =
            cudaMemcpy(&downloaded_loss, d_loss, sizeof(float),
                       cudaMemcpyDeviceToHost);
        record_gpu_stream_synchronization();
        if (download_status != cudaSuccess) {
            row_workspace.fail_consumer_completion(
                download_status,
                "cross_entropy_weighted GPU loss download");
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(float));
        row_workspace.complete_consumer();
        loss = downloaded_loss * inv_rows;
        return {loss, grad};
    }
#endif

    const float* logits = data();
    float* grad_ptr = grad.data();
    for (int row = 0; row < rows; ++row) {
        const float* row_ptr = logits + static_cast<size_t>(row) * classes;
        float max_logit = row_ptr[0];
        for (int col = 1; col < classes; ++col) {
            max_logit = std::max(max_logit, row_ptr[col]);
        }
        float sum_exp = 0.0f;
        for (int col = 0; col < classes; ++col) {
            const float exponential = std::exp(row_ptr[col] - max_logit);
            grad_ptr[static_cast<size_t>(row) * classes + col] = exponential;
            sum_exp += exponential;
        }
        const float inverse_sum = 1.0f / sum_exp;
        const float row_weight = row_weights[static_cast<size_t>(row)];
        for (int col = 0; col < classes; ++col) {
            grad_ptr[static_cast<size_t>(row) * classes + col] *= inverse_sum;
        }
        const int target_class = target[static_cast<size_t>(row)];
        loss += row_weight *
                (max_logit + std::log(sum_exp) - row_ptr[target_class]);
        grad_ptr[static_cast<size_t>(row) * classes + target_class] -= 1.0f;
        for (int col = 0; col < classes; ++col) {
            grad_ptr[static_cast<size_t>(row) * classes + col] *= row_weight;
        }
    }
    const float inv_rows = 1.0f / std::max(rows, 1);
    for (int index = 0; index < grad.size; ++index) {
        grad_ptr[index] *= inv_rows;
    }
    return {loss * inv_rows, grad};
}

std::pair<Tensor, Tensor> Tensor::cross_entropy_weighted_masked_sum_device(
    const std::vector<int>& target,
    const std::vector<float>& row_weights) const {
    const int rank = static_cast<int>(shape.size());
    if (rank < 2) {
        throw std::runtime_error(
            "cross_entropy_weighted_masked_sum expects rank >= 2 logits");
    }
    const int classes = shape.back();
    if (classes <= 0 || size % classes != 0 ||
        size / classes > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "cross_entropy_weighted_masked_sum has an invalid logits shape");
    }
    const int rows = static_cast<int>(size / classes);
    if (static_cast<int>(target.size()) != rows ||
        static_cast<int>(row_weights.size()) != rows) {
        throw std::runtime_error(
            "cross_entropy_weighted_masked_sum row count mismatch");
    }
    for (int row = 0; row < rows; ++row) {
        const int target_value = target[static_cast<size_t>(row)];
        if (target_value < -1 || target_value >= classes) {
            throw std::out_of_range(
                "cross_entropy_weighted_masked_sum target out of range");
        }
        const float weight_value = row_weights[static_cast<size_t>(row)];
        if (!std::isfinite(weight_value) || weight_value < 0.0f) {
            throw std::invalid_argument(
                "cross_entropy_weighted_masked_sum requires finite "
                "non-negative row weights");
        }
        if (target_value == -1 && weight_value != 0.0f) {
            throw std::invalid_argument(
                "cross_entropy_weighted_masked_sum ignored rows require "
                "zero weight");
        }
    }

    Tensor grad = Tensor::zeros(shape.dims, device);
    float loss = 0.0f;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this)) {
        if (!gpu_custom_kernels_supported()) {
            throw std::runtime_error(
                "cross_entropy_weighted_masked_sum has no supported GPU "
                "kernel for this backend");
        }
        Tensor device_loss = Tensor::zeros({1}, Device::GPU);
        float* d_loss = device_loss.raw_data();
        auto& row_workspace = ce_row_workspace();
        const auto staged =
            row_workspace.stage(target, &row_weights);
        Tensor row_losses;
        if (determinism::deterministic_reductions_enabled()) {
            row_losses = Tensor::uninitialized({rows}, Device::GPU);
            if (!launch_fused_cross_entropy_deterministic(
                d_loss, row_losses.raw_data(), grad.raw_data(), raw_data(),
                staged.targets, staged.weights, rows, classes)) {
                throw std::runtime_error(
                    "Deterministic masked cross-entropy launcher rejected "
                    "invalid arguments");
            }
        } else {
            launch_fused_cross_entropy_weighted(
                d_loss, grad.raw_data(), raw_data(), staged.targets,
                staged.weights, rows, classes);
        }
        sync_cuda();
        row_workspace.record_consumer_completion();
        return {std::move(device_loss), std::move(grad)};
    }
#endif

    const float* logits = data();
    float* grad_ptr = grad.data();
    for (int row = 0; row < rows; ++row) {
        const int target_class = target[static_cast<size_t>(row)];
        const float row_weight = row_weights[static_cast<size_t>(row)];
        if (target_class < 0 || row_weight == 0.0f) {
            continue;
        }
        const float* row_ptr =
            logits + static_cast<size_t>(row) * classes;
        float* grad_row =
            grad_ptr + static_cast<size_t>(row) * classes;
        float max_logit = row_ptr[0];
        for (int col = 1; col < classes; ++col) {
            max_logit = std::max(max_logit, row_ptr[col]);
        }
        double sum_exp = 0.0;
        for (int col = 0; col < classes; ++col) {
            const float value = std::exp(row_ptr[col] - max_logit);
            grad_row[col] = value;
            sum_exp += value;
        }
        const double inv_sum = 1.0 / sum_exp;
        for (int col = 0; col < classes; ++col) {
            grad_row[col] = row_weight *
                static_cast<float>(grad_row[col] * inv_sum);
        }
        grad_row[target_class] -= row_weight;
        loss += row_weight * static_cast<float>(
            max_logit + std::log(sum_exp) - row_ptr[target_class]);
    }
    Tensor loss_tensor({1}, Device::CPU);
    loss_tensor.data()[0] = loss;
    return {std::move(loss_tensor), std::move(grad)};
}

std::pair<float, Tensor> Tensor::cross_entropy_weighted_masked_sum(
    const std::vector<int>& target,
    const std::vector<float>& row_weights) const {
    auto result = cross_entropy_weighted_masked_sum_device(
        target, row_weights);
    Tensor host_loss = result.first.get_device() == Device::GPU
                           ? result.first.cpu()
                           : result.first;
    if (host_loss.size != 1) {
        throw std::logic_error(
            "cross_entropy_weighted_masked_sum returned invalid loss scalar");
    }
    return {host_loss.data()[0], std::move(result.second)};
}

std::pair<float, Tensor> Tensor::mse_loss(const Tensor& target) const {
    if (size != target.size) {
        throw std::runtime_error("mse_loss shape mismatch");
    }

    Tensor aligned_target =
        target.shape == shape ? target : target.reshape(shape.dims);
    if (aligned_target.get_device() != device) {
        aligned_target = aligned_target.to(device);
    }

#ifdef USE_CUDA
    if (device == Device::GPU && gpu_custom_kernels_supported()) {
        Tensor diff = sub(aligned_target);
        const float l2_norm = diff.norm();
        const float inv_size =
            1.0f / static_cast<float>(std::max<int64_t>(size, 1));
        return {l2_norm * l2_norm * inv_size,
                diff.mul(2.0f * inv_size)};
    }
#endif

    Tensor grad(shape.dims, device);
    const float* src = data();
    const float* tgt = aligned_target.data();
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
    if (device == Device::GPU) {
        cpu().print(name, max_elements);
        return;
    }
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
    bool source_host_accessible = d == Device::CPU;
    int source_gpu_device = -1;
    if (t.size > 0 && d == Device::GPU) {
#ifdef USE_CUDA
        cudaPointerAttributes attributes{};
        const cudaError_t pointer_status =
            cudaPointerGetAttributes(&attributes, ptr);
        if (pointer_status != cudaSuccess) {
            (void)cudaGetLastError();
            throw std::invalid_argument(
                "from_blob Device::GPU requires a CUDA device or managed pointer");
        }
#if CUDART_VERSION >= 10000
        if (attributes.type != cudaMemoryTypeDevice &&
            attributes.type != cudaMemoryTypeManaged) {
            throw std::invalid_argument(
                "from_blob Device::GPU rejects host-pointer storage");
        }
        int selected_device = -1;
        std::string selection_error;
        if (!gpu::select_preferred_device(
                &selected_device, &selection_error)) {
            throw std::runtime_error(
                "from_blob could not select a GPU: " +
                selection_error);
        }
        if (attributes.type == cudaMemoryTypeDevice &&
            attributes.device != selected_device) {
            throw std::invalid_argument(
                "from_blob rejects device storage owned by a different "
                "GPU than the immutable NSOS device selection");
        }
        source_host_accessible = attributes.type == cudaMemoryTypeManaged;
        source_gpu_device = attributes.device;
#else
        if (attributes.memoryType != cudaMemoryTypeDevice &&
            attributes.isManaged == 0) {
            throw std::invalid_argument(
                "from_blob Device::GPU rejects host-pointer storage");
        }
        int selected_device = -1;
        std::string selection_error;
        if (!gpu::select_preferred_device(
                &selected_device, &selection_error)) {
            throw std::runtime_error(
                "from_blob could not select a GPU: " +
                selection_error);
        }
        if (attributes.memoryType == cudaMemoryTypeDevice &&
            attributes.device != selected_device) {
            throw std::invalid_argument(
                "from_blob rejects device storage owned by a different "
                "GPU than the immutable NSOS device selection");
        }
        source_host_accessible = attributes.isManaged != 0;
        source_gpu_device = attributes.device;
#endif
#else
        throw std::runtime_error(
            "from_blob Device::GPU requires a CUDA-enabled build");
#endif
    }
    if (take_ownership) {
        t.data_ptr = std::shared_ptr<float>(
            static_cast<float*>(ptr),
            TensorDeleter(d, false, source_gpu_device));
        t.host_accessible_storage_ = source_host_accessible;
    } else {
        t = Tensor(s, d);
        if (t.size > 0) {
            copy_tensor_bytes(t.raw_data(), d, static_cast<const float*>(ptr), d,
                              static_cast<size_t>(t.size) * sizeof(float));
        }
    }
    return t;
}

} // namespace nsos
