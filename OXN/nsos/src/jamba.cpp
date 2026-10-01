#include "../include/jamba.h"
#include "../include/training_runtime_policy.h"
#include "../include/optimizer_runtime_policy.h"
#include "../include/gpu_backend.h"
#include "../include/sparse_attention.h"
#include "../include/layer_audit.h"
#include "../include/jamba_utils.h"
#include "../include/nsos_serializer.h"
#include "../include/nsos/determinism.h"
#include "../include/mcts_reasoning.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include "../include/cuda/sparse_attention_kernels.cuh"
#include "../include/cuda/device_buffer.h"
#include "../include/checkpoint_io.h"
#include "../include/nsos/sha256.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#endif

// ── Profiler hook globals (legacy — kept null, never used) ─────────────
// The profiler now attaches callbacks AS INSTANCE MEMBERS via
// JambaModel::set_profiler_callbacks().  The globals below are no
// longer read by the forward path but remain defined so any external
// build that links against nsos_core and references them does not
// break.  They are always null.
extern "C" {
    typedef void (*nsos_profiler_layer_event_fn)(void* profiler, int layer_idx);
    typedef void (*nsos_profiler_layer_end_fn)(void* profiler);
    nsos_profiler_layer_event_fn g_nsos_profiler_begin_layer = nullptr;
    nsos_profiler_layer_end_fn   g_nsos_profiler_end_layer   = nullptr;
}

namespace nsos {

#ifdef NSOS_ENABLE_TEST_HOOKS
namespace {
std::atomic<long long> model_checkpoint_stage_failure_countdown{-1};
}  // namespace

namespace testing {
void set_model_checkpoint_stage_failure_countdown(
    long long countdown) {
  model_checkpoint_stage_failure_countdown.store(
      countdown, std::memory_order_relaxed);
}

void clear_model_checkpoint_stage_failure() {
  model_checkpoint_stage_failure_countdown.store(
      -1, std::memory_order_relaxed);
}
}  // namespace testing

namespace detail {
void model_checkpoint_stage_fault_point() {
  long long remaining =
      model_checkpoint_stage_failure_countdown.load(
          std::memory_order_relaxed);
  while (remaining >= 0) {
    if (remaining == 0) {
      throw std::bad_alloc();
    }
    if (model_checkpoint_stage_failure_countdown
            .compare_exchange_weak(
                remaining, remaining - 1,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
      return;
    }
  }
}
}  // namespace detail
#endif

namespace {

constexpr int kStableMoEExperts = 8;
constexpr int kStableTopKExperts = 2;
constexpr uint32_t kEdgePackMagic = 0x31454744; // DGE1
constexpr uint32_t kEdgePackLegacyVersion = 2;
constexpr uint32_t kEdgePackVersion = 3;
constexpr uint32_t kEdgePackIntegrityMagic = 0x33494745; // EGI3
constexpr uint64_t kEdgePackIntegrityBytes = 4u + 8u + 64u;

#ifdef USE_CUDA
void gpu_runtime_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed on " + NSOS_GPU_BACKEND_NAME +
        ": " + cudaGetErrorString(status));
  }
}

// Persistent workspace for the batched MoE forward + backward GPU
// pipelines.  Replaces 6 GpuDeviceBuffer constructions / destructions
// per MoE call (12 per layer counting forward+backward) — each pair
// was costing ~1-5 ms in cudaMalloc/cudaFree plus an implicit stream
// sync.  With 4 MoE layers and 2 passes per step that was 48-240 ms
// per step of pure allocator churn on top of the actual compute.
//
// One static instance; the training loop is single-threaded so no
// concurrency guard needed.  Buffers grow geometrically (2×) when a
// larger N_active comes through and never shrink during execution. The
// thread-local destructor releases every allocation before context teardown.
class MoeWorkspace {
 public:
  MoeWorkspace() = default;
  MoeWorkspace(const MoeWorkspace&) = delete;
  MoeWorkspace& operator=(const MoeWorkspace&) = delete;

  // num_experts-sized buffers (counts, offsets+1, workspace_counters).
  // These rarely change shape (num_experts is a model constant), so the
  // allocation happens once and the buffers are reused forever.
  int* counts(int num_experts) {
    return ensure(counts_, num_experts, "expert counts");
  }
  int* offsets(int num_experts) {
    return ensure(offsets_, num_experts + 1, "expert offsets");
  }
  int* workspace_counters(int num_experts) {
    return ensure(work_, num_experts, "assignment counters");
  }
  // N_active-sized buffers.  N_active varies per batch (= sum of top-k
  // selections), so growth happens more often early in training, then
  // stabilizes once the buffers exceed typical batch maxima.
  int* permutation(int n) {
    return ensure(perm_, n, "permutation");
  }
  int* assignment(int n) {
    return ensure(assign_, n, "assignment");
  }
  float* scale(int n) {
    return ensure(scale_, n, "routing scales");
  }
  // Pre-filled buffer of 1.0f used by backward's scatter (where the
  // per-slot weighting was already applied earlier on the dy side).
  // We refill the vector after a grow but skip the upload when the
  // existing N_active still fits — avoids a redundant H2D copy per
  // backward call.
  float* unit_scale(int n) {
    float* const previous = unit_.get();
    float* const pointer =
        ensure(unit_, n, "unit routing scales");
    const bool grew =
        pointer != previous ||
        unit_filled_ < static_cast<int>(unit_.capacity());
    // Refill only if we grew OR caller wants more elements than we've
    // initialized so far (the buffer is overprovisioned so a single
    // upload covers many subsequent same-or-smaller N_active calls).
    if (grew || n > unit_filled_) {
      if (unit_.capacity() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "MoeWorkspace unit-scale capacity exceeds supported range");
      }
      const int capacity =
          static_cast<int>(unit_.capacity());
      std::vector<float> ones(
          static_cast<size_t>(capacity), 1.0f);
      gpu_runtime_check(
          cudaMemcpy(pointer, ones.data(),
                     static_cast<size_t>(capacity) * sizeof(float),
                     cudaMemcpyHostToDevice),
          "MoeWorkspace unit-scale upload");
      record_gpu_transfer(
          Device::GPU, Device::CPU,
          static_cast<size_t>(capacity) * sizeof(float));
      unit_filled_ = capacity;
    }
    return pointer;
  }

 private:
  template <typename T>
  static T* ensure(cuda_detail::DeviceBuffer<T>& buffer,
                   int requested,
                   const char* label) {
    if (requested <= 0) {
      throw std::invalid_argument(
          std::string("MoeWorkspace ") + label +
          " requires a positive element count");
    }
    T* pointer =
        buffer.ensure(static_cast<size_t>(requested));
    if (!pointer) {
      throw std::runtime_error(
          std::string("MoeWorkspace ") + label +
          " allocation failed");
    }
    return pointer;
  }

  cuda_detail::DeviceBuffer<int> counts_;
  cuda_detail::DeviceBuffer<int> offsets_;
  cuda_detail::DeviceBuffer<int> work_;
  cuda_detail::DeviceBuffer<int> perm_;
  cuda_detail::DeviceBuffer<int> assign_;
  cuda_detail::DeviceBuffer<float> scale_;
  cuda_detail::DeviceBuffer<float> unit_;
  int    unit_filled_  = 0;  // how many leading elements of unit_ are 1.0f
};
MoeWorkspace& moe_workspace() {
  // K6 (race): thread_local, not a shared static.  The MoE GPU forward/backward
  // run during inference too; the HTTP server drives concurrent decode replicas
  // on separate worker threads, and a shared workspace would let two threads
  // cudaFree/cudaMalloc/scatter into the same device buffers at once
  // (use-after-free / corrupted routing).  Per-thread instances make the GPU
  // MoE path replica-safe; single-threaded training is unaffected.  Mirrors
  // gemm_lowp_workspace()'s rationale (tensor.cpp).
  thread_local MoeWorkspace ws;
  return ws;
}

uint8_t* moe_valid_rows_scratch(size_t count) {
  if (count == 0) return nullptr;
  thread_local cuda_detail::DeviceBuffer<uint8_t> buffer;
  uint8_t* pointer = buffer.ensure(count);
  if (!pointer) {
    throw std::runtime_error(
        "MoE valid-row staging allocation failed");
  }
  return pointer;
}

int* moe_valid_indices_scratch(size_t count) {
  if (count == 0) return nullptr;
  thread_local cuda_detail::DeviceBuffer<int> buffer;
  int* pointer = buffer.ensure(count);
  if (!pointer) {
    throw std::runtime_error(
        "MoE valid-index staging allocation failed");
  }
  return pointer;
}
#endif

// Audit timing must measure device execution, not just asynchronous launch
// latency. Events are created only while auditing is enabled, so the production
// hot path pays neither synchronization nor event-management overhead.
class AuditLatencyTimer {
public:
    AuditLatencyTimer(bool enabled, Device device)
        : enabled_(enabled),
          cpu_started_(enabled ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{}) {
#ifdef USE_CUDA
        gpu_timing_ = enabled && device == Device::GPU;
        if (gpu_timing_) {
            gpu_runtime_check(cudaEventCreate(&gpu_started_),
                              "audit start-event creation");
            try {
                gpu_runtime_check(cudaEventCreate(&gpu_finished_),
                                  "audit finish-event creation");
                gpu_runtime_check(cudaEventRecord(gpu_started_, nsos::gpu::current_stream()),
                                  "audit start-event record");
            } catch (...) {
                release_events();
                throw;
            }
        }
#else
        (void)device;
#endif
    }

    AuditLatencyTimer(const AuditLatencyTimer&) = delete;
    AuditLatencyTimer& operator=(const AuditLatencyTimer&) = delete;

    ~AuditLatencyTimer() {
#ifdef USE_CUDA
        release_events();
#endif
    }

    double elapsed_ms() {
        if (!enabled_ || stopped_) {
            return elapsed_ms_;
        }
#ifdef USE_CUDA
        if (gpu_timing_) {
            gpu_runtime_check(cudaEventRecord(gpu_finished_, nsos::gpu::current_stream()),
                              "audit finish-event record");
            gpu_runtime_check(cudaEventSynchronize(gpu_finished_),
                              "audit finish-event synchronize");
            record_gpu_stream_synchronization();
            float value_ms = 0.0f;
            gpu_runtime_check(
                cudaEventElapsedTime(&value_ms, gpu_started_, gpu_finished_),
                "audit elapsed-time query");
            elapsed_ms_ = static_cast<double>(value_ms);
            stopped_ = true;
            release_events();
            return elapsed_ms_;
        }
#endif
        elapsed_ms_ =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpu_started_)
                .count();
        stopped_ = true;
        return elapsed_ms_;
    }

private:
#ifdef USE_CUDA
    void release_events() noexcept {
        if (gpu_started_ != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(gpu_started_),
                "audit start-event destruction");
            gpu_started_ = nullptr;
        }
        if (gpu_finished_ != nullptr) {
            gpu::report_cleanup_status(
                cudaEventDestroy(gpu_finished_),
                "audit finish-event destruction");
            gpu_finished_ = nullptr;
        }
    }
#endif

    bool enabled_ = false;
    bool stopped_ = false;
    double elapsed_ms_ = 0.0;
    std::chrono::steady_clock::time_point cpu_started_;
#ifdef USE_CUDA
    bool gpu_timing_ = false;
    cudaEvent_t gpu_started_ = nullptr;
    cudaEvent_t gpu_finished_ = nullptr;
#endif
};

uint64_t hash_token_sequence(const std::vector<int>& tokens) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (int token : tokens) {
        uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<unsigned char>((value >> shift) & 0xFFu);
            hash *= kPrime;
        }
    }
    return hash;
}

// (auditoria #8) copy_moe_bytes removida — copy_tensor_bytes unificada (tensor.h).

void zero_sequence_suffix_inplace(Tensor& tensor, const std::vector<int>& lengths) {
    if (tensor.shape.size() != 3 || lengths.empty()) {
        return;
    }
    const int batch_size = tensor.shape[0];
    const int seq_len = tensor.shape[1];
    const int dim = tensor.shape[2];

#ifdef USE_CUDA
    // GPU path: cudaMemsetAsync per padded row.  Queued on the default
    // stream and returns immediately — no cudaDeviceSynchronize, no
    // host pointer access, no D2H staging.  Previously this function
    // called sync_host_access() (= cudaDeviceSynchronize on first call)
    // 14 times per forward pass (1 + 12 layers + 1 final), costing
    // ~150-300 ms/step on Colab T4 even though the actual zeroing work
    // is microseconds.  cudaMemsetAsync overlaps with whatever kernel
    // launches next on the same stream, costing effectively zero.
    if (tensor.get_device() == Device::GPU) {
        float* base = tensor.raw_data();
        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid = std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
            const int padding_tokens = seq_len - valid;
            if (padding_tokens <= 0) continue;
            float* start = base + (static_cast<size_t>(batch) * seq_len + valid) *
                                  static_cast<size_t>(dim);
            const size_t bytes = static_cast<size_t>(padding_tokens) *
                                 static_cast<size_t>(dim) * sizeof(float);
            gpu_runtime_check(cudaMemsetAsync(start, 0, bytes, nsos::gpu::current_stream()),
                              "zero padded sequence suffix");
        }
        return;
    }
#endif

    // CPU path (unchanged) — still needs the sync barrier on Pascal+
    // Windows UM, but it's a no-op on host-resident tensors.
    tensor.sync_host_access();
    float* ptr = tensor.data();
    for (int batch = 0; batch < batch_size; ++batch) {
        const int valid = std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
        for (int token = valid; token < seq_len; ++token) {
            std::fill_n(ptr + ((batch * seq_len + token) * dim), dim, 0.0f);
        }
    }
}

std::vector<uint8_t> flattened_valid_row_mask(
    const Tensor& tensor, const std::vector<int>& lengths) {
    if (tensor.shape.size() != 3 || lengths.empty()) {
        return {};
    }
    const int batch_size = tensor.shape[0];
    const int seq_len = tensor.shape[1];
    if (static_cast<int>(lengths.size()) != batch_size) {
        throw std::invalid_argument(
            "batched valid-length count does not match tensor batch dimension");
    }
    std::vector<uint8_t> mask(
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len), 0U);
    for (int batch = 0; batch < batch_size; ++batch) {
        const int valid =
            std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
        std::fill_n(mask.begin() + static_cast<size_t>(batch) * seq_len,
                    valid, static_cast<uint8_t>(1U));
    }
    return mask;
}

Tensor apply_training_dropout(const Tensor& input,
                              float dropout_rate,
                              const std::string& scope,
                              int salt,
                              uint64_t forward_sequence,
                              Tensor* out_mask = nullptr) {
    const float rate = std::clamp(dropout_rate, 0.0f, 0.95f);
    if (rate <= 1e-6f || input.size == 0) {
        if (out_mask) *out_mask = Tensor();  // no dropout -> no mask
        return input;
    }
    Tensor mask_host(input.shape.dims, Device::CPU);
    auto rng =
        determinism::DeterminismManager::instance().get_rng_for_operation(
            "jamba_dropout", scope,
            forward_sequence ^
                (static_cast<uint64_t>(static_cast<uint32_t>(std::max(salt, 0)))
                 << 32U) ^
                static_cast<uint64_t>(input.shape.numel()));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float keep_scale = 1.0f / std::max(1.0f - rate, 1e-6f);
    float* mask_ptr = mask_host.data();
    for (int i = 0; i < mask_host.size; ++i) {
        mask_ptr[i] = dist(rng) >= rate ? keep_scale : 0.0f;
    }
    Tensor mask = input.get_device() == Device::GPU ? mask_host.to(Device::GPU) : mask_host;
    if (out_mask) *out_mask = mask;  // carried to backward for a correct STE
    return input.mul(mask);
}


void copy_float_bytes_device_safe(float* dst,
                                  Device dst_device,
                                  const float* src,
                                  Device src_device,
                                  size_t bytes) {
    if (bytes == 0 || dst == src) {
        return;
    }
#ifdef USE_CUDA
    if (dst_device == Device::GPU || src_device == Device::GPU) {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        if (dst_device == Device::GPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (dst_device == Device::GPU && src_device == Device::CPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (dst_device == Device::CPU && src_device == Device::GPU) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }
        cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
        }
        record_gpu_transfer(dst_device, src_device, bytes);
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
    record_gpu_transfer(dst_device, src_device, bytes);
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed to write edge pack payload");
    }
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Failed to read edge pack payload");
    }
    return value;
}

template <typename T>
void write_vector(std::ofstream& out, const std::vector<T>& values) {
    const uint64_t count = static_cast<uint64_t>(values.size());
    write_pod(out, count);
    if (count == 0) {
        return;
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) {
        throw std::runtime_error("Failed to write edge pack vector");
    }
}

template <typename T>
std::vector<T> read_vector(std::ifstream& in, uint64_t max_count, const char* label) {
    const uint64_t count = read_pod<uint64_t>(in);
    if (count > max_count ||
        count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        throw std::runtime_error(std::string("Edge pack vector exceeds limit: ") + label);
    }
    std::vector<T> values(static_cast<size_t>(count));
    if (count == 0) {
        return values;
    }
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error("Failed to read edge pack vector");
    }
    return values;
}

void validate_edge_packed_state(const BitLinearPackedState& state) {
    if (state.in_features <= 0 || state.out_features <= 0) {
        throw std::runtime_error(
            "Edge pack contains non-positive linear dimensions");
    }
    const uint64_t total_weights =
        static_cast<uint64_t>(state.in_features) *
        static_cast<uint64_t>(state.out_features);
    const uint64_t required_words = (total_weights + 15u) / 16u;
    const uint64_t required_out =
        static_cast<uint64_t>(state.out_features);
    const uint64_t required_in =
        static_cast<uint64_t>(state.in_features);
    const auto require_exact = [](size_t got, uint64_t need,
                                  const char* label) {
        if (static_cast<uint64_t>(got) != need) {
            throw std::runtime_error(
                std::string("Edge pack ") + label +
                " length does not match linear metadata");
        }
    };
    require_exact(state.packed_weights.size(), required_words,
                  "packed_weights");
    require_exact(state.magnitude.size(), required_out, "magnitude");
    require_exact(state.bias.size(), state.use_bias ? required_out : 0,
                  "bias");
    require_exact(state.flat_alpha.size(), required_in, "flat_alpha");
    require_exact(state.flat_beta.size(), required_in, "flat_beta");
    if (!std::isfinite(state.weight_scale) ||
        state.weight_scale <= 0.0f) {
        throw std::runtime_error(
            "Edge pack contains an invalid weight scale");
    }
    const auto require_finite = [](const std::vector<float>& values,
                                   const char* label) {
        if (!std::all_of(
                values.begin(), values.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                std::string("Edge pack contains non-finite ") + label);
        }
    };
    require_finite(state.magnitude, "magnitude");
    require_finite(state.bias, "bias");
    require_finite(state.flat_alpha, "flat_alpha");
    require_finite(state.flat_beta, "flat_beta");
    for (uint64_t index = 0; index < total_weights; ++index) {
        const uint32_t code =
            (state.packed_weights[static_cast<size_t>(index / 16u)] >>
             (2u * static_cast<unsigned int>(index % 16u))) &
            0x3u;
        if (code == 0x3u) {
            throw std::runtime_error(
                "Edge pack contains a reserved ternary weight code");
        }
    }
}

} // namespace

JambaModel::JambaModel(int nl, int dm, int vs, Device dev)
    : JambaModel([&] {
          ModelConfig cfg;
          cfg.num_layers = nl;
          cfg.d_model = dm;
          cfg.vocab_size = vs;
          cfg.n_heads = sanitize_head_count(dm, cfg.n_heads);
          while (cfg.n_heads > 1 &&
                 (dm % cfg.n_heads != 0 || ((dm / cfg.n_heads) & 1) != 0)) {
              --cfg.n_heads;
          }
          cfg.n_kv_heads = select_kv_heads(cfg.n_heads, cfg.n_kv_heads);
          const int inner = cfg.mamba_expand * dm;
          cfg.mamba_head_dim = std::min(cfg.mamba_head_dim, inner);
          while (cfg.mamba_head_dim > 1 && inner % cfg.mamba_head_dim != 0) {
              --cfg.mamba_head_dim;
          }
          const int mamba_heads = inner / cfg.mamba_head_dim;
          cfg.mamba_n_groups = std::min(cfg.mamba_n_groups, mamba_heads);
          cfg.use_cuda = (dev == Device::GPU);
          return cfg;
      }(),
      dev) {}

JambaModel::JambaModel(const ModelConfig& config, Device dev)
    : num_layers(config.num_layers),
      d_model(config.d_model),
      vocab_size(config.vocab_size),
      device(dev),
      model_config_(config) {
    validate_model_config(config);
    if (dev == Device::GPU &&
        std::getenv("NSOS_GPU_ALLOW_HOST_FALLBACK") == nullptr) {
        set_strict_gpu_execution(true);
    }
    model_config_.num_layers = num_layers;
    model_config_.d_model = d_model;
    model_config_.vocab_size = vocab_size;
    model_config_.use_cuda = (dev == Device::GPU);
    const bool use_exact_attention_training = model_config_.use_exact_attention_training;

    embedding = std::make_unique<Embedding>(vocab_size, d_model);
    if ((model_config_.mamba2_faithful || model_config_.mamba3_enabled)) {
        // MambaLMHeadModel initializes token embeddings from N(0, 0.02).
        embedding->weight.data =
            Tensor::random({vocab_size, d_model}, Device::CPU);
        embedding->weight.mark_updated();
    }
    if ((model_config_.mamba2_faithful || model_config_.mamba3_enabled)) {
        final_norm_weight_ = Parameter(
            Tensor::ones({d_model}, Device::CPU), "norm_f.weight");
    }
    // Slender head-to-toe quantization (opt-in, 2026-05-25 wiring).
    // When true, the Embedding uses ternary-quantized lookup via
    // slender_forward_cpu_ instead of the dense float path.
    if (model_config_.use_slender_embedding) {
        embedding->set_slender_quantization(true);
    }

    for (int i = 0; i < num_layers; ++i) {
        const int layer_one_based = i + 1;
        bool use_attention =
            layer_matches_schedule(layer_one_based, model_config_.attention_period,
                                   model_config_.attention_slot);
        // GUARD (last-layer-Mamba): empirically isolated (attn audit, 3 rounds)
        // — in the faithful config, an attention layer as the FINAL layer of the
        // stack STALLS training (loss stuck at ~chance), while a Mamba final
        // layer learns; the trigger is the last-layer TYPE, not the attention
        // count, and non-faithful is immune. Force the last layer to be Mamba.
        if (use_attention && (model_config_.mamba2_faithful || model_config_.mamba3_enabled) &&
            model_config_.force_mamba_last_layer &&
            layer_one_based == num_layers) {
            use_attention = false;
        }
        const bool use_moe =
            model_config_.use_moe &&
            layer_matches_schedule(layer_one_based, model_config_.moe_period,
                                   model_config_.moe_slot);
        const bool use_ttt =
            model_config_.use_ttt &&
            layer_matches_schedule(layer_one_based, model_config_.ttt_period,
                                   model_config_.ttt_slot);
        layers.push_back(std::make_unique<JambaBlock>(
            d_model, use_attention, use_moe, use_ttt, i, num_layers,
            model_config_.n_heads, model_config_.n_kv_heads, model_config_.num_experts,
            model_config_.num_experts_per_token,
            use_exact_attention_training,
            model_config_.dropout,
            model_config_.use_gradient_checkpointing,
            // Nemotron K·m invariant (Cherry-pick #4).  0 = default dm*4.
            model_config_.moe_expert_hidden_dim,
            // CHRASS topological injection (2026-05-25).  off by default.
            model_config_.use_chrass,
            model_config_.chrass_density,
            model_config_.chrass_seed,
            // KAN FFN (non-MoE blocks).  off by default.
            model_config_.use_kan,
            // K1: corrected Mamba-2 SSD path, default ON via ModelConfig.
            model_config_.mamba_proper_ssm,
            model_config_.mamba_state_expansion,
            model_config_.mamba_d_state,
            model_config_.mamba_conv_kernel,
            model_config_.mamba2_faithful,
            model_config_.mamba_expand,
            model_config_.mamba_head_dim,
            model_config_.mamba_n_groups,
            model_config_.rope_theta,
            model_config_.sliding_window,
            model_config_.max_context_tokens,
            model_config_.hybrid_composition,
            model_config_.faithful_attention_linears,
            model_config_.hybrid_mamba_gate_init,
            model_config_.hybrid_attention_gate_init,
            model_config_.hybrid_ffn_gate_init,
            model_config_.mamba3_enabled,
            [&] {
                Mamba3Config c;
                c.schema_version=model_config_.mamba3_schema_version;
                c.expand=model_config_.mamba_expand;c.head_dim=model_config_.mamba_head_dim;
                c.n_groups=model_config_.mamba_n_groups;c.state_dim=model_config_.mamba3_state_dim;
                c.mimo=model_config_.mamba3_mimo;c.mimo_rank=model_config_.mamba3_mimo_rank;
                c.outproj_norm=model_config_.mamba3_outproj_norm;c.rope_fraction=model_config_.mamba3_rope_fraction;
                c.norm_eps=model_config_.mamba3_norm_eps;c.a_floor=model_config_.mamba3_a_floor;
                c.seed=1+static_cast<std::uint64_t>(i);return c;
            }()));
    }

    value_head = std::make_unique<BitLinear>(
        d_model, vocab_size, !(model_config_.mamba2_faithful || model_config_.mamba3_enabled));
    if ((model_config_.mamba2_faithful || model_config_.mamba3_enabled)) {
        value_head->set_exact_linear_mode(true);
        const float bound =
            1.0f / std::sqrt(static_cast<float>(std::max(d_model, 1)));
        value_head->weight.data =
            Tensor::uniform({vocab_size, d_model}, -bound, bound,
                            Device::CPU);
        value_head->weight.mark_updated();
    }
    tie_word_embeddings_ = model_config_.tie_word_embeddings;  // N6
    to(dev);  // to() applies weight tying at the end when enabled
    initialize_parameter_registry_();
}

// N6: re-point value_head's weight storage at the embedding's weight buffer so
// the LM head and the token embedding share one matrix.  Idempotent; safe to
// call after any device move or deserialize (both reallocate the buffers).  Only
// ties when the shapes match exactly ([vocab, d_model]); otherwise leaves the
// head independent.
void JambaModel::apply_weight_tying_() {
    if (!tie_word_embeddings_ || !embedding || !value_head) {
        weight_tied_ = false;
        tied_embedding_version_observed_ = 0;
        return;
    }
    Tensor& emb = embedding->weight.data;
    Tensor& head = value_head->weight.data;
    if (emb.size == 0 || head.size == 0 || emb.shape != head.shape) {
        weight_tied_ = false;
        tied_embedding_version_observed_ = 0;
        return;
    }
    if (emb.get_device() != head.get_device()) {
        // Bring the head's (about-to-be-discarded) buffer to the embedding device
        // so the shared buffer is consistent; the data is overwritten by the
        // share anyway.
        head = head.to(emb.get_device());
    }
    // Share the underlying storage: value_head now reads/writes the embedding's
    // matrix.  value_head keeps its own magnitude/bias (independent output
    // affine), so only the [vocab, d_model] matrix is tied.
    value_head->weight.data = embedding->weight.data;  // shared_ptr buffer share
    // Shared storage has one canonical content version. This is consumed by
    // the low-precision weight cache as well as legacy BitLinear caches.
    value_head->weight.version = embedding->weight.version;
    tied_embedding_version_observed_ = embedding->weight.version;
    weight_tied_ = true;
}

void JambaModel::refresh_weight_tying_cache_version_() {
    if (!weight_tied_ || !embedding || !value_head) {
        return;
    }
    if (embedding->weight.version !=
        tied_embedding_version_observed_) {
        // The two Parameters intentionally share storage but retain distinct
        // metadata. An optimizer update is published through
        // embedding.weight, so explicitly invalidate every BitLinear cache
        // owned by the aliased value-head Parameter before its next use.
        value_head->weight.version = embedding->weight.version;
        tied_embedding_version_observed_ =
            embedding->weight.version;
    }
}

void JambaModel::save(const std::string& filename) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    const std::filesystem::path destination(filename);
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    const std::filesystem::path temporary =
        checkpoint_io::unique_temporary_path(destination);
    try {
        ModelSerializer::save(this, temporary.string());
        checkpoint_io::flush_file(temporary);
        checkpoint_io::atomic_replace(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

ModelConfig JambaModel::save_model_pack_artifacts(
    const std::string& weights_filename,
    const std::string& edge_linear_filename) {
    std::lock_guard<std::recursive_mutex> execution_lock(
        execution_mutex_);
    save(weights_filename);
    save_edge_linear_pack(edge_linear_filename);
    return model_config_;
}

void JambaModel::load(const std::string& filename, bool strict) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("model load");
    ModelSerializer::load(this, filename, strict);
    for(auto& layer:layers) if(layer && layer->mamba3_layer) layer->mamba3_layer->reset();
    apply_weight_tying_();  // N6: re-share the buffer after deserialize realloc
}

Tensor JambaModel::forward(const Tensor& x, Context* ctx) {
    return forward_impl(x, ctx, false);
}

Tensor JambaModel::forward_impl(const Tensor& x, Context* ctx, bool last_logits_only, bool headless) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    // A new model forward invalidates the previous model backward ticket.
    // Release only its Mamba3 pending tapes; owning external tapes are distinct.
    for(auto& layer:layers) if(layer && layer->mamba3_layer) layer->mamba3_layer->cancel_pending();
    if (decode_graph_active()) {
        decode_graph_.reset();
        decode_graph_status_ = "idle: eager forward invalidated graph";
    }
    if (last_logits_only && training_mode_) {
        throw std::logic_error("Last-token projection is inference-only");
    }
    // A new attempt invalidates every older backward ticket even if this
    // forward later aborts or throws.
    invalidate_backward_state_();
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    if (x.size <= 0 || (x.shape.size() != 2 && x.shape.size() != 3) ||
        x.shape.back() != d_model) {
        throw std::invalid_argument(
            "JambaModel::forward expects a non-empty [seq,d_model] or "
            "[batch,seq,d_model] tensor");
    }
    Tensor hidden = x;
    const uint64_t stochastic_sequence =
        training_mode_ ? training_rng_sequence_++ : 0;
    if (hidden.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    }
    // ── OPT-IN profiler hook ───────────────────────────────────────────
    // The callbacks are stored as INSTANCE MEMBERS of JambaModel
    // (profiler_begin_layer_ / profiler_end_layer_), not as DLL
    // globals.  This is intentional: in a Windows two-pyd setup
    // (nsos_ext.pyd holds the model code, nsos_profiler_ext.pyd
    // provides the callbacks), each pyd gets its OWN copy of any
    // global initialized by a static library, so a global setter
    // pattern reads the WRONG copy on the model side.  Function
    // pointers stored on the instance bypass that — they're plain
    // address values that the model can call regardless of which
    // pyd defined them.
    //
    // Production cost when no profiler is attached: one pointer load
    // (profiler_begin_layer_) and one branch (predicted not-taken
    // after first iteration).  Effectively free.
    const auto profiler_begin_layer = profiler_begin_layer_;
    const auto profiler_end_layer   = profiler_end_layer_;
    // Diagnóstico NSOS_LAYER_TIMING=1: tempo por camada (sync em volta de cada
    // layer->forward) impresso uma linha por forward — localiza ONDE o forward
    // gasta (mamba vs attn vs moe por índice).  OFF por default (custo zero).
    static const bool nsos_layer_timing = [] {
        const char* e = std::getenv("NSOS_LAYER_TIMING");
        return e != nullptr && e[0] == '1';
    }();
    std::vector<double> layer_ms;
    if (nsos_layer_timing) {
        layer_ms.assign(layers.size(), 0.0);
    }
    int layer_index = 0;
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        layer->set_batch_valid_lengths(last_input_batch_lengths_);
        layer->set_dropout_sequence(stochastic_sequence);
        if (profiler_begin_layer) {
            profiler_begin_layer(profiler_, layer_index);
        }
        AuditLatencyTimer layer_timing_timer(
            nsos_layer_timing, hidden.get_device());
        hidden = layer->forward(hidden, ctx);
        {
            static const bool attn_dbg = []() {
                const char* e = std::getenv("NSOS_ATTN_DEBUG");
                return e != nullptr && e[0] == '1';
            }();
            if (attn_dbg) {
                std::fprintf(stderr, "[fwd L%d %-14s] out_absmean=%.5g\n",
                             layer_index, layer->audit_block_type().c_str(),
                             tensor_abs_mean(hidden));
            }
        }
        if (nsos_layer_timing) {
            layer_ms[static_cast<size_t>(layer_index)] +=
                layer_timing_timer.elapsed_ms();
        }
        if (profiler_end_layer) {
            profiler_end_layer(profiler_);
        }
        if (hidden.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
            zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
        }
        ++layer_index;
    }
    saved_final_hidden_ = hidden;
    const float final_norm_eps =
        (model_config_.mamba2_faithful || model_config_.mamba3_enabled) ? 1e-5f : 1e-6f;
    saved_final_rms_ = hidden.rmsnorm(final_norm_eps);
    saved_final_norm_ =
        (model_config_.mamba2_faithful || model_config_.mamba3_enabled)
            ? saved_final_rms_.mul(final_norm_weight_.data)
            : saved_final_rms_;
    if (saved_final_norm_.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(saved_final_rms_, last_input_batch_lengths_);
        zero_sequence_suffix_inplace(saved_final_norm_, last_input_batch_lengths_);
    }
    if (nsos_layer_timing) {
        std::cerr << "[ltime]";
        for (size_t i = 0; i < layer_ms.size(); ++i) {
            std::cerr << " L" << i << "=" << static_cast<int>(layer_ms[i]);
        }
        std::cerr << "ms" << std::endl;
    }
    const bool audit_head =
        audit_collector_ && audit_collector_->enabled();
    AuditLatencyTimer head_timer(audit_head, saved_final_norm_.get_device());
    refresh_weight_tying_cache_version_();
    Tensor head_input = saved_final_norm_;
    if (headless) {
        if (!training_mode_) throw std::logic_error("Headless forward requires training mode");
        grant_backward_state_();
        headless_backward_ = true;
        tiled_head_ready_ = true;
        return saved_final_norm_;
    }
    if (last_logits_only && head_input.shape.size() == 2 && head_input.shape[0] > 1) {
        const int last = head_input.shape[0] - 1;
        head_input = head_input.slice(0, last, last + 1);
    }
    Tensor logits = value_head->forward(head_input);
    if (logits.shape.size() == 3 && !last_input_batch_lengths_.empty()) {
        zero_sequence_suffix_inplace(logits, last_input_batch_lengths_);
    }
    if (audit_head) {
        const double latency_ms = head_timer.elapsed_ms();
        audit_collector_->record_forward(-1, "value_head", "logits", head_input,
                                         logits, latency_ms);
    }
    grant_backward_state_();
    return logits;
}

Tensor JambaModel::forward(const Tensor& x) {
    return forward(x, nullptr);
}

void JambaModel::to(Device dev) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    if (dev == Device::GPU &&
        std::getenv("NSOS_GPU_ALLOW_HOST_FALLBACK") == nullptr) {
        // A CPU-constructed model moved later must receive the same native
        // device-memory and fail-closed execution contract as a model that was
        // constructed directly on the GPU.
        set_strict_gpu_execution(true);
    }
    device = dev;
    model_config_.use_cuda = (dev == Device::GPU);
    // Device moves reallocate every buffer a captured decode graph points at.
    if (decode_graph_) {
        decode_graph_.reset();
        if (!decode_graph_disabled_) decode_graph_status_ = "idle";
    }
    embedding->to(dev);
    for (auto& layer : layers) {
        layer->to(dev);
    }
    if (final_norm_weight_.data.size > 0) {
        final_norm_weight_.data = final_norm_weight_.data.to(dev);
    }
    value_head->to(dev);
    apply_weight_tying_();  // N6: re-share the buffer after the device move
}

void JambaModel::initialize_parameter_registry_() {
    if (parameter_registry_initialized_) {
        return;
    }

    // This is the sole naming traversal for a model instance. Nested modules
    // compose relative names during construction; the resulting absolute
    // identities are frozen before the constructor publishes the model.
    bump_parameter_name_epoch();
    std::vector<Parameter*> params;
    auto embedding_params = embedding->parameters();
    prefix_parameter_names(embedding_params, "embedding.");
    params.insert(params.end(), embedding_params.begin(), embedding_params.end());
    for (size_t index = 0; index < layers.size(); ++index) {
        auto& layer = layers[index];
        auto layer_params = layer->parameters();
        prefix_parameter_names(layer_params, "layers." + std::to_string(index) + ".");
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    if (final_norm_weight_.data.size > 0) {
        final_norm_weight_.assign_relative_name("norm_f.weight");
        params.push_back(&final_norm_weight_);
    }
    auto head_params = value_head->parameters();
    prefix_parameter_names(head_params, "value_head.");
    for (Parameter* hp : head_params) {
        // N6: when tied, the shared [vocab, d_model] matrix is owned/optimized as
        // embedding.weight — skip value_head.weight so the optimizer steps it
        // exactly once (its gradient is folded into embedding.weight in
        // backward()).  value_head's magnitude/bias stay independent.
        if (weight_tied_ && hp == &value_head->weight) {
            continue;
        }
        params.push_back(hp);
    }
    std::unordered_set<Parameter*> unique_parameters;
    std::unordered_set<std::string> unique_names;
    unique_parameters.reserve(params.size() * 2);
    unique_names.reserve(params.size() * 2);
    for (Parameter* parameter : params) {
        if (!parameter) {
            throw std::logic_error(
                "model parameter registry contains a null entry");
        }
        if (!unique_parameters.insert(parameter).second) {
            throw std::logic_error(
                "model parameter registry contains a duplicate pointer");
        }
        if (parameter->name.empty() ||
            !unique_names.insert(parameter->name).second) {
            throw std::logic_error(
                "model parameter registry contains a missing or duplicate "
                "absolute name: '" + parameter->name + "'");
        }
        parameter->freeze_name();
    }
    parameter_registry_ = std::move(params);
    parameter_registry_names_.reserve(parameter_registry_.size());
    for (const Parameter* parameter : parameter_registry_) {
        parameter_registry_names_.push_back(parameter->name);
    }
    parameter_registry_initialized_ = true;
}

std::vector<Parameter*> JambaModel::parameters() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    if (!parameter_registry_initialized_) {
        initialize_parameter_registry_();
    }
    if (parameter_registry_.size() != parameter_registry_names_.size()) {
        throw std::logic_error("model parameter registry metadata is inconsistent");
    }
    for (size_t index = 0; index < parameter_registry_.size(); ++index) {
        const Parameter* parameter = parameter_registry_[index];
        if (!parameter || !parameter->name_frozen() ||
            parameter->name != parameter_registry_names_[index]) {
            throw std::logic_error(
                "model parameter canonical identity was mutated at registry "
                "index " + std::to_string(index));
        }
    }
    return parameter_registry_;
}

std::vector<Parameter*> JambaModel::no_weight_decay_parameters() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    const auto registry = parameters();
    std::vector<Parameter*> excluded;
    for (const auto& layer : layers) {
        if (!layer->mamba3_layer) continue;
        for (auto* parameter : layer->mamba3_layer->no_weight_decay_parameters()) {
            if (std::find(registry.begin(), registry.end(), parameter) == registry.end())
                throw std::logic_error("Mamba3 optimizer exclusion absent from canonical registry");
            if (std::find(excluded.begin(), excluded.end(), parameter) == excluded.end())
                excluded.push_back(parameter);
        }
    }
    return excluded;
}

std::vector<std::pair<std::string, std::string>>
JambaModel::parameter_aliases() const {
    std::lock_guard<std::recursive_mutex> execution_lock(
        execution_mutex_);
    std::vector<std::pair<std::string, std::string>> aliases;
    if (weight_tied_) {
        if (!embedding || !value_head ||
            embedding->weight.data.shape !=
                value_head->weight.data.shape ||
            embedding->weight.data.raw_data() !=
                value_head->weight.data.raw_data()) {
            throw std::logic_error(
                "weight-tying metadata does not match tensor storage");
        }
        aliases.emplace_back(
            "value_head.weight", "embedding.weight");
    }
    return aliases;
}

std::vector<BitLinear*> JambaModel::collect_bitlinear_layers() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    refresh_weight_tying_cache_version_();
    std::vector<BitLinear*> layers_out;
    if (value_head) {
        layers_out.push_back(value_head.get());
    }
    for (auto& layer : layers) {
        layer->collect_bitlinear_layers(layers_out);
    }
    return layers_out;
}

void JambaModel::invalidate_decode_graph_(const char* reason) {
    if (!decode_graph_) return;
    decode_graph_.reset();
    if (!decode_graph_disabled_) decode_graph_status_ = std::string("idle: ") + reason;
}

void JambaModel::set_reference_path(bool use_reference_path) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("reference path changed");
    invalidate_backward_state_();
    for (BitLinear* layer : collect_bitlinear_layers()) {
        if (!layer) {
            continue;
        }
        // Sensitive projections always stay on the float reference path; never
        // route them through the ternary kernel.
        if (!use_reference_path && layer->quantization_sensitive()) {
            layer->set_reference_path(true);
            continue;
        }
        layer->set_reference_path(use_reference_path);
    }
}

void JambaModel::set_sparse_attention(bool enabled, int block_size,
                                      int top_k_blocks, int local_blocks,
                                      int sink_blocks) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("attention policy changed");
    invalidate_backward_state_();
    for (auto& block : layers) {
        if (block && block->attn_layer) {
            block->attn_layer->set_sparse_attention(
                enabled, block_size, top_k_blocks, local_blocks, sink_blocks);
        }
    }
}

float JambaModel::accumulate_sparse_selector_grads(float gradient_weight) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    if (!backward_ready_ || !sparse_selector_backward_ready_) {
        throw std::runtime_error(
            "Sparse-selector gradients require a fresh successful training "
            "forward");
    }
    // Consume before the first selector-gradient mutation. A failure requires
    // a new forward and cannot be retried against partially accumulated data.
    sparse_selector_backward_ready_ = false;
    float total = 0.0f;
    for (auto& block : layers) {
        if (block && block->attn_layer) {
            total += block->attn_layer->accumulate_selector_distill_grad(
                gradient_weight);
        }
    }
    return total;
}

void JambaModel::set_gpu_packed_inference(bool enabled) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("packed inference policy changed");
    invalidate_backward_state_();
    // Toggle the BitLinear GPU __dp4a fast path on every linear layer
    // (attention QKV/O, mamba projections, FFN/MoE experts).  Used by
    // the inference engine after model load + repack to opt the entire
    // model into the dp4a path with one call.  Training callers should
    // leave this disabled — gradients flow through the float matmul
    // path only.
    for (BitLinear* layer : collect_bitlinear_layers()) {
        if (layer) {
            layer->set_gpu_packed_inference(enabled);
        }
    }
}

void Attention::reserve_kv_cache(int total_tokens, Device device, int batch_size) {
    // Public wrapper around the private ensure_kv_cache_capacity.  This
    // is the inference-side hook that lets the SDK pre-allocate the
    // cache to its known maximum before the decode loop runs, so the
    // per-token forward never triggers the page-growth path.
    //
    // We round up to a multiple of cache_page_tokens_ so the resulting
    // capacity is aligned with the page-growth strategy — if the user
    // generates a few more than they asked for (e.g. continuing past
    // max_tokens because EOS didn't fire), we still avoid one realloc.
    if (total_tokens <= 0) {
        return;
    }
    const int pages = gpu::ceil_div_positive(
        total_tokens, cache_page_tokens_);
    if (pages > std::numeric_limits<int>::max() / cache_page_tokens_) {
        throw std::overflow_error(
            "KV-cache reservation exceeds the representable token capacity");
    }
    const int aligned = pages * cache_page_tokens_;
    if (compact_kv_ && device == Device::GPU && compact_kv_->batch() == std::max(batch_size, 1)) {
        ensure_compact_kv_capacity(aligned, std::max(batch_size, 1));
        return;
    }
    ensure_kv_cache_capacity(aligned, device, std::max(batch_size, 1));
}

void Attention::make_kv_cache_unique() {
    if (compact_kv_ && compact_kv_.use_count() > 1)
        compact_kv_ = compact_kv_->resize(cache_capacity_tokens_, cached_tokens_);
    // clone() deep-copies into a fresh buffer (use_count 1), same shape so
    // cache_capacity_tokens_ is unchanged.  Only fires when a snapshot is
    // aliasing the buffer; a no-op otherwise.  Must run OUTSIDE any capture.
    if (key_cache_buffer_.size > 0 &&
        key_cache_buffer_.data_ptr.use_count() > 1) {
        key_cache_buffer_ = key_cache_buffer_.clone();
    }
    if (value_cache_buffer_.size > 0 &&
        value_cache_buffer_.data_ptr.use_count() > 1) {
        value_cache_buffer_ = value_cache_buffer_.clone();
    }
}

void JambaModel::reserve_kv_cache(int total_tokens, Device device, int batch_size) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("KV reservation changed");
    invalidate_backward_state_();
    // Iterate every JambaBlock that has an attention layer and reserve
    // its KV cache.  Mamba-only blocks have no KV cache to reserve.
    // TTT blocks similarly don't need this.  Safe to call repeatedly:
    // ensure_kv_cache_capacity is idempotent when the target is already
    // <= current capacity.
    for (auto& layer : layers) {
        if (layer && layer->attn_layer) {
            layer->attn_layer->reserve_kv_cache(total_tokens, device, batch_size);
        }
    }
}

void JambaModel::set_moe_inference_top_k(int k) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("expert routing policy changed");
    invalidate_backward_state_();
    // Pacote A.1: propagate the inference-only top-k override to every
    // JambaBlock that owns a router.  k <= 0 clears the override (each
    // block falls back to router->top_k).  k >= 1 will be clamped to
    // [1, router->num_experts] inside forward_moe — we keep the value
    // as-set here so a caller can configure once and switch models
    // without re-reading.  The override only takes effect when the
    // block's training_mode_ is false (decode path).
    for (auto& layer : layers) {
        if (layer) {
            layer->set_inference_top_k_override(k);
        }
    }
}

void JambaModel::release_full_precision_linear_weights() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("weight storage changed");
    invalidate_backward_state_();
    for (BitLinear* layer : collect_bitlinear_layers()) {
        // Keep quantization-sensitive projections (e.g. Mamba dt/B/C) in float:
        // releasing them would force the ternary path at inference, breaking the
        // mixed-precision design that QAT preserved during training.
        if (layer && !layer->quantization_sensitive()) {
            layer->release_full_precision_weight();
        }
    }
}

void JambaModel::save_edge_linear_pack(const std::string& path) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    namespace fs = std::filesystem;

    const fs::path destination(path);
    if (!destination.parent_path().empty()) {
        fs::create_directories(destination.parent_path());
    }
    const fs::path temporary =
        checkpoint_io::unique_temporary_path(destination);
    const auto linear_layers = collect_bitlinear_layers();
    try {
        std::ofstream out(
            temporary, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error(
                "Could not open edge pack temporary file for writing: " +
                temporary.string());
        }

        write_pod(out, kEdgePackMagic);
        write_pod(out, kEdgePackVersion);
        write_pod(out, static_cast<uint32_t>(linear_layers.size()));
        for (BitLinear* layer : linear_layers) {
            if (!layer) {
                throw std::runtime_error(
                    "Null BitLinear layer while exporting edge pack");
            }
            const uint8_t sensitive =
                layer->quantization_sensitive() ? 1u : 0u;
            write_pod(out, sensitive);
            const BitLinearPackedState state =
                layer->export_packed_state();
            validate_edge_packed_state(state);
            write_pod(out, static_cast<int32_t>(state.in_features));
            write_pod(out, static_cast<int32_t>(state.out_features));
            write_pod(
                out, static_cast<uint8_t>(state.use_bias ? 1 : 0));
            write_pod(out, state.weight_scale);
            write_vector(out, state.packed_weights);
            write_vector(out, state.magnitude);
            write_vector(out, state.bias);
            write_vector(out, state.flat_alpha);
            write_vector(out, state.flat_beta);
            if (sensitive) {
                // Mixed precision: store the FP32 latent weights so the
                // sensitive projection loads on the exact float path.
                Tensor weight =
                    layer->weight.data.get_device() == Device::GPU
                        ? layer->weight.data.cpu()
                        : layer->weight.data;
                if (weight.size == 0) {
                    throw std::runtime_error(
                        "Sensitive BitLinear has no float weights to export "
                        "to edge pack");
                }
                std::vector<float> values(
                    weight.data(), weight.data() + weight.size);
                if (!std::all_of(
                        values.begin(), values.end(),
                        [](float value) {
                            return std::isfinite(value);
                        })) {
                    throw std::runtime_error(
                        "Sensitive BitLinear contains non-finite weights");
                }
                write_vector(out, values);
            }
        }

        out.flush();
        if (!out) {
            throw std::runtime_error(
                "Could not flush edge pack payload: " +
                temporary.string());
        }
        out.close();
        if (!out) {
            throw std::runtime_error(
                "Could not close edge pack payload: " +
                temporary.string());
        }

        const uint64_t payload_bytes =
            static_cast<uint64_t>(fs::file_size(temporary));
        const std::string payload_sha256 =
            integrity::sha256_file_prefix(
                temporary, payload_bytes);
        if (payload_sha256.size() != 64) {
            throw std::logic_error(
                "Internal edge-pack SHA-256 has invalid length");
        }
        std::ofstream trailer(
            temporary, std::ios::binary | std::ios::app);
        if (!trailer.is_open()) {
            throw std::runtime_error(
                "Could not append edge pack integrity trailer");
        }
        write_pod(trailer, kEdgePackIntegrityMagic);
        write_pod(trailer, payload_bytes);
        trailer.write(
            payload_sha256.data(),
            static_cast<std::streamsize>(payload_sha256.size()));
        trailer.flush();
        if (!trailer) {
            throw std::runtime_error(
                "Could not flush edge pack integrity trailer");
        }
        trailer.close();
        if (!trailer) {
            throw std::runtime_error(
                "Could not close edge pack integrity trailer");
        }
        checkpoint_io::flush_file(temporary);
        checkpoint_io::atomic_replace(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

void JambaModel::load_edge_linear_pack(const std::string& path,
                                       bool release_full_precision) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_decode_graph_("edge pack load");
    // Invalidate before parsing: even a rejected or truncated pack must not
    // leave an older backward ticket live after a state-changing attempt.
    invalidate_backward_state_();
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open edge pack for reading: " + path);
    }

    const uint32_t magic = read_pod<uint32_t>(in);
    if (magic != kEdgePackMagic) {
        throw std::runtime_error("Invalid NSOS edge pack header");
    }
    const uint32_t version = read_pod<uint32_t>(in);
    if (version != kEdgePackLegacyVersion &&
        version != kEdgePackVersion) {
        throw std::runtime_error("Unsupported NSOS edge pack version");
    }
    uint64_t validated_payload_bytes = 0;
    if (version >= kEdgePackVersion) {
        const uint64_t file_bytes =
            static_cast<uint64_t>(
                std::filesystem::file_size(path));
        if (file_bytes < 12u + kEdgePackIntegrityBytes) {
            throw std::runtime_error(
                "Edge pack is too small for its integrity trailer");
        }
        in.seekg(static_cast<std::streamoff>(
            file_bytes - kEdgePackIntegrityBytes));
        const uint32_t integrity_magic =
            read_pod<uint32_t>(in);
        const uint64_t payload_bytes =
            read_pod<uint64_t>(in);
        std::string expected_sha256(64, '\0');
        in.read(
            expected_sha256.data(),
            static_cast<std::streamsize>(
                expected_sha256.size()));
        if (!in ||
            integrity_magic != kEdgePackIntegrityMagic ||
            payload_bytes != file_bytes - kEdgePackIntegrityBytes) {
            throw std::runtime_error(
                "Edge pack integrity trailer is missing or corrupt");
        }
        if (integrity::sha256_file_prefix(
                path, payload_bytes) != expected_sha256) {
            throw std::runtime_error(
                "Edge pack SHA-256 integrity mismatch");
        }
        validated_payload_bytes = payload_bytes;
        in.clear();
        in.seekg(8, std::ios::beg);
    }

    const auto linear_layers = collect_bitlinear_layers();
    const uint32_t expected_layers = read_pod<uint32_t>(in);
    if (expected_layers != linear_layers.size()) {
        throw std::runtime_error("Edge pack layer count mismatch");
    }

    struct PendingEdgeLinearLoad {
        BitLinear* layer = nullptr;
        BitLinearPackedState state;
        std::vector<float> sensitive_weight;
        Tensor staged_sensitive_weight;
        std::unique_ptr<BitLinear::PreparedPackedState> prepared;
        bool sensitive = false;
    };
    std::vector<PendingEdgeLinearLoad> pending_loads;
    pending_loads.reserve(expected_layers);
    for (uint32_t index = 0; index < expected_layers; ++index) {
        const uint8_t sensitive = read_pod<uint8_t>(in);
        if (sensitive > 1) {
            throw std::runtime_error("Edge pack contains an invalid sensitivity flag");
        }
        BitLinearPackedState state;
        state.in_features = read_pod<int32_t>(in);
        state.out_features = read_pod<int32_t>(in);
        const uint8_t use_bias = read_pod<uint8_t>(in);
        if (use_bias > 1) {
            throw std::runtime_error(
                "Edge pack contains an invalid bias flag");
        }
        state.use_bias = use_bias != 0;
        state.weight_scale = read_pod<float>(in);
        BitLinear* layer = linear_layers[index];
        const int expected_in = layer->input_features();
        const int expected_out = layer->output_features();
        const bool expected_bias = layer->uses_bias();
        if (state.in_features != expected_in || state.out_features != expected_out ||
            state.use_bias != expected_bias) {
            throw std::runtime_error("Edge pack layer metadata does not match model architecture");
        }
        if ((sensitive != 0) != layer->quantization_sensitive()) {
            throw std::runtime_error("Edge pack sensitivity metadata does not match model architecture");
        }
        const uint64_t required_in_out =
            static_cast<uint64_t>(state.in_features) *
            static_cast<uint64_t>(state.out_features);
        const uint64_t required_weight_words =
            (required_in_out + 15u) / 16u;
        const uint64_t required_out = static_cast<uint64_t>(state.out_features);
        const uint64_t required_in = static_cast<uint64_t>(state.in_features);
        state.packed_weights =
            read_vector<uint32_t>(in, required_weight_words, "packed_weights");
        state.magnitude = read_vector<float>(in, required_out, "magnitude");
        state.bias =
            read_vector<float>(in, state.use_bias ? required_out : 0, "bias");
        state.flat_alpha = read_vector<float>(in, required_in, "flat_alpha");
        state.flat_beta = read_vector<float>(in, required_in, "flat_beta");

        auto require_exact = [](size_t got, uint64_t need, const char* what) {
            if (static_cast<uint64_t>(got) != need) {
                throw std::runtime_error(
                    std::string("Edge pack ") + what +
                    " length does not match model architecture");
            }
        };
        require_exact(state.packed_weights.size(), required_weight_words, "packed_weights");
        require_exact(state.magnitude.size(), required_out, "magnitude");
        require_exact(state.bias.size(), state.use_bias ? required_out : 0, "bias");
        require_exact(state.flat_alpha.size(), required_in, "flat_alpha");
        require_exact(state.flat_beta.size(), required_in, "flat_beta");
        validate_edge_packed_state(state);

        PendingEdgeLinearLoad pending;
        pending.layer = layer;
        pending.state = std::move(state);
        pending.sensitive = sensitive != 0;
        if (sensitive) {
            pending.sensitive_weight =
                read_vector<float>(in, required_in_out, "sensitive_weight");
            require_exact(
                pending.sensitive_weight.size(), required_in_out,
                "sensitive_weight");
            if (!std::all_of(
                    pending.sensitive_weight.begin(),
                    pending.sensitive_weight.end(),
                    [](float value) {
                        return std::isfinite(value);
                    })) {
                throw std::runtime_error(
                    "Edge pack contains non-finite sensitive weights");
            }
        }
        pending_loads.push_back(std::move(pending));
    }

    if (version >= kEdgePackVersion) {
        const std::streamoff consumed = in.tellg();
        if (consumed < 0 ||
            static_cast<uint64_t>(consumed) !=
                validated_payload_bytes) {
            throw std::runtime_error(
                "Edge pack payload length does not match integrity trailer");
        }
    } else {
        char trailing = 0;
        if (in.read(&trailing, 1)) {
            throw std::runtime_error(
                "Legacy edge pack contains trailing payload");
        }
        if (!in.eof()) {
            throw std::runtime_error(
                "Legacy edge pack read failed before EOF");
        }
    }

    // Stage every tensor, unpacked representation, row sum and optional exact
    // FP32 weight before the first live-layer mutation. Validation, host OOM,
    // device OOM and transfer failures therefore leave the model unchanged.
    for (size_t index = 0; index < pending_loads.size(); ++index) {
        auto& pending = pending_loads[index];
        if (pending.sensitive) {
            const std::vector<int> shape = {
                pending.state.out_features,
                pending.state.in_features};
            Tensor host = Tensor::from_blob(
                              pending.sensitive_weight.data(),
                              shape, Device::CPU)
                              .clone();
            pending.staged_sensitive_weight =
                device == Device::CPU ? std::move(host)
                                      : host.to(device);
        }
        try {
            pending.prepared =
                std::make_unique<BitLinear::PreparedPackedState>(
                    pending.layer->prepare_packed_state(
                        pending.state, device,
                        pending.sensitive ? false
                                          : release_full_precision,
                        pending.sensitive
                            ? &pending.staged_sensitive_weight
                            : nullptr));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "Edge pack layer " + std::to_string(index) +
                " preparation failed: " + error.what());
        }
    }

    for (auto& pending : pending_loads) {
        pending.layer->commit_prepared_packed_state(
            std::move(*pending.prepared));
    }

    // Packed-state commits for sensitive layers intentionally install their
    // staged FP32 Tensor. Re-establish the model-level alias topology after
    // every layer has committed; when the value-head FP32 weight was released,
    // apply_weight_tying_ also records that no live storage alias exists.
    apply_weight_tying_();
    set_reference_path(!release_full_precision);
}

bool JambaModel::supports_streaming_inference() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return true;
}

bool JambaModel::supports_batched_streaming_inference() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    for (const auto& layer : layers) {
        if (!layer) {
            continue;
        }
        if (!layer->mamba_layer && !layer->mamba3_layer && !layer->ttt_layer && !layer->attn_layer) {
            return false;
        }
        // The corrected selective-SSM path now supports batched streaming: its
        // SSD state + conv ring are carried per-sequence through the batched
        // snapshot/restore (Mamba2SSD::{snapshot,restore}_streaming_state_batch)
        // and advanced per-row by the batched proper decode step.  No fallback
        // needed.
    }
    return !layers.empty();
}

void JambaModel::set_streaming_inference(bool enabled) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    if (enabled && training_mode_) {
        set_training_mode(false);
    }
    streaming_inference_enabled_ = enabled;
    // The captured decode graph embeds this session's device buffers; any
    // streaming-mode transition ends the session it was captured for.
    if (decode_graph_) {
        decode_graph_.reset();
        if (!decode_graph_disabled_) decode_graph_status_ = "idle";
    }
    for (auto& layer : layers) {
        layer->set_streaming_inference(enabled);
    }
}

void JambaModel::set_training_mode(bool enabled) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    if (decode_graph_) decode_graph_.reset();
    training_mode_ = enabled;
    for (auto& layer : layers) {
        layer->set_training_mode(enabled);
    }
    // Propagate to every BitLinear so inference (enabled == false) skips the
    // clone-heavy backward-state saves in BitLinear::forward.  collect_bitlinear
    // _layers() walks all blocks, sublayers, experts and the value head.
    for (BitLinear* bl : collect_bitlinear_layers()) {
        if (bl) {
            bl->set_training_mode(enabled);
        }
    }
}

bool JambaModel::training_mode() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return training_mode_;
}

uint64_t JambaModel::training_rng_sequence() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return training_rng_sequence_;
}

void JambaModel::set_training_rng_sequence(uint64_t sequence) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    training_rng_sequence_ = sequence;
}

void JambaModel::set_audit_collector(LayerAuditCollector* collector) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    audit_collector_ = collector;
    for (auto& layer : layers) {
        if (layer) {
            layer->set_audit_collector(collector);
        }
    }
}

void JambaModel::record_audit_token_context(const std::vector<int>& token_ids_sample,
                                            size_t batch_size,
                                            size_t prompt_tokens_total,
                                            size_t prompt_tokens_used,
                                            int context_limit,
                                            bool truncated) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    if (audit_collector_ && audit_collector_->enabled()) {
        audit_collector_->record_token_context(token_ids_sample,
                                               batch_size,
                                               prompt_tokens_total,
                                               prompt_tokens_used,
                                               context_limit,
                                               truncated);
    }
}

LayerAuditCollector* JambaModel::audit_collector() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return audit_collector_;
}

void JambaModel::attach_profiler(void* profiler) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    profiler_ = profiler;
}

void JambaModel::set_profiler_callbacks(
    ProfilerBeginLayerFn begin_fn,
    ProfilerEndLayerFn end_fn) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    profiler_begin_layer_ = begin_fn;
    profiler_end_layer_ = end_fn;
}

void* JambaModel::profiler() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return profiler_;
}

JambaModel::ProfilerBeginLayerFn
JambaModel::profiler_begin_layer_fn() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return profiler_begin_layer_;
}

JambaModel::ProfilerEndLayerFn
JambaModel::profiler_end_layer_fn() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return profiler_end_layer_;
}

void JambaModel::reset_runtime_telemetry() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    for (auto& layer : layers) {
        if (layer && layer->mamba3_layer) layer->mamba3_layer->reset_runtime_telemetry();
        if (layer && layer->mamba_layer) {
            layer->mamba_layer->reset_runtime_telemetry();
        }
    }
}

RuntimeTelemetrySnapshot JambaModel::runtime_telemetry() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    RuntimeTelemetrySnapshot snapshot;
    snapshot.mamba_layers.reserve(layers.size());
    for(const auto& layer:layers) if(layer && layer->mamba3_layer) snapshot.mamba3_layers.push_back(layer->mamba3_layer->telemetry());
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto& layer = layers[layer_index];
        if (!layer || !layer->mamba_layer) {
            continue;
        }
        const auto& mamba = *layer->mamba_layer;
        RuntimeTelemetrySnapshot::MambaLayer entry;
        entry.layer_index = static_cast<int>(layer_index);
        entry.fast_path_hits = mamba.gpu_fast_path_hits();
        entry.fast_path_fallbacks = mamba.gpu_fast_path_fallbacks();
        entry.faithful_forward_gpu_calls =
            mamba.faithful_forward_gpu_calls();
        entry.faithful_forward_host_fallbacks =
            mamba.faithful_forward_host_fallbacks();
        entry.faithful_backward_gpu_calls =
            mamba.faithful_backward_gpu_calls();
        entry.faithful_backward_host_fallbacks =
            mamba.faithful_backward_host_fallbacks();
        entry.faithful_streaming_gpu_calls =
            mamba.faithful_streaming_gpu_calls();
        entry.faithful_streaming_host_fallbacks =
            mamba.faithful_streaming_host_fallbacks();
        entry.stream_priming_gpu_calls =
            mamba.stream_priming_gpu_calls();
        entry.stream_priming_host_fallbacks =
            mamba.stream_priming_host_fallbacks();
        entry.faithful_recompute_forwards =
            mamba.faithful_recompute_forwards();
        entry.faithful_selective_history_recomputes =
            mamba.faithful_selective_history_recomputes();
        entry.faithful_full_block_recompute_forwards =
            mamba.faithful_full_block_recompute_forwards();
        entry.faithful_warp_aggregated_backward_calls =
            mamba.faithful_warp_aggregated_backward_calls();
        entry.faithful_deterministic_backward_calls =
            mamba.faithful_deterministic_backward_calls();
        entry.faithful_scalar_atomic_backward_calls =
            mamba.faithful_scalar_atomic_backward_calls();
        entry.faithful_reduced_conv_backward_calls =
            mamba.faithful_reduced_conv_backward_calls();
        entry.faithful_generic_atomic_conv_backward_calls =
            mamba.faithful_generic_atomic_conv_backward_calls();
        entry.faithful_peak_state_history_bytes =
            mamba.faithful_peak_state_history_bytes();
        entry.faithful_grouped_projection_forward_calls =
            mamba.faithful_grouped_projection_forward_calls();
        entry.faithful_grouped_projection_backward_calls =
            mamba.faithful_grouped_projection_backward_calls();
        entry.faithful_grouped_projection_cache_rebuilds =
            mamba.faithful_grouped_projection_cache_rebuilds();
        entry.faithful_grouped_projection_full_forward_calls =
            mamba.faithful_grouped_projection_full_forward_calls();
        entry.faithful_grouped_projection_sensitive_forward_calls =
            mamba.faithful_grouped_projection_sensitive_forward_calls();
        entry.faithful_grouped_projection_full_backward_calls =
            mamba.faithful_grouped_projection_full_backward_calls();
        entry.faithful_grouped_projection_sensitive_backward_calls =
            mamba.faithful_grouped_projection_sensitive_backward_calls();
        entry.last_fallback_reason = mamba.last_fallback_reason();

        snapshot.mamba_fast_path_hits += entry.fast_path_hits;
        snapshot.mamba_fast_path_fallbacks += entry.fast_path_fallbacks;
        snapshot.faithful_forward_gpu_calls +=
            entry.faithful_forward_gpu_calls;
        snapshot.faithful_forward_host_fallbacks +=
            entry.faithful_forward_host_fallbacks;
        snapshot.faithful_backward_gpu_calls +=
            entry.faithful_backward_gpu_calls;
        snapshot.faithful_backward_host_fallbacks +=
            entry.faithful_backward_host_fallbacks;
        snapshot.faithful_streaming_gpu_calls +=
            entry.faithful_streaming_gpu_calls;
        snapshot.faithful_streaming_host_fallbacks +=
            entry.faithful_streaming_host_fallbacks;
        snapshot.stream_priming_gpu_calls +=
            entry.stream_priming_gpu_calls;
        snapshot.stream_priming_host_fallbacks +=
            entry.stream_priming_host_fallbacks;
        snapshot.faithful_recompute_forwards +=
            entry.faithful_recompute_forwards;
        snapshot.faithful_selective_history_recomputes +=
            entry.faithful_selective_history_recomputes;
        snapshot.faithful_full_block_recompute_forwards +=
            entry.faithful_full_block_recompute_forwards;
        snapshot.faithful_warp_aggregated_backward_calls +=
            entry.faithful_warp_aggregated_backward_calls;
        snapshot.faithful_deterministic_backward_calls +=
            entry.faithful_deterministic_backward_calls;
        snapshot.faithful_scalar_atomic_backward_calls +=
            entry.faithful_scalar_atomic_backward_calls;
        snapshot.faithful_reduced_conv_backward_calls +=
            entry.faithful_reduced_conv_backward_calls;
        snapshot.faithful_generic_atomic_conv_backward_calls +=
            entry.faithful_generic_atomic_conv_backward_calls;
        snapshot.faithful_peak_state_history_bytes = std::max(
            snapshot.faithful_peak_state_history_bytes,
            entry.faithful_peak_state_history_bytes);
        snapshot.faithful_grouped_projection_forward_calls +=
            entry.faithful_grouped_projection_forward_calls;
        snapshot.faithful_grouped_projection_backward_calls +=
            entry.faithful_grouped_projection_backward_calls;
        snapshot.faithful_grouped_projection_cache_rebuilds +=
            entry.faithful_grouped_projection_cache_rebuilds;
        snapshot.faithful_grouped_projection_full_forward_calls +=
            entry.faithful_grouped_projection_full_forward_calls;
        snapshot.faithful_grouped_projection_sensitive_forward_calls +=
            entry.faithful_grouped_projection_sensitive_forward_calls;
        snapshot.faithful_grouped_projection_full_backward_calls +=
            entry.faithful_grouped_projection_full_backward_calls;
        snapshot.faithful_grouped_projection_sensitive_backward_calls +=
            entry.faithful_grouped_projection_sensitive_backward_calls;
        if (!entry.last_fallback_reason.empty()) {
            snapshot.mamba_last_fallback_reason =
                entry.last_fallback_reason;
        }
        snapshot.mamba_layers.push_back(std::move(entry));
    }
    return snapshot;
}

void JambaModel::validate_token_ids(const std::vector<int>& ids,
                                    const char* operation) const {
    if (ids.empty()) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires at least one token");
    }
    if (ids.size() >
        static_cast<size_t>(model_config_.max_context_tokens)) {
        throw std::length_error(std::string(operation) +
                                " exceeds max_context_tokens");
    }
    for (const int token : ids) {
        if (token < 0 || token >= vocab_size) {
            throw std::out_of_range(std::string(operation) +
                                    " token id is outside the vocabulary");
        }
    }
}

Tensor JambaModel::forward_ids(const std::vector<int>& ids, Context* ctx) {
    return forward_ids_impl(ids, ctx, false);
}

Tensor JambaModel::forward_ids_last(const std::vector<int>& ids, Context* ctx) {
    return forward_ids_impl(ids, ctx, true);
}

Tensor JambaModel::forward_ids_impl(const std::vector<int>& ids, Context* ctx,
                                  bool last_logits_only, bool headless) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    invalidate_backward_state_();
    validate_token_ids(ids, "JambaModel::forward_ids");
    if (streaming_inference_enabled_ && !last_input_ids_.empty() && ids.size() == 1) {
        last_input_ids_.push_back(ids.front());
    } else {
        last_input_ids_ = ids;
    }
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    record_audit_token_context(ids, 1, ids.size(), ids.size(),
                               model_config_.max_context_tokens, false);
    Tensor x = embedding->forward(ids);
    return forward_impl(x, ctx, last_logits_only, headless);
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA-graph decode (opt-in NSOS_CUDA_GRAPH_DECODE=1; requires NSOS_CUDA_PTDS)
//
// One streaming single-token forward is captured into a CUDA graph and then
// replayed per generated token: the whole per-token kernel cascade (embedding
// gather, per-layer norms/projections/Mamba step/attention decode/FFN, final
// norm + LM head) becomes ONE cudaGraphLaunch.  The two per-token host inputs
// (token id, KV position) enter through pinned staging ints that the graph's
// captured H2D memcpy nodes RE-READ at every replay — the host just rewrites
// two pinned ints per token.
//
// Guarded adoption: any capture problem restores the pre-capture session
// state and permanently disables the path for this model instance; the caller
// falls back to the eager forward_ids with identical results.  The parity
// contract (graph token sequence == eager token sequence) is enforced by
// tests/gpu/test_gpu_parity_decode_incremental.cpp validates the eager
// incremental path on the target GPU; graph-specific promotion remains gated
// separately until graph replay is re-enabled.
// ═══════════════════════════════════════════════════════════════════════════
#ifdef USE_CUDA
struct JambaDecodeGraph {
    enum class Phase { Warm, Captured };
    Phase phase = Phase::Warm;
    cudaGraphExec_t exec = nullptr;
    int* h_token = nullptr;  // pinned: the captured H2D nodes re-read these at
    int* h_pos = nullptr;    //   every replay (contents-at-execution semantics)
    int* d_token = nullptr;
    int* d_pos = nullptr;
    int next_pos = 0;        // host mirror of the KV position of the NEXT step
    size_t replays = 0;
    bool owns_capture = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t completed = nullptr;
    gpu::ExecutionContext* context = nullptr;
    Tensor logits;           // captured output buffer; held alive so the pool
                             // cannot hand it to anyone while the graph lives
    bool alloc_buffers() {
        return cudaEventCreate(&completed) == cudaSuccess &&
               cudaMallocHost(&h_token, sizeof(int)) == cudaSuccess &&
               cudaMallocHost(&h_pos, sizeof(int)) == cudaSuccess &&
               cudaMalloc(&d_token, sizeof(int)) == cudaSuccess &&
               cudaMalloc(&d_pos, sizeof(int)) == cudaSuccess;
    }
    ~JambaDecodeGraph() {
        if (stream) gpu::report_cleanup_status(cudaStreamSynchronize(stream), "decode graph drain");
        // Destroy the exec FIRST (no more replays can reference the pooled
        // buffers), then drain the capture quarantine back to the driver.
        // The `logits` Tensor member is destroyed AFTER this body; by then
        // release_capture() has cleared its captured-tracking, so it returns
        // to the pool normally (the graph is already dead — safe to reuse).
        if (exec) {
            gpu::report_cleanup_status(
                cudaGraphExecDestroy(exec), "decode graph exec destruction");
        }
        if (d_token) {
            gpu::report_cleanup_status(
                cudaFree(d_token), "decode token buffer release");
        }
        if (d_pos) {
            gpu::report_cleanup_status(
                cudaFree(d_pos), "decode position buffer release");
        }
        if (h_token) {
            gpu::report_cleanup_status(
                cudaFreeHost(h_token), "decode token staging release");
        }
        if (h_pos) {
            gpu::report_cleanup_status(
                cudaFreeHost(h_pos), "decode position staging release");
        }
        if (completed) gpu::report_cleanup_status(cudaEventDestroy(completed), "decode completion fence");
        if (owns_capture) gpu_pool_release_capture();
        if (context) context->freeze(false);
    }
};
#endif  // USE_CUDA

bool JambaModel::decode_graph_active() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
#ifdef USE_CUDA
    return decode_graph_ != nullptr &&
           decode_graph_->phase == JambaDecodeGraph::Phase::Captured;
#else
    return false;
#endif
}

std::string JambaModel::decode_graph_status() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return decode_graph_status_;
}

Tensor JambaModel::forward_ids_decode_graph(int token) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    invalidate_backward_state_();
#if !defined(USE_CUDA)
    (void)token;
    decode_graph_status_ = "unavailable: CPU build";
    return Tensor();
#else
    // Explicit streams, device-resident position and frozen typed workspaces
    // replace the old per-thread-stream/static-position capture contract.
    // This remains opt-in until the deployment's numerical/performance gates.
    static const bool env_enabled = [] {
        const char* v = std::getenv("NSOS_GPU_GRAPH_DECODE");
        if (!v) v = std::getenv("NSOS_CUDA_GRAPH_DECODE");
        return v != nullptr && v[0] == '1';
    }();
    if (!env_enabled) {
        decode_graph_status_ = "disabled by default: set NSOS_GPU_GRAPH_DECODE=1";
        return Tensor();
    }
    if (decode_graph_disabled_) return Tensor();
    validate_token_ids({token}, "graph decode");
    if (training_mode_) {
        decode_graph_status_ = "unavailable: training mode";
        return Tensor();
    }

    auto attn_layers = [&](auto&& fn) {
        for (auto& layer : layers) {
            if (layer && layer->uses_attention() && layer->attn_layer) {
                fn(*layer->attn_layer);
            }
        }
    };
    auto disable = [&](const std::string& why) {
        decode_graph_disabled_ = true;
        decode_graph_status_ = "disabled: " + why;
        decode_graph_.reset();
        std::fprintf(stderr, "[nsos] decode CUDA graph disabled: %s\n",
                     why.c_str());
    };

    // ── replay hot path ─────────────────────────────────────────────────
    if (decode_graph_ &&
        decode_graph_->phase == JambaDecodeGraph::Phase::Captured) {
        JambaDecodeGraph& dg = *decode_graph_;
        bool full = false;
        attn_layers([&](Attention& a) { full |= dg.next_pos >= a.kv_cache_capacity(); });
        if (full) { disable("reserved KV capacity exhausted"); return Tensor(); }
        gpu_runtime_check(cudaEventSynchronize(dg.completed), "decode staging ownership");
        *dg.h_token = token;
        *dg.h_pos = dg.next_pos;
        if (cudaGraphLaunch(dg.exec, gpu::current_stream()) != cudaSuccess) {
            disable("cudaGraphLaunch failed mid-generation");
            throw std::runtime_error("Graph launch failed; session state is uncertain and must be reset");
        }
        if (cudaEventRecord(dg.completed, gpu::current_stream()) != cudaSuccess) {
            disable("decode completion fence failed");
            throw std::runtime_error("Graph completion fence failed; reset session before retrying");
        }
        gpu::record_dispatch(gpu::DispatchPath::GraphReplay);
        record_gpu_transfer(Device::GPU, Device::CPU, 2 * sizeof(int));
        ++dg.next_pos;
        ++dg.replays;
        last_input_ids_.push_back(token);
        // A replay bypasses Attention::forward, so the host position mirrors
        // must advance here to keep session forks/snapshots truthful.
        attn_layers([](Attention& a) { a.advance_cached_tokens_external(); });
        return dg.logits;
    }

    // ── one-time preconditions + warm-up step ───────────────────────────
    if (!decode_graph_) {
        const char* sync_env = std::getenv("NSOS_CUDA_SYNC");
        if (sync_env && sync_env[0] == '1') {
            disable("NSOS_CUDA_SYNC=1 (device-wide syncs are illegal during "
                    "stream capture)");
            return Tensor();
        }
        if (device != Device::GPU) {
            disable("model is not on the GPU");
            return Tensor();
        }
        if(model_config_.mamba3_enabled) {
            disable("Mamba3 v1 publishes session status through an explicit audit boundary");
            return Tensor();
        }
        if (!streaming_inference_enabled_) {
            disable("streaming inference is off");
            return Tensor();
        }
        if (last_input_ids_.empty()) {
            disable("no prefill ran before decode");
            return Tensor();
        }
        for (auto& layer : layers) {
            if (layer && layer->attn_layer && layer->attn_layer->sparse_attention_enabled()) {
                disable("sparse attention has no graph position contract");
                return Tensor();
            }
            if (layer && layer->uses_moe()) {
                // Selected-expert decode and its dense device fallback are
                // capturable within this guard. Larger expert counts may
                // dispatch through host-synchronized routing instead.
                const int experts =
                    layer->router ? layer->router->num_experts : 0;
                if (experts <= 0 || experts > 32) {
                    disable("MoE layer with num_experts > 32 (dense device "
                            "decode path guard) — the batched dispatch is "
                            "host-synced and not capturable");
                    return Tensor();
                }
            }
            if (layer && layer->uses_ttt()) {
                disable("TTT layer present (host-side per-token adaptation)");
                return Tensor();
            }
        }
        bool has_mamba = false;
        for (auto& layer : layers) {
            if (layer && !layer->uses_attention() && !layer->uses_ttt()) {
                has_mamba = true;
            }
        }
        const char* step_env = std::getenv("NSOS_MAMBA_GPU_STEP");
        // GPU-first: the fused device step (diagonal AND N-state) is default
        // ON; only an explicit =0 opt-out forces the host step, which does
        // per-token D2H and cannot be captured.
        if (has_mamba && step_env != nullptr && step_env[0] == '0') {
            disable("NSOS_MAMBA_GPU_STEP=0 forces the HOST Mamba decode step "
                    "(per-token D2H; not capturable)");
            return Tensor();
        }
        if (cuda_graphs_supported() == 0) {
            disable("driver/device reports no CUDA graph support");
            return Tensor();
        }
        decode_graph_ = std::make_shared<JambaDecodeGraph>();
        decode_graph_->stream = gpu::current_stream();
        decode_graph_->context = &execution_context_;
        decode_graph_status_ = "warming";
        // Warm-up: run THIS token through the normal eager path.  It warms
        // the tensor pool with decode-shaped buffers, uploads lazy device
        // state (Mamba stream ring/h), and populates the BitLinear inference
        // weight caches — everything that would otherwise allocate or sync
        // INSIDE the capture.
        return forward_ids({token}, nullptr);
    }

    // ── capture (second decode token) ───────────────────────────────────
    JambaDecodeGraph& dg = *decode_graph_;
    if (!dg.alloc_buffers()) {
        (void)cudaGetLastError();
        disable("pinned/device staging allocation failed");
        return Tensor();
    }
    int pos0 = -1;
    bool lockstep = true;
    bool reserved = true;
    attn_layers([&](Attention& a) {
        const int p = a.cached_tokens();
        if (pos0 < 0) pos0 = p;
        else if (p != pos0) lockstep = false;
        // Cache addresses must remain stable. Tiled attention scratch no
        // longer grows with the sequence length on replay.
        if (a.kv_cache_capacity() <= p + 1) reserved = false;
    });
    if (!lockstep) { disable("attention layers out of position lockstep"); return Tensor(); }
    if (!reserved) {
        disable("KV cache not pre-reserved (call reserve_kv_cache for "
                "prompt+max_tokens before the decode loop)");
        return Tensor();
    }
    if (pos0 < 0) pos0 = static_cast<int>(last_input_ids_.size());
    *dg.h_token = token;
    *dg.h_pos = pos0;
    dg.next_pos = pos0;

    // Snapshot BEFORE capture: recording executes the HOST side of the
    // forward (position mirrors advance) without running any device work; a
    // failed capture must undo those host advances before the eager retry.
    JambaSessionSnapshot pre_capture = fork_session();
    // fork_session shares the KV cache buffer (use_count 2) for batch<=1.  If
    // left shared, ensure_kv_cache_capacity would reallocate copy-on-write on
    // the FIRST recorded decode step — a realloc + D2D copy INSIDE the capture
    // that corrupts the graph (observed as an illegal memory access on the T4).
    // Detach to a uniquely-owned cache now, before BeginCapture, so recording
    // sees a stable buffer and never reallocs.
    attn_layers([](Attention& a) { a.make_kv_cache_unique(); });
    cudaGraph_t graph = nullptr;
    bool capture_open = false;
    // Make the tensor pool capture-safe for the whole recording: no
    // cudaFree/trim/memGetInfo (all synchronize the device — illegal during
    // capture), and every buffer allocated here is quarantined on free so no
    // address the graph writes on replay is ever recycled.  Drained by
    // ~JambaDecodeGraph::release_capture() when the graph is destroyed.
    bool tracking_open = false;
    try {
        gpu_pool_begin_capture();
        tracking_open = true;
        dg.owns_capture = true;
        execution_context_.freeze(true);
        if (cudaStreamBeginCapture(gpu::current_stream(),
                                   cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            throw std::runtime_error("cudaStreamBeginCapture failed");
        }
        capture_open = true;
        gpu::set_decode_position(dg.d_pos);
        if (cudaMemcpyAsync(dg.d_token, dg.h_token, sizeof(int),
                            cudaMemcpyHostToDevice,
                            gpu::current_stream()) != cudaSuccess ||
            cudaMemcpyAsync(dg.d_pos, dg.h_pos, sizeof(int),
                            cudaMemcpyHostToDevice,
                            gpu::current_stream()) != cudaSuccess) {
            throw std::runtime_error("token/pos H2D memcpy node failed");
        }
        Tensor x = embedding->forward_device_ids(dg.d_token, 1);
        Tensor lg = forward(x, nullptr);
        gpu::set_decode_position(nullptr);
        cudaError_t end_status = cudaStreamEndCapture(gpu::current_stream(), &graph);
        capture_open = false;
        if (end_status != cudaSuccess || graph == nullptr) {
            throw std::runtime_error(std::string("cudaStreamEndCapture: ") +
                                     cudaGetErrorString(end_status));
        }
        if (cudaGraphInstantiate(&dg.exec, graph, 0) != cudaSuccess) {
            throw std::runtime_error("cudaGraphInstantiate failed");
        }
        const cudaError_t upload_status =
            cudaGraphUpload(dg.exec, gpu::current_stream());
        if (upload_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGraphUpload: ") +
                cudaGetErrorString(upload_status));
        }
        const cudaError_t destroy_status = cudaGraphDestroy(graph);
        if (destroy_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGraphDestroy: ") +
                cudaGetErrorString(destroy_status));
        }
        graph = nullptr;
        dg.logits = lg;
        // Stop tracking new allocations as captured.  The captured set +
        // quarantine persist (drained by ~JambaDecodeGraph); `x`/`lg` freed
        // as this scope unwinds still route to quarantine via the captured
        // set, independent of the capturing flag.
        gpu_pool_end_capture();
        tracking_open = false;
    } catch (const std::exception& e) {
        gpu::set_decode_position(nullptr);
        if (capture_open) {
            cudaGraph_t partial = nullptr;
            gpu::report_cleanup_status(
                cudaStreamEndCapture(gpu::current_stream(), &partial),
                "failed decode stream capture termination");
            if (partial) {
                gpu::report_cleanup_status(
                    cudaGraphDestroy(partial),
                    "partial decode graph destruction");
            }
        }
        if (graph) {
            gpu::report_cleanup_status(
                cudaGraphDestroy(graph), "decode graph rollback destruction");
        }
        // End capture tracking and drain the quarantine now — this graph never
        // reached Captured, so nothing will replay against those buffers.
        if (tracking_open) gpu_pool_end_capture();
        if (dg.owns_capture) { gpu_pool_release_capture(); dg.owns_capture = false; }
        execution_context_.freeze(false);
        restore_session(pre_capture);
        disable(std::string("capture failed: ") + e.what());
        return Tensor();
    }
    // The hooks only influence what gets RECORDED; clear them so any later
    // eager forward (e.g. after a mid-generation disable) behaves normally.

    // Capture only RECORDS — this launch EXECUTES the captured step.
    if (cudaGraphLaunch(dg.exec, gpu::current_stream()) != cudaSuccess) {
        disable("first cudaGraphLaunch failed");
        throw std::runtime_error("First graph launch failed; reset session before retrying");
    }
    if (cudaEventRecord(dg.completed, gpu::current_stream()) != cudaSuccess) {
        disable("first decode completion fence failed");
        throw std::runtime_error("Graph completion fence failed; reset session before retrying");
    }
    gpu::record_dispatch(gpu::DispatchPath::GraphReplay);
    record_gpu_transfer(Device::GPU, Device::CPU, 2 * sizeof(int));
    dg.phase = JambaDecodeGraph::Phase::Captured;
    dg.next_pos = pos0 + 1;
    dg.replays = 1;
    last_input_ids_.push_back(token);
    decode_graph_status_ = "active";
    std::fprintf(stderr,
                 "[nsos] decode GPU graph ACTIVE: captured 1 step at pos=%d; "
                 "replaying one launch per token\n",
                 pos0);
    return dg.logits;
#endif
}

Tensor JambaModel::forward_ids_batch(const std::vector<std::vector<int>>& batch_ids, Context* ctx) {
    return forward_ids_batch_impl(batch_ids, ctx, false);
}

Tensor JambaModel::forward_ids_training_hidden(const std::vector<int>& ids, Context* ctx) {
    return forward_ids_impl(ids, ctx, false, true);
}

Tensor JambaModel::forward_ids_batch_training_hidden(const std::vector<std::vector<int>>& ids, Context* ctx) {
    return forward_ids_batch_impl(ids, ctx, true);
}

Tensor JambaModel::forward_ids_batch_impl(const std::vector<std::vector<int>>& batch_ids, Context* ctx, bool headless) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    invalidate_backward_state_();
    if (batch_ids.empty()) {
        throw std::invalid_argument(
            "JambaModel::forward_ids_batch requires a non-empty batch");
    }
    for (const auto& ids : batch_ids) {
        validate_token_ids(ids, "JambaModel::forward_ids_batch");
    }
    if (streaming_inference_enabled_ &&
        !std::all_of(batch_ids.begin(), batch_ids.end(), [](const std::vector<int>& ids) {
            return ids.size() == 1;
        })) {
        throw std::runtime_error(
            "Streaming inference does not support batched token-id inputs yet");
    }
    int max_seq_len = 0;
    std::vector<std::vector<int>> padded_batch = batch_ids;
    last_input_batch_lengths_.clear();
    last_input_batch_lengths_.reserve(batch_ids.size());
    for (const auto& ids : batch_ids) {
        const int current_len = static_cast<int>(ids.size());
        max_seq_len = std::max(max_seq_len, current_len);
        last_input_batch_lengths_.push_back(current_len);
    }
    for (auto& ids : padded_batch) {
        ids.resize(static_cast<size_t>(max_seq_len), 0);
    }
    last_input_ids_.clear();
    last_input_batches_ = padded_batch;
    std::vector<int> sample_ids;
    if (!padded_batch.empty()) {
        sample_ids = padded_batch.front();
    }
    size_t total_tokens = 0;
    for (int length : last_input_batch_lengths_) {
        total_tokens += static_cast<size_t>(std::max(length, 0));
    }
    record_audit_token_context(sample_ids, padded_batch.size(), total_tokens,
                               total_tokens, model_config_.max_context_tokens, false);
    Tensor x = embedding->forward_batch(padded_batch);
    zero_sequence_suffix_inplace(x, last_input_batch_lengths_);
    return forward_impl(x, ctx, false, headless);
}

Tensor JambaModel::forward_trunk(const std::vector<int>& ids, Context* ctx) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    // Trunk-only forwards overwrite per-layer activations but do not populate
    // the language-model head/final-norm state required by backward().
    invalidate_backward_state_();
    validate_token_ids(ids, "JambaModel::forward_trunk");
    last_input_ids_ = ids;
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    Tensor hidden = embedding->forward(ids);
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        hidden = layer->forward(hidden, ctx);
    }
    return hidden.rmsnorm();
}

Tensor JambaModel::forward_trunk_batch(const std::vector<std::vector<int>>& batch_ids,
                                       Context* ctx) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    invalidate_backward_state_();
    if (batch_ids.empty()) {
        throw std::invalid_argument(
            "JambaModel::forward_trunk_batch requires a non-empty batch");
    }
    for (const auto& ids : batch_ids) {
        validate_token_ids(ids, "JambaModel::forward_trunk_batch");
    }
    int max_seq_len = 0;
    std::vector<std::vector<int>> padded_batch = batch_ids;
    last_input_batch_lengths_.clear();
    last_input_batch_lengths_.reserve(batch_ids.size());
    for (const auto& ids : batch_ids) {
        const int current_len = static_cast<int>(ids.size());
        max_seq_len = std::max(max_seq_len, current_len);
        last_input_batch_lengths_.push_back(current_len);
    }
    for (auto& ids : padded_batch) {
        ids.resize(static_cast<size_t>(max_seq_len), 0);
    }
    last_input_ids_.clear();
    last_input_batches_ = padded_batch;
    if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
        throw AbortException();
    }
    Tensor hidden = embedding->forward_batch(padded_batch);
    zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    for (auto& layer : layers) {
        if (ctx && ctx->abort_signal && ctx->abort_signal->load(std::memory_order_relaxed)) {
            throw AbortException();
        }
        layer->set_batch_valid_lengths(last_input_batch_lengths_);
        hidden = layer->forward(hidden, ctx);
        zero_sequence_suffix_inplace(hidden, last_input_batch_lengths_);
    }
    Tensor trunk = hidden.rmsnorm();
    zero_sequence_suffix_inplace(trunk, last_input_batch_lengths_);
    return trunk;
}

Tensor JambaModel::forward_embedding(const Tensor& x, Context* ctx) {
    return forward(x, ctx);
}

void JambaModel::set_reasoning_policy(ReasoningPolicy policy) {
    policy.validate();
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    reasoning_policy_ = std::move(policy);
    last_reasoning_report_ = {};
}

void JambaModel::clear_reasoning_policy() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    reasoning_policy_ = {};
    last_reasoning_report_ = {};
}

bool JambaModel::has_reasoning_policy() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return bool(reasoning_policy_.propose) && bool(reasoning_policy_.verify);
}

VerifiedReasoningReport JambaModel::last_reasoning_report() const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return last_reasoning_report_;
}

Tensor JambaModel::reason(const Tensor& x, int num_simulations) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    // Validate before any forward, transfer or mutation. There is deliberately
    // no fallback to entropy/margin and no implicit noise-based proposer.
    reasoning_policy_.validate();
    if (x.size == 0 || x.shape.empty() || x.shape.back() != d_model || x.get_device() != device) {
        throw std::invalid_argument("reason expects non-empty hidden states, not vocabulary logits");
    }
    MCTSConfig cfg;
    cfg.num_simulations = num_simulations > 0 ? num_simulations
                                              : model_config_.mcts_simulations;
    cfg.max_depth = model_config_.mcts_depth;
    cfg.num_children_per_expansion = 3;
    cfg.max_nodes = static_cast<size_t>(std::max(cfg.num_simulations, 1));
    cfg.use_noise = false;
    last_reasoning_report_ = {};
    invalidate_backward_state_();
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    auto result = run_verified_reasoning(x, reasoning_policy_, cfg);
    last_reasoning_report_ = std::move(result.report);
    return std::move(result.state);
}

Tensor JambaModel::forward_thought(const Tensor& x, int steps) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    Tensor hidden = x;
    for (int i = 0; i < std::max(steps, 1); ++i) {
        hidden = forward_embedding(hidden, nullptr);
    }
    // Only the final iteration's activations survive. Advertising a normal
    // full-model backward here would silently omit the earlier recurrent
    // thought steps.
    invalidate_backward_state_();
    return hidden;
}

void JambaModel::run_reasoning_loop(int iterations) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    // Sem estado de entrada definido, não há o que refinar.
    // Esta sobrecarga existe para compatibilidade; use a versão com Tensor.
    (void)iterations;
    std::cerr << "[JambaModel] run_reasoning_loop(int) chamado sem estado de entrada."
                 " Use run_reasoning_loop(Tensor, int) para MCTS real.\n";
}

Tensor JambaModel::run_reasoning_loop(const Tensor& x, int iterations) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    if (x.size == 0 || iterations <= 0) return x;

    // Iterative MCTS Refinement:
    // Cada iteração do loop usa o melhor estado MCTS da iteração anterior
    // como novo ponto de partida, criando um refinamento progressivo.
    Tensor current_state = x;
    const int sims_per_iter = std::max(200 / std::max(iterations, 1), 10);

    for (int i = 0; i < iterations; ++i) {
        Tensor refined = reason(current_state, sims_per_iter);
        if (refined.size == 0) break;

        // Do not transform an accepted state after verification: that would
        // return an unverified answer and advance mutable model state.
        current_state = std::move(refined);

        std::cout << "[ReasoningLoop] Iteração " << i + 1 << "/" << iterations
                  << " concluída.\n";
    }
    invalidate_backward_state_();
    return current_state;
}

void JambaModel::backward_external(const Tensor& grad, Context& ctx) {
    backward(grad, ctx);
}

void JambaModel::backward_embedding(const Tensor& grad, Context& ctx) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    (void)ctx;
    if (!last_input_batches_.empty()) {
        embedding->backward_batch(grad, last_input_batches_);
    } else if (!last_input_ids_.empty()) {
        embedding->backward(grad, last_input_ids_);
    }
}

TiledCrossEntropyResult JambaModel::tiled_head_loss(const TiledCrossEntropyOptions& options) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    if (!backward_ready_ || !headless_backward_ || !tiled_head_ready_)
        throw std::logic_error("Tiled head requires a fresh headless training forward");
    tiled_head_ready_ = false;
    try { return value_head->cross_entropy_tiled(saved_final_norm_, options); }
    catch (...) { invalidate_backward_state_(); throw; }
}

void JambaModel::backward_training_hidden(const Tensor& grad, Context& ctx) {
    backward_impl(grad, ctx, true);
}

void JambaModel::backward(const Tensor& grad, Context& ctx) {
    backward_impl(grad, ctx, false);
}

void JambaModel::backward_impl(const Tensor& grad, Context& ctx, bool headless) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    gpu::ExecutionContext::Scope gpu_scope(execution_context_, device == Device::GPU);
    // Opt-in diagnostic paired with the forward-side [ltime] report.  Each
    // timer fences the device, so this must never be enabled for throughput
    // measurement; it exists only to localize backward latency by layer.
    static const bool nsos_layer_timing = [] {
        const char* e = std::getenv("NSOS_LAYER_TIMING");
        return e != nullptr && e[0] == '1';
    }();
    std::vector<double> backward_layer_ms;
    if (nsos_layer_timing) {
        backward_layer_ms.assign(layers.size(), 0.0);
    }
    const auto cancellation_requested = [&ctx]() {
        return ctx.abort_signal != nullptr &&
               ctx.abort_signal->load(
                   std::memory_order_acquire);
    };
    if (cancellation_requested()) {
        invalidate_backward_state_();
        throw AbortException();
    }
    if (!backward_ready_) {
        throw std::runtime_error(
            "JambaModel backward requires a fresh successful training "
            "forward");
    }
    // Consume before the first mutation. If any lower layer fails, callers
    // must perform a new forward and cannot accidentally double-accumulate a
    // partially completed pass.
    if (headless != headless_backward_ || (headless && tiled_head_ready_))
        throw std::logic_error("Backward head policy does not match its forward/loss ticket");
    if (headless && (grad.shape != saved_final_norm_.shape || grad.get_device() != saved_final_norm_.get_device()))
        throw std::invalid_argument("Headless backward gradient shape/device mismatch");
    invalidate_backward_state_();
    const bool audit_head =
        audit_collector_ && audit_collector_->enabled();
    AuditLatencyTimer diagnostic_head_timer(
        nsos_layer_timing, grad.get_device());
    AuditLatencyTimer head_timer(audit_head, grad.get_device());
    Tensor dy = headless ? grad : value_head->backward(grad);
    const double diagnostic_head_ms =
        nsos_layer_timing ? diagnostic_head_timer.elapsed_ms() : 0.0;
    if (audit_head) {
        const double latency_ms = head_timer.elapsed_ms();
        audit_collector_->record_backward(-1, "value_head", grad, dy, latency_ms);
    }
    AuditLatencyTimer diagnostic_norm_timer(
        nsos_layer_timing, dy.get_device());
    if (saved_final_hidden_.size > 0 && saved_final_norm_.size > 0) {
        const float final_norm_eps =
            (model_config_.mamba2_faithful || model_config_.mamba3_enabled) ? 1e-5f : 1e-6f;
        if ((model_config_.mamba2_faithful || model_config_.mamba3_enabled) &&
            saved_final_rms_.size > 0 &&
            final_norm_weight_.data.size > 0) {
            const int width = saved_final_hidden_.shape.back();
            const int rows = saved_final_hidden_.size / width;
            final_norm_weight_.add_grad(
                dy.mul(saved_final_rms_).reshape({rows, width}).sum(0));
            dy = dy.mul(final_norm_weight_.data);
        }
        dy = saved_final_hidden_.rmsnorm_backward(
            dy,
            (model_config_.mamba2_faithful || model_config_.mamba3_enabled) ? saved_final_rms_
                                          : saved_final_norm_,
            final_norm_eps);
    }
    const double diagnostic_norm_ms =
        nsos_layer_timing ? diagnostic_norm_timer.elapsed_ms() : 0.0;
    static const bool attn_dbg_bwd = []() {
        const char* e = std::getenv("NSOS_ATTN_DEBUG");
        return e != nullptr && e[0] == '1';
    }();
    if (attn_dbg_bwd) {
        std::fprintf(stderr, "[bwd final-norm] dy_absmean=%.5g\n",
                     tensor_abs_mean(dy));
    }
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        if (cancellation_requested()) {
            throw AbortException();
        }
        AuditLatencyTimer diagnostic_layer_timer(
            nsos_layer_timing, dy.get_device());
        dy = layers[i]->backward(dy, &ctx);
        if (nsos_layer_timing) {
            backward_layer_ms[static_cast<size_t>(i)] =
                diagnostic_layer_timer.elapsed_ms();
        }
        if (attn_dbg_bwd) {
            std::fprintf(stderr, "[bwd L%d %-14s] dy_in_absmean=%.5g\n", i,
                         layers[i]->audit_block_type().c_str(),
                         tensor_abs_mean(dy));
        }
    }
    if (cancellation_requested()) {
        throw AbortException();
    }
    AuditLatencyTimer diagnostic_embedding_timer(
        nsos_layer_timing, dy.get_device());
    backward_embedding(dy, ctx);
    const double diagnostic_embedding_ms =
        nsos_layer_timing ? diagnostic_embedding_timer.elapsed_ms() : 0.0;

    // N6: fold the LM head's gradient for the shared matrix into embedding.weight
    // (the single trained copy), then clear the head's grad so it never double-
    // counts (it is excluded from parameters(), so the trainer never zeroes it).
    if (weight_tied_ && value_head->weight.grad.size > 0) {
        embedding->weight.add_grad(value_head->weight.grad);
        value_head->weight.zero_grad();
    }
    if (nsos_layer_timing) {
        std::cerr << "[lbtime] head="
                  << static_cast<int>(diagnostic_head_ms)
                  << " norm=" << static_cast<int>(diagnostic_norm_ms);
        for (size_t i = 0; i < backward_layer_ms.size(); ++i) {
            std::cerr << " L" << i << "="
                      << static_cast<int>(backward_layer_ms[i]);
        }
        std::cerr << " embed="
                  << static_cast<int>(diagnostic_embedding_ms)
                  << "ms" << std::endl;
    }
}

void JambaModel::reset_session() {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    last_input_ids_.clear();
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_rms_ = Tensor();
    saved_final_norm_ = Tensor();
    // A captured decode graph embeds pointers into THIS session's device
    // state (KV cache, Mamba ring); replaying it after a reset would write
    // stale buffers.  Drop it — a new generation re-captures cheaply.
    if (decode_graph_) {
        decode_graph_.reset();
        if (!decode_graph_disabled_) decode_graph_status_ = "idle";
    }
    for (auto& layer : layers) {
        layer->reset();
    }
}

namespace {
void detach_graph_snapshot(AttentionCacheSnapshot& cache) {
    // A replay bypasses eager copy-on-write. Keep the graph's addresses fixed
    // and copy the snapshot, never let a later replay mutate shared storage.
    if (cache.compact)
        cache.compact = cache.compact->resize(cache.cache_capacity_tokens, cache.cached_tokens);
    if (cache.key_cache.size) cache.key_cache = cache.key_cache.clone();
    if (cache.value_cache.size) cache.value_cache = cache.value_cache.clone();
}
}

JambaSessionSnapshot JambaModel::fork_session(bool device_resident) const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    JambaSessionSnapshot snapshot;
    snapshot.streaming_enabled = streaming_inference_enabled_;
    snapshot.input_ids = last_input_ids_;
    snapshot.blocks.reserve(layers.size());
    for (const auto& layer : layers) {
        snapshot.blocks.push_back(layer ? layer->snapshot_session_state(device_resident)
                                        : JambaBlockSessionSnapshot{});
        if (decode_graph_active()) detach_graph_snapshot(snapshot.blocks.back().attention);
    }
    return snapshot;
}

std::vector<JambaSessionSnapshot> JambaModel::fork_session_batch(bool device_resident) const {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    if (!supports_batched_streaming_inference()) {
        throw std::runtime_error("Model does not support batched streaming snapshots");
    }
    std::vector<std::vector<JambaBlockSessionSnapshot>> layer_snapshots;
    layer_snapshots.reserve(layers.size());
    size_t batch_size = 0;
    for (const auto& layer : layers) {
        auto snapshots = layer ? layer->snapshot_session_state_batch(device_resident)
                               : std::vector<JambaBlockSessionSnapshot>{};
        if (!snapshots.empty()) {
            batch_size = std::max(batch_size, snapshots.size());
        }
        layer_snapshots.push_back(std::move(snapshots));
    }

    std::vector<JambaSessionSnapshot> snapshots(batch_size);
    for (size_t item = 0; item < batch_size; ++item) {
        snapshots[item].streaming_enabled = streaming_inference_enabled_;
        snapshots[item].blocks.reserve(layers.size());
        for (size_t layer_index = 0; layer_index < layer_snapshots.size(); ++layer_index) {
            const auto& layer_batch = layer_snapshots[layer_index];
            if (item < layer_batch.size()) {
                snapshots[item].blocks.push_back(layer_batch[item]);
                if (decode_graph_active()) detach_graph_snapshot(snapshots[item].blocks.back().attention);
            } else {
                snapshots[item].blocks.push_back(JambaBlockSessionSnapshot{});
            }
        }
    }
    return snapshots;
}

void JambaModel::restore_session(const JambaSessionSnapshot& snapshot) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    set_streaming_inference(snapshot.streaming_enabled);
    last_input_ids_ = snapshot.input_ids;
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_rms_ = Tensor();
    saved_final_norm_ = Tensor();

    const size_t shared_layers = std::min(layers.size(), snapshot.blocks.size());
    for (size_t index = 0; index < shared_layers; ++index) {
        layers[index]->restore_session_state(snapshot.blocks[index]);
    }
    for (size_t index = shared_layers; index < layers.size(); ++index) {
        layers[index]->reset();
    }
}

void JambaModel::restore_session_batch(const std::vector<JambaSessionSnapshot>& snapshots) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    if (snapshots.empty()) {
        reset_session();
        return;
    }
    if (!supports_batched_streaming_inference()) {
        throw std::runtime_error("Model does not support batched streaming restore");
    }

    set_streaming_inference(snapshots.front().streaming_enabled);
    last_input_ids_.clear();
    last_input_batches_.clear();
    last_input_batch_lengths_.clear();
    saved_final_hidden_ = Tensor();
    saved_final_rms_ = Tensor();
    saved_final_norm_ = Tensor();

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        std::vector<JambaBlockSessionSnapshot> layer_batch;
        layer_batch.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            if (layer_index < snapshot.blocks.size()) {
                layer_batch.push_back(snapshot.blocks[layer_index]);
            } else {
                layer_batch.push_back(JambaBlockSessionSnapshot{});
            }
        }
        layers[layer_index]->restore_session_state_batch(layer_batch);
    }
}

void JambaModel::set_hamiltonian_mode(bool enabled) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    for (auto& layer : layers) {
        if (layer->ttt_layer) {
            layer->ttt_layer->set_use_hamiltonian(enabled);
        }
    }
}

void JambaModel::session_adapt(const Tensor& x, const Tensor& y) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    invalidate_backward_state_();
    for (auto& layer : layers) {
        if (layer->ttt_layer) {
            layer->ttt_layer->initialize_from_meta(x, y);
        }
    }
}

Tensor JambaModel::run_simd_inference(const Tensor& x, int steps) {
    std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
    return forward_thought(x, steps);
}

JambaBlock::JambaBlock(int dm,
                       bool is_attn,
                       bool is_moe_flag,
                       bool is_ttt_layer,
                       int li,
                       int tl,
                       int query_heads,
                       int kv_heads,
                       int configured_experts,
                       int configured_top_k,
                       bool exact_attention_training,
                       float dropout_rate,
                       bool use_gradient_checkpointing,
                       int configured_expert_hidden_dim,
                       bool use_chrass,
                       float chrass_density,
                       uint32_t chrass_seed,
                       bool use_kan,
                       bool mamba_proper_ssm,
                       bool mamba_state_expansion,
                       int mamba_d_state,
                       int mamba_conv_kernel,
                       bool mamba2_faithful,
                       int mamba_expand,
                       int mamba_head_dim,
                       int mamba_n_groups,
                       float rope_theta,
                       int sliding_window,
                       int max_context_tokens,
                       HybridComposition hybrid_composition,
                       bool faithful_attention_linears,
                       float hybrid_mamba_gate_init,
                       float hybrid_attention_gate_init,
                       float hybrid_ffn_gate_init,
                       bool mamba3_enabled,
                       const Mamba3Config& mamba3_config)
    : is_attention(is_attn),
      is_moe(is_moe_flag),
      is_ttt(is_ttt_layer),
      hybrid_parallel_(
          hybrid_composition == HybridComposition::ParallelGated &&
          is_attn && !is_ttt_layer),
      faithful_mamba_core_only_((mamba2_faithful || mamba3_enabled) && !is_attn &&
                                !is_ttt_layer && !is_moe_flag && !use_kan &&
                                !use_chrass),
      core_norm_eps_(
          (mamba2_faithful || mamba3_enabled) &&
                  ((!is_attn && !is_ttt_layer) ||
                   (hybrid_composition ==
                        HybridComposition::ParallelGated &&
                    is_attn && !is_ttt_layer))
                         ? 1e-5f
                         : 1e-6f),
      layer_idx(li),
      total_layers(tl),
      d_model(dm),
      num_experts(std::max(configured_experts, 1)),
      dropout_rate_(std::clamp(dropout_rate, 0.0f, 0.95f)) {

    learnable_core_norm_ =
        (mamba2_faithful || mamba3_enabled) && !is_ttt_layer &&
        (!is_attn || hybrid_parallel_);
    if (learnable_core_norm_) {
        core_norm_weight_ = Parameter(
            Tensor::ones({dm}, Device::CPU),
            hybrid_parallel_ ? "mamba.pre_norm.weight" : "norm.weight");
    }
    learnable_attention_norm_ =
        (mamba2_faithful || mamba3_enabled) && hybrid_parallel_;
    learnable_ff_norm_ =
        (mamba2_faithful || mamba3_enabled) && hybrid_parallel_;
    if (learnable_attention_norm_) {
        attention_norm_weight_ = Parameter(
            Tensor::ones({dm}, Device::CPU), "attn.pre_norm.weight");
    }
    if (learnable_ff_norm_) {
        ff_norm_weight_ = Parameter(
            Tensor::ones({dm}, Device::CPU), "ffn.pre_norm.weight");
    }
    if (hybrid_parallel_) {
        mamba_gate_ = Parameter(
            Tensor::ones({dm}, Device::CPU)
                .mul(hybrid_mamba_gate_init),
            "mamba.gate");
        attention_gate_ = Parameter(
            Tensor::ones({dm}, Device::CPU)
                .mul(hybrid_attention_gate_init),
            "attn.gate");
        ffn_gate_ = Parameter(
            Tensor::ones({dm}, Device::CPU)
                .mul(hybrid_ffn_gate_init),
            "ffn.gate");
    }

    // Deterministic LayerScale on faithful legacy replacement-attention blocks.
    // Its fixed initialization is part of the architecture contract.
    use_attn_layerscale_ =
        (mamba2_faithful || mamba3_enabled) && is_attn && !hybrid_parallel_;
    if (use_attn_layerscale_) {
        // Validated default: gamma init 0.01 makes terminal attention SOLVE MQAR
        // n_kv=8 (acc 0.97, loss->0.10 on the 1050 A/B), vs 0.35 at init 0.1 —
        // starting attention very quiet lets Mamba establish recall while gamma
        // grows slowly (standard small-LayerScale-init wisdom).
        attn_layerscale_ = Parameter(
            Tensor::ones({dm}, Device::CPU).mul(0.01f), "attn.layerscale");
    }

    // CHRASS slot (parallel with FFN/MoE).  Each layer gets a distinct
    // random adjacency derived from (chrass_seed + layer_idx).  Self-loops
    // excluded; weights uniform [-1,1]; row-normalized inside ctor.
    if (use_chrass && dm > 1) {
        const float density = std::clamp(chrass_density, 0.0f, 1.0f);
        const uint32_t layer_seed = chrass_seed + static_cast<uint32_t>(li);
        auto adj = ChrassLayer::random_adjacency(dm, density, layer_seed);
        chrass_layer = std::make_unique<ChrassLayer>(dm, adj);
    }

    // Two distinct hidden dimensions kept separate so the Nemotron K·m
    // override only affects MoE experts (its semantic scope), not TTT or
    // regular FFN layers.
    //
    // - default_ffn_hidden: historical dm * 4 used by TTT, regular FFN, and
    //   as the fallback for MoE experts when no override is set.
    //
    // - moe_expert_hidden: only differs from default when the user enables
    //   the Nemotron K·m invariant tuning by setting
    //   ModelConfig::moe_expert_hidden_dim > 0.  In that case we use the
    //   override ONLY for MoE expert BitLinear sizing.
    //
    // This separation matters because the field is named "moe_expert_hidden_dim"
    // and the paper (Nemotron 3 Super, Sec 2.1.1) defines m as the MoE
    // expert FFN intermediate dimension specifically — it is not a general
    // FFN-width knob.  See OXN/nsos/docs/NEMOTRON_KM_INTEGRATION.md.
    const int default_ffn_hidden = dm * 4;
    const int moe_expert_hidden = (configured_expert_hidden_dim > 0)
                                  ? configured_expert_hidden_dim
                                  : default_ffn_hidden;

    if (is_ttt) {
        ttt_layer = std::make_unique<TTTLayer>(dm, default_ffn_hidden);
    } else {
        if (is_attention) {
            attn_layer = std::make_unique<Attention>(
                dm, query_heads, 512, kv_heads, rope_theta,
                sliding_window);
            attn_layer->set_exact_training_path(
                exact_attention_training);
            attn_layer->set_exact_linear_mode(
                faithful_attention_linears);
        }
        if (!is_attention || hybrid_parallel_) {
        if (mamba3_enabled) {
            mamba3_layer = std::make_unique<Mamba3Layer>(dm, mamba3_config);
        } else {
        MambaConfig config;
        config.recompute_ssd = use_gradient_checkpointing;
        config.save_intermediates = !use_gradient_checkpointing;
        config.max_seq_for_storage = max_context_tokens;
        // K1: corrected selective SSM (independent delta/B/C/z projections +
        // causal conv1d + single-C linear readout + SiLU gate) is now the
        // DEFAULT, driven by ModelConfig (mamba_proper_ssm / state_expansion).
        // The legacy degenerate path is reachable only by explicitly setting the
        // config flag false (to reload a pre-correction checkpoint).
        config.proper_selective_ssm = mamba_proper_ssm;
        config.proper_state_expansion = mamba_state_expansion && mamba_proper_ssm;
        config.conv_kernel = mamba_conv_kernel;
        config.faithful_mamba2 = mamba2_faithful;
        config.expand = mamba_expand;
        const int inner = config.expand * dm;
        config.head_dim = mamba_head_dim;
        config.n_groups = mamba_n_groups;
        config.out_proj_init_scale =
            1.0f / std::sqrt(static_cast<float>(std::max(tl, 1)));
        // d_state (N) is capped at the N-state GPU kernel's MAX_N (=64, see
        // src/cuda/mamba_kernels.cu) so the full SSD scan stays on the device
        // fast path instead of the host fallback — critical now that the N-state
        // SSD is the default (K1).  64 is also the standard Mamba-2 head-state
        // size; the old dm/2 (e.g. 160) was both slower and non-standard.
        mamba_layer = std::make_unique<Mamba2SSD>(dm, mamba_d_state,
                                                  std::max(dm / 16, 1), config);
        }
        }
    }

    if (is_moe) {
        router = std::make_unique<MoERouter>(
            dm, num_experts, configured_top_k);
        for (int j = 0; j < num_experts; ++j) {
            expert_gate_up.push_back(std::make_unique<BitLinear>(dm, moe_expert_hidden));
            expert_down.push_back(std::make_unique<BitLinear>(moe_expert_hidden, dm));
            expert_gate_up.back()->track_gradient_contributions();
            expert_down.back()->track_gradient_contributions();
        }
    } else if (use_kan) {
        // KAN FFN: a single Kolmogorov-Arnold layer (learnable RBF activations)
        // replaces the dense gate-up -> squared-ReLU -> down FFN.
        kan_ffn = std::make_unique<BitFastKANLayer>(dm, dm);
        // N5: honor the 1.58-bit invariant — the model's KAN FFN uses ternary
        // fake-quant (STE) weights like every other BitLinear.
        kan_ffn->set_quantized(true);
    } else if (!faithful_mamba_core_only_) {
        ffn_gate_up = std::make_unique<BitLinear>(dm, default_ffn_hidden);
        ffn_down = std::make_unique<BitLinear>(default_ffn_hidden, dm);
        if (faithful_attention_linears && is_attention) {
            ffn_gate_up->set_exact_linear_mode(true);
            ffn_down->set_exact_linear_mode(true);
        }
    }
}

JambaBlock::~JambaBlock() = default;

std::string JambaBlock::audit_block_type() const {
    std::string type;
    if (hybrid_parallel_) {
        type = "mamba2+attention";
    } else if (is_ttt) {
        type = "ttt";
    } else if (is_attention) {
        type = "attention";
    } else {
        type = "mamba2";
    }
    type += is_moe ? "+moe"
                   : (faithful_mamba_core_only_ ? "+core" : "+ffn");
    return type;
}

Tensor JambaBlock::mamba_forward(const Tensor& input, Context* context) {
    if(mamba3_layer) {
        auto valid=active_batch_valid_lengths_;
        const int B=input.shape.size()==3?input.shape[0]:1;
        const int S=input.shape.size()==3?input.shape[1]:input.shape[0];
        if(mamba3_layer->streaming_mode() && S==1) valid.assign(B,1);
        return mamba3_layer->forward(input,context,valid);
    }
    return mamba_layer->forward(input,context);
}
Tensor JambaBlock::mamba_backward(const Tensor& gradient, Context& context) {
    return mamba3_layer ? mamba3_layer->backward(gradient,context) : mamba_layer->backward(gradient,context);
}
Tensor JambaBlock::forward(const Tensor& x, Context* ctx) {
    const bool audit_enabled =
        audit_collector_ && audit_collector_->enabled();
    AuditLatencyTimer audit_timer(audit_enabled, x.get_device());
    auto scale_ff = [&](const Tensor& ff_in) -> Tensor {
        if (hybrid_parallel_) {
            saved_ls_ff_ = ff_in;
            return ff_in.mul(ffn_gate_.data);
        }
        if (use_attn_layerscale_) {
            saved_ls_ff_ = ff_in;  // pre-scale, for gamma grad
            return ff_in.mul(attn_layerscale_.data);
        }
        return ff_in;
    };
    last_batch_size_ = x.shape.size() == 3 ? x.shape[0] : 0;
    saved_input_ = x;
    saved_core_rms_ = x.rmsnorm(core_norm_eps_);
    saved_core_norm_ =
        learnable_core_norm_
            ? saved_core_rms_.mul(core_norm_weight_.data)
            : saved_core_rms_;
    // Clear last forward's dropout masks so backward never reapplies stale ones.
    saved_drop_core_ = Tensor();
    saved_drop_mamba_ = Tensor();
    saved_drop_attention_ = Tensor();
    saved_drop_moe_ = Tensor();
    saved_drop_ff_hidden_ = Tensor();
    saved_drop_ff_out_ = Tensor();
    saved_ls_core_ = Tensor();
    saved_ls_ff_ = Tensor();
    saved_mamba_core_ = Tensor();
    saved_attention_core_ = Tensor();
    Tensor core;
    Tensor core_contrib;
    Tensor hybrid_mamba_contribution;
    Tensor hybrid_attention_contribution;
    if (hybrid_parallel_) {
        if ((!mamba_layer && !mamba3_layer) || !attn_layer ||
            mamba_gate_.data.size == 0 ||
            attention_gate_.data.size == 0) {
            throw std::logic_error(
                "parallel hybrid block is missing a required branch or gate");
        }
        saved_attention_rms_ = saved_core_rms_;
        saved_attention_norm_ =
            learnable_attention_norm_
                ? saved_attention_rms_.mul(attention_norm_weight_.data)
                : saved_attention_rms_;
        Tensor mamba_core = mamba_forward(saved_core_norm_, ctx);
        Tensor attention_core =
            attn_layer->forward(saved_attention_norm_, ctx);
        if (training_mode_ && dropout_rate_ > 1e-6f) {
            mamba_core = apply_training_dropout(
                mamba_core, dropout_rate_,
                "jamba_mamba_" + std::to_string(layer_idx),
                layer_idx * 31 + 1, dropout_sequence_,
                &saved_drop_mamba_);
            attention_core = apply_training_dropout(
                attention_core, dropout_rate_,
                "jamba_attention_" + std::to_string(layer_idx),
                layer_idx * 31 + 2, dropout_sequence_,
                &saved_drop_attention_);
        }
        saved_mamba_core_ = mamba_core;
        saved_attention_core_ = attention_core;
        hybrid_mamba_contribution =
            mamba_core.mul(mamba_gate_.data);
        hybrid_attention_contribution =
            attention_core.mul(attention_gate_.data);
        core_contrib =
            hybrid_mamba_contribution.add(
                hybrid_attention_contribution);
    } else if (is_ttt) {
        core = ttt_layer->forward(saved_core_norm_);
    } else if (is_attention) {
        core = attn_layer->forward(saved_core_norm_, ctx);
    } else {
        core = mamba_forward(saved_core_norm_, ctx);
    }
    if (!hybrid_parallel_ &&
        training_mode_ && dropout_rate_ > 1e-6f) {
        core = apply_training_dropout(core, dropout_rate_,
                                      "jamba_core_" + std::to_string(layer_idx),
                                      layer_idx * 17 + 1, dropout_sequence_,
                                      &saved_drop_core_);
    }

    if (!hybrid_parallel_) {
        core_contrib = core;
        if (use_attn_layerscale_) {
            saved_ls_core_ = core;  // pre-scale, for gamma grad
            core_contrib = core.mul(attn_layerscale_.data);
        }
    }
    saved_residual_ = x.add(core_contrib);
    const auto record_hybrid_forward =
        [&](const Tensor& ffn_signal,
            const Tensor& ffn_contribution) {
            if (!audit_enabled || !hybrid_parallel_) {
                return;
            }
            audit_collector_->record_hybrid_interaction(
                layer_idx, "forward",
                saved_mamba_core_, saved_attention_core_,
                ffn_signal,
                hybrid_mamba_contribution,
                hybrid_attention_contribution,
                ffn_contribution);
        };

    // A faithful Mamba-only block has no FFN by contract. Return before
    // computing/saving an RMSNorm that no forward or backward consumer uses.
    // This removes one full-tensor normalization kernel and its allocations
    // from every pure-Mamba layer (the dominant product/training topology).
    if (faithful_mamba_core_only_) {
        saved_ff_rms_ = Tensor();
        saved_ff_norm_ = Tensor();
        saved_ff_hidden_pre_ = Tensor();
        if (audit_enabled) {
            const double latency_ms = audit_timer.elapsed_ms();
            audit_collector_->record_forward(
                layer_idx, audit_block_type(), "activation", x,
                saved_residual_, latency_ms);
        }
        return saved_residual_;
    }

    saved_ff_rms_ = saved_residual_.rmsnorm(
        hybrid_parallel_ ? 1e-5f : 1e-6f);
    saved_ff_norm_ =
        learnable_ff_norm_
            ? saved_ff_rms_.mul(ff_norm_weight_.data)
            : saved_ff_rms_;

    if (is_moe) {
        Tensor ff = forward_moe(saved_ff_norm_, ctx, "L" + std::to_string(layer_idx));
        if (training_mode_ && dropout_rate_ > 1e-6f) {
            ff = apply_training_dropout(ff, dropout_rate_,
                                        "jamba_moe_" + std::to_string(layer_idx),
                                        layer_idx * 17 + 2, dropout_sequence_,
                                        &saved_drop_moe_);
        }
        // ── CHRASS parallel slot (after FFN dropout, before residual add) ──
        if (chrass_layer && saved_ff_norm_.size > 0) {
            const auto& s = saved_ff_norm_.shape;
            int total_lead = 1;
            std::vector<int> orig_shape;
            for (size_t i = 0; i < s.size(); ++i) {
                orig_shape.push_back(s[i]);
                if (i + 1 < s.size()) total_lead *= s[i];
            }
            Tensor flat = saved_ff_norm_.reshape({total_lead, orig_shape.back()});
            Tensor c_out = chrass_layer->forward(flat);
            Tensor c_unflat = c_out.reshape(orig_shape);
            ff = ff.add(c_unflat);
        }
        Tensor ff_contribution = scale_ff(ff);
        Tensor output = saved_residual_.add(ff_contribution);
        if (audit_enabled) {
            record_hybrid_forward(ff, ff_contribution);
            const double latency_ms = audit_timer.elapsed_ms();
            audit_collector_->record_forward(layer_idx, audit_block_type(), "activation",
                                             x, output, latency_ms);
        }
        return output;
    }

    Tensor ff;
    if (kan_ffn) {
        // KAN FFN: a single Kolmogorov-Arnold layer replaces the dense
        // gate-up -> squared-ReLU -> down path (handles rank-3 natively).
        ff = kan_ffn->forward(saved_ff_norm_);
    } else {
        saved_ff_hidden_pre_ = ffn_gate_up->forward(saved_ff_norm_);
        // LEARN S1: squared ReLU (BitNet b1.58 2B4T) instead of plain ReLU.
        // Numerically stable in quantized regimes, comparable expressivity
        // to SwiGLU at moderate scale (1-2B), avoids SwiGLU's FP8/ternary
        // spike-overflow failure mode (Welleck et al., BitNet 2B4T TR 2026).
        Tensor ff_hidden = saved_ff_hidden_pre_.squared_relu();
        if (training_mode_ && dropout_rate_ > 1e-6f) {
            ff_hidden = apply_training_dropout(ff_hidden, dropout_rate_ * 0.5f,
                                               "jamba_ff_hidden_" + std::to_string(layer_idx),
                                               layer_idx * 17 + 3, dropout_sequence_,
                                               &saved_drop_ff_hidden_);
        }
        ff = ffn_down->forward(ff_hidden);
    }
    if (training_mode_ && dropout_rate_ > 1e-6f) {
        ff = apply_training_dropout(ff, dropout_rate_,
                                    "jamba_ff_out_" + std::to_string(layer_idx),
                                    layer_idx * 17 + 4, dropout_sequence_,
                                    &saved_drop_ff_out_);
    }
    // ── CHRASS parallel slot (FFN path) ──
    if (chrass_layer && saved_ff_norm_.size > 0) {
        const auto& s = saved_ff_norm_.shape;
        int total_lead = 1;
        std::vector<int> orig_shape;
        for (size_t i = 0; i < s.size(); ++i) {
            orig_shape.push_back(s[i]);
            if (i + 1 < s.size()) total_lead *= s[i];
        }
        Tensor flat = saved_ff_norm_.reshape({total_lead, orig_shape.back()});
        Tensor c_out = chrass_layer->forward(flat);
        Tensor c_unflat = c_out.reshape(orig_shape);
        ff = ff.add(c_unflat);
    }
    Tensor ff_contribution = scale_ff(ff);
    Tensor output = saved_residual_.add(ff_contribution);
    if (audit_enabled) {
        record_hybrid_forward(ff, ff_contribution);
        const double latency_ms = audit_timer.elapsed_ms();
        audit_collector_->record_forward(layer_idx, audit_block_type(), "activation",
                                         x, output, latency_ms);
    }
    return output;
}

static bool moe_router_grad_enabled();

#ifdef USE_CUDA
Tensor JambaBlock::forward_moe_gpu_sparse_decode(const Tensor& x,
                                               const Tensor& weights, int dim) {
#ifdef USE_CUDA
    const char* requested = std::getenv("NSOS_GPU_SPARSE_MOE");
    if (requested && requested[0] == '0') return Tensor();
    const int rows = static_cast<int>(x.size / dim);
    std::vector<GpuLinearView> views(static_cast<size_t>(num_experts) * 2);
    int hidden_dim = 0;
    for (int e = 0; e < num_experts; ++e) {
        auto& up = views[static_cast<size_t>(e)];
        auto& down = views[static_cast<size_t>(num_experts + e)];
        if (!expert_gate_up[e]->prepare_gpu_decode_view(up) ||
            !expert_down[e]->prepare_gpu_decode_view(down)) return Tensor();
        if (e == 0) hidden_dim = up.outputs;
        if (up.inputs != dim || up.outputs != hidden_dim ||
            down.inputs != hidden_dim || down.outputs != dim) return Tensor();
    }
    if (views != moe_decode_host_views_) {
        const size_t bytes = views.size() * sizeof(GpuLinearView);
        auto* destination = moe_decode_views_.ensure(views.size());
        if (!destination) throw std::bad_alloc();
        gpu_runtime_check(cudaMemcpy(destination, views.data(), bytes,
                                    cudaMemcpyHostToDevice), "MoE descriptor upload");
        moe_decode_host_views_ = views;
    }
    auto& workspace = gpu::current_execution_context();
    auto reserve = [&](gpu::WorkspaceSlot slot, size_t elements) {
        return static_cast<float*>(workspace.reserve(slot, gpu::StorageType::Float32, elements));
    };
    const size_t slots = size_t(num_experts) * rows;
    auto* prepared = reserve(gpu::WorkspaceSlot::MoePrepared, slots * std::max(dim, hidden_dim));
    auto* scales = reserve(gpu::WorkspaceSlot::MoeScales, slots);
    auto* hidden = reserve(gpu::WorkspaceSlot::MoeHidden, slots * hidden_dim);
    auto* contributions = reserve(gpu::WorkspaceSlot::MoeContributions, slots * dim);
    Tensor output = Tensor::uninitialized({rows, dim}, Device::GPU);
    launch_moe_sparse_decode(moe_decode_views_.get(),
        weights.raw_data(), x.raw_data(), prepared, scales,
        hidden, contributions, output.raw_data(), num_experts, dim, hidden_dim, rows);
    gpu::record_dispatch(gpu::DispatchPath::SparseMoe);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("sparse MoE decode: ") + cudaGetErrorString(status));
    return output.reshape(x.shape.dims);
#else
    (void)x; (void)weights; (void)dim;
    return Tensor();
#endif
}

Tensor JambaBlock::forward_moe_gpu_batched(const Tensor& x,
                                            const Tensor& weights,
                                            int rows, int dim,
                                            int effective_top_k) {
    const bool ordered = training_policy::ordered_moe();
    saved_moe_ordered_device_ = ordered;
    // Pre-condition (verified by caller):
    //   * x and weights are on GPU
    //   * gpu_custom_kernels_supported() == true
    //   * rows > 0, num_experts > 0, dim > 0
    //   * num_experts <= 1024 (scan kernel constraint)
    //
    // Pipeline (all device side, single CUDA default-stream chain):
    //   1. counts[E] = nonzero entries per column of weights
    //   2. offsets[E+1] = exclusive_scan(counts);
    //      offsets[E] is the total number of (row, expert) active pairs.
    //   3. permutation[N_active] = source row id, sorted by expert
    //      assignment[N_active] = expert id at slot k
    //      scale[N_active]      = router weight at slot k
    //   4. permuted_input[N_active, dim] = gather x[permutation[k], :]
    //   5. for each expert e: process permuted_input[off[e]:off[e+1]]
    //      and write into permuted_output[off[e]:off[e+1]]
    //   6. y[rows, dim] = scatter-add scale[k] * permuted_output[k, :]
    //      back into y[permutation[k], :].
    //
    // The std::memcpy gather/scatter loops in the CPU path are replaced
    // by 3 launch_moe_* kernels.  Per-expert sequencing in step 5 is
    // unavoidable here because experts have different weight tensors;
    // a fully batched expert matmul would require concatenated weights
    // and a block-sparse GEMM, which is left for a future PR.

    // Pull the persistent MoE workspace — counts/offsets/workspace_counters/
    // permutation/assignment/scale buffers are sized once and reused
    // forever.  Was 6 cudaMalloc + 6 cudaFree per call (= ~10-30 ms of
    // pure allocator overhead).
    MoeWorkspace& moe_ws = moe_workspace();
    int* counts_ptr  = moe_ws.counts(num_experts);
    int* offsets_ptr = moe_ws.offsets(num_experts);
    gpu_runtime_check(
        cudaMemsetAsync(counts_ptr, 0,
                   static_cast<size_t>(num_experts) * sizeof(int), gpu::current_stream()),
        "MoE expert-count reset");
    launch_moe_count_per_expert_kernel(weights.raw_data(), counts_ptr,
                                        rows, num_experts);
    launch_moe_exclusive_scan_small_kernel(counts_ptr, offsets_ptr,
                                            num_experts);

    // Pull offsets to host once: counts are exactly adjacent differences in
    // the exclusive-scan output. Downloading counts separately caused a
    // second blocking D2H transaction without carrying any new information.
    std::vector<int> counts_host(static_cast<size_t>(num_experts), 0);
    std::vector<int> offsets_host(static_cast<size_t>(num_experts + 1), 0);
    gpu_runtime_check(
        cudaMemcpy(offsets_host.data(), offsets_ptr,
                   static_cast<size_t>(num_experts + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "MoE expert-offset download");
    record_gpu_transfer(
        Device::CPU, Device::GPU,
        static_cast<size_t>(num_experts + 1) * sizeof(int));
    const int N_active = offsets_host[static_cast<size_t>(num_experts)];
    const int64_t maximum_active =
        static_cast<int64_t>(rows) * num_experts;
    if (offsets_host.front() != 0 || N_active < 0 ||
        static_cast<int64_t>(N_active) > maximum_active) {
        throw std::runtime_error(
            "MoE exclusive-scan offsets violate active-assignment bounds");
    }
    for (int expert = 0; expert < num_experts; ++expert) {
        const int begin = offsets_host[static_cast<size_t>(expert)];
        const int end = offsets_host[static_cast<size_t>(expert + 1)];
        if (begin < 0 || end < begin || end > N_active) {
            throw std::runtime_error(
                "MoE exclusive-scan offsets are not monotonic");
        }
        counts_host[static_cast<size_t>(expert)] = end - begin;
    }

    // AUDIT #4+#5 (2026-05-16): host-side reconstruction was the
    // dominant CPU cost in the MoE forward path.  The old code did:
    //   1. cpu() copy of the entire weights tensor (D2H sync, full data)
    //   2. partial_sort over each of `rows` rows × num_experts entries,
    //      O(rows × num_experts × log num_experts) host work
    //   3. push_back per expert (heap allocations)
    // This is now replaced by a single D2H copy of permutation_buf
    // (N_active ints, typically much smaller than rows × num_experts
    // entries since effective_top_k <= num_experts), then an O(N_active)
    // linear scan to slot row ids into per-expert buckets.
    //
    // We also defer the weights.cpu() copy: only the CPU backward path
    // needs saved_moe_weights_, and even that path can reconstruct from
    // the GPU tensor on demand.  We keep weights_host for the audit
    // hook when audit is enabled, since the audit reads expert_loads
    // which router stores already on host.
    saved_moe_weights_ = weights;          // Hold the GPU tensor handle —
                                            // backward syncs only when CPU path actually runs.
    saved_moe_n_active_ = N_active;
    saved_moe_counts_host_ = counts_host;
    saved_moe_offsets_host_ = offsets_host;
    saved_moe_rows_.clear();
    saved_moe_scale_device_ = Tensor();
    saved_moe_permuted_output_ = Tensor();
    // The permutation buffer is populated later by
    // launch_moe_compute_assignments_kernel.  We defer the D2H of
    // permutation until AFTER that kernel runs — see below.

    if (audit_collector_ && audit_collector_->enabled()) {
        std::vector<int> topk_counts(static_cast<size_t>(num_experts), 0);
        router->materialize_expert_loads();
        for (int e = 0; e < num_experts; ++e) {
            topk_counts[static_cast<size_t>(e)] = counts_host[static_cast<size_t>(e)];
        }
        audit_collector_->record_router(layer_idx,
                                        audit_block_type(),
                                        rows,
                                        num_experts,
                                        effective_top_k,
                                        topk_counts,
                                        router->expert_loads);
    }

    Tensor output_accum = ordered && N_active > 0
        ? Tensor::uninitialized({rows, dim}, Device::GPU)
        : Tensor::zeros({rows, dim}, Device::GPU);
    if (N_active <= 0) {
        return output_accum.reshape(x.shape.dims);
    }

    if (ordered) {
        auto* owner_offsets = saved_moe_offsets_device_.ensure(num_experts + 1);
        if (!owner_offsets) throw std::runtime_error("MoE ordered offset allocation failed");
        gpu_runtime_check(cudaMemcpyAsync(owner_offsets, offsets_ptr,
            static_cast<size_t>(num_experts + 1) * sizeof(int), cudaMemcpyDeviceToDevice,
            gpu::current_stream()), "MoE ordered saved offsets");
        record_gpu_transfer(Device::GPU, Device::GPU, static_cast<size_t>(num_experts + 1) * sizeof(int));
    }
    // Workspace counters for compute_assignments (atomicAdd target).
    int* permutation_ptr = nullptr;
    float* scale_ptr = nullptr;

    if (!ordered) {
        int* workspace_counters_ptr = moe_ws.workspace_counters(num_experts);
        gpu_runtime_check(cudaMemsetAsync(workspace_counters_ptr, 0,
            static_cast<size_t>(num_experts)*sizeof(int), gpu::current_stream()),
            "MoE workspace-counter reset");
        permutation_ptr = moe_ws.permutation(N_active);
        int* assignment_ptr = moe_ws.assignment(N_active);
        scale_ptr = moe_ws.scale(N_active);
        launch_moe_compute_assignments_kernel(
            weights.raw_data(), offsets_ptr, workspace_counters_ptr,
            permutation_ptr, assignment_ptr, scale_ptr, rows, num_experts);
    }

    // Retain per-token routing state per block on the device. The persistent
    // integer allocation grows geometrically; scales use the Tensor pool.
    int* saved_permutation =
        saved_moe_permutation_device_.ensure(
            static_cast<size_t>(N_active));
    if (!saved_permutation) {
        throw std::runtime_error(
            "MoE saved permutation allocation failed");
    }
    if (ordered) {
        saved_moe_scale_device_ = Tensor::uninitialized({N_active}, Device::GPU);
        auto* inverse = saved_moe_inverse_device_.ensure(static_cast<size_t>(rows) * num_experts);
        if (!inverse) throw std::runtime_error("MoE ordered inverse allocation failed");
        launch_moe_ordered_assign(weights.raw_data(), offsets_ptr, saved_permutation,
            saved_moe_scale_device_.raw_data(), inverse, rows, num_experts);
        permutation_ptr = saved_permutation;
        scale_ptr = saved_moe_scale_device_.raw_data();
    } else {
    const cudaError_t permutation_copy_status = cudaMemcpy(
        saved_permutation, permutation_ptr,
        static_cast<size_t>(N_active) * sizeof(int), cudaMemcpyDeviceToDevice);
    if (permutation_copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("MoE saved permutation copy failed: ") +
            cudaGetErrorString(permutation_copy_status));
    }
    record_gpu_transfer(
        Device::GPU, Device::GPU,
        static_cast<size_t>(N_active) * sizeof(int));
    saved_moe_scale_device_ = Tensor({N_active}, Device::GPU);
    copy_tensor_bytes(saved_moe_scale_device_.raw_data(), Device::GPU,
                      scale_ptr, Device::GPU,
                      static_cast<size_t>(N_active) * sizeof(float));
    }

    // Permuted input: contiguous by expert.
    Tensor permuted_input = Tensor::uninitialized({N_active, dim}, Device::GPU);
    launch_moe_gather_rows_kernel(x.raw_data(), permutation_ptr,
                                   permuted_input.raw_data(), N_active, dim);

    // AUDIT #4+#5: single D2H of permutation_buf, then O(N_active)
    // population of saved_moe_rows_.  This replaces the old O(rows ×
    // num_experts × log num_experts) host-side partial_sort.
    // Per-expert forward into a permuted_output buffer.  Slices of
    // permuted_input are computed cheaply (Tensor::slice is a view in
    // the current API) and the BitLinear forwards already use the GPU
    // path on x.get_device() == GPU.
    Tensor permuted_output = Tensor::uninitialized({N_active, dim}, Device::GPU);
    // LEARN S1: pre-allocate per-expert pre-activation cache for the
    // GPU batched path.  Same semantics as the CPU path — backward
    // uses these for correct squared_relu_backward chain rule.
    saved_moe_pre_activations_.assign(static_cast<size_t>(num_experts), Tensor());
    for (int e = 0; e < num_experts; ++e) {
        const int count = counts_host[static_cast<size_t>(e)];
        if (count <= 0) continue;
        const int offset = offsets_host[static_cast<size_t>(e)];
        Tensor expert_input = permuted_input.slice(0, offset, offset + count);
        // LEARN S1: squared ReLU (BitNet b1.58 2B4T) — same activation
        // used in the non-MoE FFN path above for consistency and the
        // numerical-stability reasons documented there.  We save the
        // pre-activation (output of expert_gate_up, BEFORE squared_relu)
        // so the backward pass can apply the correct chain rule.
        Tensor expert_pre_activation = expert_gate_up[e]->forward(expert_input);
        saved_moe_pre_activations_[static_cast<size_t>(e)] = expert_pre_activation;
        Tensor expert_hidden = expert_pre_activation.squared_relu();
        Tensor expert_out = expert_down[e]->forward(expert_hidden);
        // Copy expert_out into permuted_output[offset:offset+count, :].
        // expert_out is on GPU; do a contiguous device-to-device memcpy
        // into the right slice.
        gpu_runtime_check(
            cudaMemcpyAsync(
                permuted_output.raw_data() + static_cast<size_t>(offset) *
                                                  static_cast<size_t>(dim),
                expert_out.raw_data(),
                static_cast<size_t>(count) * static_cast<size_t>(dim) *
                    sizeof(float),
                cudaMemcpyDeviceToDevice, gpu::current_stream()),
            "MoE expert output copy");
        record_gpu_transfer(
            Device::GPU, Device::GPU,
            static_cast<size_t>(count) * static_cast<size_t>(dim) *
                sizeof(float));
    }

    if (ordered) {
        launch_moe_ordered_combine(permuted_output.raw_data(), saved_moe_inverse_device_.get(),
            scale_ptr, output_accum.raw_data(), rows, dim, num_experts);
    } else {
        launch_moe_scatter_add_weighted_kernel(permuted_output.raw_data(), permutation_ptr,
            scale_ptr, output_accum.raw_data(), N_active, dim);
    }
    if (training_mode_ && moe_router_grad_enabled()) {
        saved_moe_permuted_output_ = permuted_output;
    }
    gpu_runtime_check(cudaGetLastError(), "MoE batched forward kernels");

    return output_accum.reshape(x.shape.dims);
}
#endif  // USE_CUDA

static bool moe_router_grad_enabled() {
    // K2: task→router gradient is now ON by default (the router must learn from
    // the task loss, not only the load-balancing aux).  NSOS_MOE_ROUTER_GRAD=0
    // disables it (e.g. to reload/compare a checkpoint trained without it).
    static const bool on = [] {
        const char* e = std::getenv("NSOS_MOE_ROUTER_GRAD");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

Tensor JambaBlock::forward_moe(const Tensor& x, Context* ctx, const std::string& ln) {
    if (training_mode_ && training_policy::moe_wmma_training() && !training_policy::grouped_moe_training())
        throw std::invalid_argument("NSOS_MOE_WMMA_TRAINING requires grouped MoE training; no silent ignored policy");
    if(training_mode_ && optimizer_policy::device_sparse_adam_enabled() &&
        (!training_policy::grouped_moe_training() || !training_policy::ordered_moe() ||
         expert_gate_up.empty() || !expert_gate_up.front()->weight.has_device_gradient_activity()))
        throw std::logic_error("Device sparse MoE policy requires a Trainer accumulation domain");
    saved_moe_ordered_device_ = false;
    saved_moe_grouped_training_ = false;
    if (grouped_moe_training_) grouped_moe_training_->discard_backward_state();
    (void)ctx;
    (void)ln;
    if (!router || expert_gate_up.empty()) {
        return make_zero_like(x);
    }

    const std::vector<uint8_t> valid_rows =
        flattened_valid_row_mask(x, active_batch_valid_lengths_);
    auto [unused_logits, weights] = router->forward(x, valid_rows);
    (void)unused_logits;

    const int rows = x.size / x.shape.back();
    const int dim = x.shape.back();
    const Device target_device = x.get_device();
    // Pacote A.1: at inference (training_mode_=false), an explicit
    // override > 0 wins over router->top_k.  This is how we cut MoE
    // expert FLOPs to top-1 during decode without retraining.  During
    // training the override is ignored — gradients still use the same
    // top-k the router was trained with.
    int router_top_k = router->top_k;
    if (!training_mode_ && inference_top_k_override_ > 0) {
        router_top_k = inference_top_k_override_;
    }
    const int effective_top_k = std::clamp(router_top_k, 1, num_experts);

    // N2/N3 (Pacote A.1 parity): when an inference override REDUCES k below the
    // router's trained top_k, router->forward already masked + renormalized the
    // weights over the LARGER top_k set, so the surviving weights no longer sum
    // to 1 over the (smaller) effective set — top-1 decode would attenuate the
    // expert output (~router prob, e.g. 0.6 instead of 1.0).  Re-mask +
    // renormalize to effective_top_k here, device-aware and in place, so BOTH
    // the CPU and GPU-batched paths dispatch and scale by weights that sum to 1
    // over exactly the experts they use (also keeps CPU/GPU byte-consistent).
    // Only fires on the override path (effective_top_k < router->top_k) at
    // inference; the training/normal path is untouched (byte-identical).
    if (!training_mode_ && effective_top_k < router->top_k && rows > 0) {
#ifdef USE_CUDA
        if (weights.get_device() == Device::GPU && gpu_custom_kernels_supported()) {
            launch_moe_topk_mask_kernel(weights.raw_data(), rows, num_experts,
                                        effective_top_k);
        } else
#endif
        {
            Tensor wh = weights.get_device() == Device::GPU ? weights.cpu() : weights;
            float* wp = wh.data();
            std::vector<int> ranked(static_cast<size_t>(num_experts));
            std::vector<char> keep(static_cast<size_t>(num_experts), 0);
            for (int r = 0; r < rows; ++r) {
                std::iota(ranked.begin(), ranked.end(), 0);
                std::partial_sort(
                    ranked.begin(), ranked.begin() + effective_top_k, ranked.end(),
                    [&](int a, int b) {
                        return wp[r * num_experts + a] > wp[r * num_experts + b];
                    });
                std::fill(keep.begin(), keep.end(), 0);
                float s = 0.0f;
                for (int t = 0; t < effective_top_k; ++t) {
                    keep[static_cast<size_t>(ranked[static_cast<size_t>(t)])] = 1;
                    s += wp[r * num_experts + ranked[static_cast<size_t>(t)]];
                }
                const float inv = 1.0f / std::max(s, 1e-9f);
                for (int e = 0; e < num_experts; ++e) {
                    wp[r * num_experts + e] =
                        keep[static_cast<size_t>(e)] ? wp[r * num_experts + e] * inv : 0.0f;
                }
            }
            if (weights.get_device() == Device::GPU) {
                weights.copy_from(wh.to(Device::GPU));
            } else {
                weights = wh;
            }
        }
    }

    if (training_mode_ && training_policy::grouped_moe_training()) {
        if (target_device != Device::GPU || !gpu_custom_kernels_supported() ||
            !training_policy::ordered_moe())
            throw std::invalid_argument("Grouped MoE training requires GPU kernels and ordered routing");
        if (!grouped_moe_training_) grouped_moe_training_ = std::make_shared<GpuMoeTraining>();
        std::vector<BitLinear*> up, down;
        for (auto& expert : expert_gate_up) up.push_back(expert.get());
        for (auto& expert : expert_down) down.push_back(expert.get());
        Tensor output = grouped_moe_training_->forward(
            x, weights.reshape({rows, num_experts}), up, down, effective_top_k);
        saved_moe_grouped_training_ = true;
        if (audit_collector_ && audit_collector_->enabled()) {
            const auto counts = grouped_moe_training_->materialize_counts();
            router->materialize_expert_loads();
            audit_collector_->record_router(layer_idx, audit_block_type(), rows,
                num_experts, effective_top_k, counts, router->expert_loads);
        }
        return output;
    }

#ifdef USE_CUDA
    // Phase 4-extended GPU batched dispatch.
    // Small decode batches keep routing entirely on device. Prefill retains
    // the grouped-GEMM path: GEMV is not appropriate for a large row count.
    if (!training_mode_ && rows > 0 && rows <= 32 && target_device == Device::GPU &&
        weights.get_device() == Device::GPU && gpu_custom_kernels_supported() &&
        num_experts > 0 && num_experts <= 32 && dim > 0) {
        Tensor sparse = forward_moe_gpu_sparse_decode(x, weights, dim);
        if (sparse.size > 0) return sparse;
    }
    // Eligibility:
    //   * Both input x and routing weights live on the GPU.
    //   * The custom CUDA kernels are supported on this device.
    //   * Input has the canonical [rows, dim] layout (rank 2 or 3).
    //   * num_experts and rows are positive.
    //
    // The batched path replaces the per-expert std::memcpy gather/
    // scatter loops with three on-device kernels (count, gather,
    // scatter-add) and a single pass over the experts that operates
    // on contiguous slices of the permuted input.  This eliminates the
    // ~rows × num_experts host-side bookkeeping that used to dominate
    // forward_moe wall-time at batch_size > 3.
    // Deterministic mode uses the ordered per-expert path below (its scatter is
    // a sequence of non-overlapping row copies summed in fixed expert order),
    // not the batched scatter-add kernel.
    // ── GPU-first single-row decode: dense top-k on the DEVICE ──────────
    // At inference with one row, run ALL experts on it and accumulate with
    // the top-k-masked routing weights read from device memory.  Non-selected
    // experts carry weight 0.0f (masked+renormalized on the device by
    // router->forward / the N2/N3 block above), so the sum is algebraically
    // identical to dispatching only the selected experts — and the per-expert
    // accumulation order (e = 0..E-1) matches the host loop's, so the result
    // is bit-equal.  With num_experts small the extra expert FLOPs on a
    // [1, dim] row are negligible next to what this removes: the per-token
    // weights D2H + host partial_sort (host loop) or counts D2H (batched
    // path).  No host decision depends on device data -> the path is
    // CUDA-graph capturable.
    if (!training_mode_ && rows == 1 && target_device == Device::GPU &&
        weights.get_device() == Device::GPU &&
        gpu_custom_kernels_supported() && num_experts > 0 &&
        num_experts <= 32 && dim > 0) {
        Tensor row_input = x.reshape({1, dim});
        Tensor output_accum = Tensor::zeros({1, dim}, Device::GPU);
        for (int e = 0; e < num_experts; ++e) {
            Tensor pre = expert_gate_up[static_cast<size_t>(e)]->forward(row_input);
            Tensor hidden = pre.squared_relu();
            Tensor expert_out = expert_down[static_cast<size_t>(e)]->forward(hidden);
            launch_moe_scale_accum_row_kernel(output_accum.raw_data(),
                                              expert_out.raw_data(),
                                              weights.raw_data() + e, dim);
        }
        return output_accum.reshape(x.shape.dims);
    }

    // moe_router_grad only matters for BACKWARD (it needs the per-expert
    // outputs saved by the host path below); at inference there is no
    // backward, so the router-grad flag must not force decode through the
    // host loop (which costs a weights D2H + per-row host sort every step).
    if (target_device == Device::GPU &&
        weights.get_device() == Device::GPU &&
        gpu_custom_kernels_supported() && rows > 0 && num_experts > 0 &&
        num_experts <= 1024 && dim > 0 &&
        (!determinism::deterministic_reductions_enabled() ||
         training_policy::ordered_moe())) {
        return forward_moe_gpu_batched(x, weights, rows, dim,
                                        effective_top_k);
    }
#endif

    Tensor weights_host = weights.get_device() == Device::GPU ? weights.cpu() : weights;
    Tensor output_accum = Tensor::zeros({rows, dim}, target_device);
    const float* weight_ptr = weights_host.data();
    std::vector<std::vector<int>> expert_rows(static_cast<size_t>(num_experts));
    // LEARN S1: pre-allocate per-expert pre-activation cache.  Each
    // entry stays default-constructed (Tensor of size 0) for experts
    // that don't receive any rows in this forward pass; backward
    // checks size > 0 before applying squared_relu_backward.
    saved_moe_pre_activations_.assign(static_cast<size_t>(num_experts), Tensor());
    if (moe_router_grad_enabled()) {
        saved_moe_expert_out_.assign(static_cast<size_t>(num_experts), Tensor());
    } else {
        saved_moe_expert_out_.clear();
    }
    std::vector<int> ranked_experts(static_cast<size_t>(num_experts));
    std::iota(ranked_experts.begin(), ranked_experts.end(), 0);

    for (int row = 0; row < rows; ++row) {
        std::partial_sort(
            ranked_experts.begin(),
            ranked_experts.begin() + effective_top_k,
            ranked_experts.end(),
            [&](int lhs, int rhs) {
                return weight_ptr[row * num_experts + lhs] >
                       weight_ptr[row * num_experts + rhs];
            });
        for (int rank = 0; rank < effective_top_k; ++rank) {
            expert_rows[static_cast<size_t>(ranked_experts[rank])].push_back(row);
        }
    }

    saved_moe_weights_ = weights_host;
    saved_moe_rows_ = expert_rows;
    if (audit_collector_ && audit_collector_->enabled()) {
        std::vector<int> topk_counts(static_cast<size_t>(num_experts), 0);
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            topk_counts[static_cast<size_t>(expert_idx)] =
                static_cast<int>(expert_rows[static_cast<size_t>(expert_idx)].size());
        }
        audit_collector_->record_router(layer_idx,
                                        audit_block_type(),
                                        rows,
                                        num_experts,
                                        effective_top_k,
                                        topk_counts,
                                        router->expert_loads);
    }

    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const auto& selected_rows = expert_rows[static_cast<size_t>(expert_idx)];
        if (selected_rows.empty()) {
            continue;
        }

        Tensor expert_input({static_cast<int>(selected_rows.size()), dim}, target_device);
        // Routing is host-ordered in deterministic mode, but these operands
        // remain on their original device. Only the device-aware copy helper
        // may consume their addresses; data() would request forbidden host
        // access to cudaMalloc-backed storage.
        float* expert_input_ptr = expert_input.raw_data();
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            copy_tensor_bytes(expert_input_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           x.raw_data() + source_row * dim,
                           x.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }

        // LEARN S1: squared ReLU (BitNet b1.58 2B4T) — see comment in
        // forward_moe_gpu_batched and the non-MoE FFN path.  We save
        // the pre-activation tensor for use by backward (correct chain
        // rule through squared_relu).
        Tensor expert_pre_activation = expert_gate_up[expert_idx]->forward(expert_input);
        saved_moe_pre_activations_[static_cast<size_t>(expert_idx)] = expert_pre_activation;
        Tensor expert_hidden = expert_pre_activation.squared_relu();
        Tensor expert_out = expert_down[expert_idx]->forward(expert_hidden);
        if (moe_router_grad_enabled()) {
            // Save the UNSCALED expert output so backward can form
            // g_w[r,e] = sum_dim(dy[r] * expert_out_e[r]) for the router grad.
            saved_moe_expert_out_[static_cast<size_t>(expert_idx)] = expert_out;
        }
        std::vector<float> selected_weights(selected_rows.size(), 0.0f);
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            selected_weights[local_row] = weight_ptr[target_row * num_experts + expert_idx];
        }
        Tensor scale_tensor({static_cast<int>(selected_rows.size()), 1}, Device::CPU);
        std::memcpy(scale_tensor.data(),
                    selected_weights.data(),
                    selected_weights.size() * sizeof(float));
        Tensor scaled_out = expert_out.mul(scale_tensor.to(target_device));
        Tensor expert_scatter = Tensor::zeros({rows, dim}, target_device);
        float* scatter_ptr = expert_scatter.raw_data();
        const float* scaled_ptr = scaled_out.raw_data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            copy_tensor_bytes(scatter_ptr + target_row * dim,
                           target_device,
                           scaled_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           static_cast<size_t>(dim) * sizeof(float));
        }
        output_accum = output_accum.add(expert_scatter);
    }

    return output_accum.reshape(x.shape.dims);
}

#ifdef USE_CUDA
Tensor JambaBlock::backward_moe_gpu_batched(const Tensor& dy, const Tensor& x) {
    // Pre-condition: caller verified eligibility (dy on GPU,
    // saved_moe_weights_ populated, gpu_custom_kernels_supported).
    //
    // Pipeline mirrors forward_moe_gpu_batched using the exact permutation
    // produced by that forward.  Re-launching the atomic assignment kernel can
    // change within-expert order and pair dY with another row's cached state.
    //   1. Upload saved permutation + scales.
    //   2. gather dy into permuted_dy [N_active, dim]
    //   4. scale rows of permuted_dy by the stored router weights
    //   5. for each expert: down.backward + gate_up.backward on the
    //      contiguous slice of permuted_dy
    //   6. scatter-add per-slot grad back into grad_accum [rows, dim]

    const int dim = x.shape.back();
    const int rows = static_cast<int>(saved_moe_weights_.size) /
                     std::max(num_experts, 1);

    // Reuse the persistent MoE workspace — same struct as forward.
    // Most allocations are no-ops because forward already grew the
    // buffers to peak size in this step; backward typically hits the
    // workspace at the same N_active that forward set up.
    MoeWorkspace& moe_ws = moe_workspace();
    const std::vector<int>& counts_host = saved_moe_counts_host_;
    const std::vector<int>& offsets_host = saved_moe_offsets_host_;
    const int N_active = saved_moe_n_active_;

    Tensor grad_accum = Tensor::zeros(dy.shape.dims, Device::GPU);
    if (N_active <= 0) return grad_accum;
    if (counts_host.size() != static_cast<size_t>(num_experts) ||
        offsets_host.size() != static_cast<size_t>(num_experts + 1) ||
        !saved_moe_permutation_device_.get() ||
        saved_moe_scale_device_.size != N_active) {
        throw std::runtime_error("MoE backward routing state is incomplete");
    }

    int* permutation_ptr =
        saved_moe_permutation_device_.get();
    float* scale_ptr = saved_moe_scale_device_.raw_data();
    int* offsets_ptr = saved_moe_ordered_device_ ? saved_moe_offsets_device_.get()
                                               : moe_ws.offsets(num_experts);
    if (!saved_moe_ordered_device_) {
    const cudaError_t offsets_copy_status = cudaMemcpy(
        offsets_ptr, offsets_host.data(),
        static_cast<size_t>(num_experts + 1) * sizeof(int),
        cudaMemcpyHostToDevice);
    if (offsets_copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("MoE backward offsets upload failed: ") +
            cudaGetErrorString(offsets_copy_status));
    }
    record_gpu_transfer(
        Device::GPU, Device::CPU,
        static_cast<size_t>(num_experts + 1) * sizeof(int));
    }

    Tensor router_grad_weights;
    if (moe_router_grad_enabled() && router) {
        if (static_cast<int64_t>(saved_moe_permuted_output_.size) != static_cast<int64_t>(N_active) * dim ||
            saved_moe_permuted_output_.get_device() != Device::GPU) {
            throw std::runtime_error(
                "MoE batched router gradient is missing unscaled expert outputs");
        }
        router_grad_weights =
            Tensor::zeros({rows, num_experts}, Device::GPU);
        launch_moe_router_weight_grad_kernel(
            dy.raw_data(), saved_moe_permuted_output_.raw_data(),
            permutation_ptr, offsets_ptr, router_grad_weights.raw_data(),
            N_active, dim, num_experts);
    }

    // Gather dy into permuted layout (contiguous by expert).
    Tensor permuted_dy = Tensor::uninitialized({N_active, dim}, Device::GPU);
    launch_moe_gather_rows_kernel(dy.raw_data(), permutation_ptr,
                                   permuted_dy.raw_data(), N_active, dim);

    // Apply per-slot router-weight scaling in-place via multiply-by-vector.
    // We use a simple mul broadcast: scale_buf has shape [N_active], so
    // we treat permuted_dy as [N_active, dim] and multiply each row by
    // scale_buf[row].  Done with the existing mul_vector_broadcast kernel.
    {
        Tensor scaled;
        // scale_buf is the per-row vector; reuse mul_vector_broadcast
        // semantics via a direct kernel launch.  We emit a small
        // strided-mul through the existing infrastructure to keep
        // changes localized; since launch_mul_vector_broadcast_kernel
        // expects [rows, cols] * [cols] (column-broadcast), we instead
        // do an explicit row-scale via the scatter kernel later — so
        // here we just retain permuted_dy unmodified and pass scale
        // into the scatter step.
        (void)scaled;
    }

    // Per-expert backward on contiguous slices.  expert_grad will be
    // copied into permuted_grad_input at the same offsets so the final
    // scatter aggregates correctly.
    Tensor permuted_grad_input = Tensor::uninitialized({N_active, dim}, Device::GPU);
    for (int e = 0; e < num_experts; ++e) {
        const int count = counts_host[static_cast<size_t>(e)];
        if (count <= 0) continue;
        const int offset = offsets_host[static_cast<size_t>(e)];

        // Pull the matching slice of permuted_dy and apply per-row
        // scale (CPU loop over count is small; the device-side
        // alternative is an extra kernel for marginal benefit).
        Tensor expert_dy = permuted_dy.slice(0, offset, offset + count);

        // Build a per-row scale tensor for this expert from scale_buf.
        // scale_buf[offset:offset+count] contains the router weights
        // for these slots — copy them to a small CPU vector then to
        // a device tensor for mul.
        Tensor scaled_dy({count, dim}, Device::GPU);
        launch_moe_scale_rows_kernel(
            expert_dy.raw_data(), scale_ptr + static_cast<size_t>(offset),
            scaled_dy.raw_data(), count, dim);

        Tensor expert_grad = expert_down[e]->backward(scaled_dy);
        // LEARN S1: apply squared_relu_backward between the two
        // BitLinears.  expert_grad on entry is dL/d(squared_relu_out);
        // we transform to dL/d(squared_relu_in) = dy * 2 * max(0, pre).
        // The saved pre-activation was captured in forward; if it is
        // empty (count was 0 last forward — shouldn't happen here
        // because we check count > 0 above, but defend in depth) we
        // skip the multiply and emit a zero grad to break the chain
        // cleanly.
        if (e < static_cast<int>(saved_moe_pre_activations_.size()) &&
            saved_moe_pre_activations_[static_cast<size_t>(e)].size > 0) {
            expert_grad = Tensor::squared_relu_backward(
                expert_grad,
                saved_moe_pre_activations_[static_cast<size_t>(e)]);
        }
        expert_grad = expert_gate_up[e]->backward(expert_grad);

        gpu_runtime_check(
            cudaMemcpyAsync(
                permuted_grad_input.raw_data() +
                    static_cast<size_t>(offset) * static_cast<size_t>(dim),
                expert_grad.raw_data(),
                static_cast<size_t>(count) * static_cast<size_t>(dim) *
                    sizeof(float),
                cudaMemcpyDeviceToDevice, gpu::current_stream()),
            "MoE expert gradient copy");
        record_gpu_transfer(
            Device::GPU, Device::GPU,
            static_cast<size_t>(count) * static_cast<size_t>(dim) *
                sizeof(float));
    }

    // Scatter-add the per-slot grads back into grad_accum.  Pass a
    // unit-scale tensor because the per-slot scaling was already
    // applied above on the dy side.  unit_scale is owned by the
    // workspace and pre-filled with 1.0f — H2D copy only on first
    // call or when N_active grows past the prior peak.
    if (saved_moe_ordered_device_) {
        launch_moe_ordered_combine(permuted_grad_input.raw_data(), saved_moe_inverse_device_.get(),
            nullptr, grad_accum.raw_data(), rows, dim, num_experts);
    } else {
        float* unit_scale_ptr = moe_ws.unit_scale(N_active);
        launch_moe_scatter_add_weighted_kernel(permuted_grad_input.raw_data(), permutation_ptr,
            unit_scale_ptr, grad_accum.raw_data(), N_active, dim);
    }

    if (router_grad_weights.size > 0) {
        Tensor router_input_grad =
            router->accumulate_task_router_grad(router_grad_weights);
        if (router_input_grad.size > 0) {
            if (router_input_grad.shape != grad_accum.shape &&
                router_input_grad.size == grad_accum.size) {
                router_input_grad =
                    router_input_grad.reshape(grad_accum.shape.dims);
            }
            if (router_input_grad.shape != grad_accum.shape ||
                router_input_grad.get_device() != Device::GPU) {
                throw std::runtime_error(
                    "MoE batched router input-gradient contract mismatch");
            }
            grad_accum = grad_accum.add(router_input_grad);
        }
    }

    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("MoE batched backward kernel failed: ") +
            cudaGetErrorString(launch_status));
    }

    return grad_accum;
}
#endif  // USE_CUDA

Tensor JambaBlock::backward_moe(const Tensor& dy, Context* ctx, const std::string& ln,
                                const Tensor& x) {
    (void)ctx;
    (void)ln;
    const Device target_device = dy.get_device();

    if (saved_moe_grouped_training_) {
        if (!grouped_moe_training_) throw std::logic_error("Missing grouped MoE tape");
        auto [input_grad, weight_grad] = grouped_moe_training_->backward(
            dy, router && moe_router_grad_enabled());
        saved_moe_grouped_training_ = false;
        if (weight_grad.size > 0) {
            Tensor router_grad = router->accumulate_task_router_grad(weight_grad);
            if (router_grad.size > 0) {
                if (router_grad.size == input_grad.size && router_grad.shape != input_grad.shape)
                    router_grad = router_grad.reshape(input_grad.shape.dims);
                if (router_grad.shape != input_grad.shape || router_grad.get_device() != target_device)
                    throw std::runtime_error("Grouped MoE router input-gradient contract mismatch");
                input_grad = input_grad.add(router_grad);
            }
        }
        return input_grad;
    }

#ifdef USE_CUDA
    // Ordered device dispatch has a single owner per output element and can
    // therefore retain deterministic routing and the task->router derivative.
    if (target_device == Device::GPU && gpu_custom_kernels_supported() &&
        saved_moe_permutation_device_.get() &&
        saved_moe_weights_.size > 0 &&
        num_experts > 0 && num_experts <= 1024 && x.shape.back() > 0 &&
        (!determinism::deterministic_reductions_enabled() || saved_moe_ordered_device_)) {
        return backward_moe_gpu_batched(dy, x);
    }
#endif

    Tensor grad_accum = Tensor::zeros(dy.shape.dims, target_device);
    if (saved_moe_rows_.empty() || saved_moe_weights_.size == 0) {
        throw std::logic_error("MoE backward requires matching forward routing state; refusing an all-expert surrogate gradient");
    }

    const int dim = x.shape.back();
    // AUDIT #4+#5: saved_moe_weights_ may now be on GPU (we stopped
    // doing the eager .cpu() copy in forward to save D2H time).  The
    // CPU backward path needs host-side access to the weight values
    // for per-expert scaling, so we materialize once here (single D2H)
    // and then read from the host copy.  This is the rare path —
    // GPU-eligible backward goes through backward_moe_gpu_batched
    // which doesn't need the weights tensor at all.
    Tensor moe_weights_host =
        saved_moe_weights_.get_device() == Device::GPU
            ? saved_moe_weights_.cpu()
            : saved_moe_weights_;
    const float* weight_ptr = moe_weights_host.data();

    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        const auto& selected_rows = saved_moe_rows_[static_cast<size_t>(expert_idx)];
        if (selected_rows.empty()) {
            continue;
        }

        Tensor expert_dy({static_cast<int>(selected_rows.size()), dim}, target_device);
        float* expert_dy_ptr = expert_dy.raw_data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            copy_tensor_bytes(expert_dy_ptr + static_cast<int>(local_row) * dim,
                           target_device,
                           dy.raw_data() + source_row * dim,
                           dy.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }
        std::vector<float> selected_weights(selected_rows.size(), 0.0f);
        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int source_row = selected_rows[local_row];
            selected_weights[local_row] = weight_ptr[source_row * num_experts + expert_idx];
        }
        Tensor scale_tensor({static_cast<int>(selected_rows.size()), 1}, Device::CPU);
        std::memcpy(scale_tensor.data(),
                    selected_weights.data(),
                    selected_weights.size() * sizeof(float));
        expert_dy = expert_dy.mul(scale_tensor.to(target_device));

        Tensor expert_grad = expert_down[expert_idx]->backward(expert_dy);
        // LEARN S1: squared_relu backward — uses pre-activation saved
        // by the corresponding forward branch.  See the GPU batched
        // backward path for the same fix.
        if (expert_idx < static_cast<int>(saved_moe_pre_activations_.size()) &&
            saved_moe_pre_activations_[static_cast<size_t>(expert_idx)].size > 0) {
            expert_grad = Tensor::squared_relu_backward(
                expert_grad,
                saved_moe_pre_activations_[static_cast<size_t>(expert_idx)]);
        }
        expert_grad = expert_gate_up[expert_idx]->backward(expert_grad);
        Tensor expert_scatter = Tensor::zeros(dy.shape.dims, target_device);
        float* scatter_ptr = expert_scatter.raw_data();
        const float* expert_grad_ptr = expert_grad.raw_data();

        for (size_t local_row = 0; local_row < selected_rows.size(); ++local_row) {
            const int target_row = selected_rows[local_row];
            copy_tensor_bytes(scatter_ptr + target_row * dim,
                           target_device,
                           expert_grad_ptr + static_cast<int>(local_row) * dim,
                           expert_grad.get_device(),
                           static_cast<size_t>(dim) * sizeof(float));
        }
        grad_accum = grad_accum.add(expert_scatter);
    }

    // Task gradient to the router (opt-in): the forward scaled each expert's
    // output by w[r,e], so dL/dw[r,e] = sum_dim(dy[r] * expert_out_e[r]).  Feed
    // that to the router, which backprops through renorm(top-k(softmax)) into
    // the gate.  Without this the router never learns from the task loss.
    if (moe_router_grad_enabled() && router &&
        saved_moe_expert_out_.size() == static_cast<size_t>(num_experts) &&
        !saved_moe_rows_.empty()) {
        Tensor dy_host = dy.get_device() == Device::GPU ? dy.cpu() : dy;
        const float* dy_ptr = dy_host.data();
        const int rows_total = dim > 0 ? dy_host.size / dim : 0;
        Tensor g_w = Tensor::zeros({rows_total, num_experts}, Device::CPU);
        float* gw = g_w.data();
        for (int e = 0; e < num_experts; ++e) {
            const Tensor& eo = saved_moe_expert_out_[static_cast<size_t>(e)];
            const auto& rows_e = saved_moe_rows_[static_cast<size_t>(e)];
            if (eo.size == 0 || rows_e.empty()) continue;
            Tensor eo_host = eo.get_device() == Device::GPU ? eo.cpu() : eo;
            const float* eo_ptr = eo_host.data();
            for (size_t lr = 0; lr < rows_e.size(); ++lr) {
                const int r = rows_e[lr];
                if (r < 0 || r >= rows_total) continue;
                double acc = 0.0;
                const float* dyr = dy_ptr + static_cast<size_t>(r) * dim;
                const float* eor = eo_ptr + lr * dim;
                for (int d = 0; d < dim; ++d) acc += static_cast<double>(dyr[d]) * eor[d];
                gw[static_cast<size_t>(r) * num_experts + e] = static_cast<float>(acc);
            }
        }
        Tensor router_input_grad = router->accumulate_task_router_grad(g_w);
        if (router_input_grad.size > 0) {
            if (router_input_grad.shape != grad_accum.shape &&
                router_input_grad.size == grad_accum.size) {
                router_input_grad = router_input_grad.reshape(grad_accum.shape.dims);
            }
            if (router_input_grad.shape != grad_accum.shape) {
                throw std::runtime_error(
                    "MoE router input-gradient shape mismatch");
            }
            if (router_input_grad.get_device() != grad_accum.get_device()) {
                router_input_grad = router_input_grad.to(grad_accum.get_device());
            }
            grad_accum = grad_accum.add(router_input_grad);
        }
    }

    return grad_accum;
}

Tensor JambaBlock::backward(const Tensor& dy, Context* ctx) {
    const bool audit_enabled =
        audit_collector_ && audit_collector_->enabled();
    AuditLatencyTimer audit_timer(audit_enabled, dy.get_device());
    // Reapply the forward dropout masks to the gradient (STE): grad flows only
    // through kept units and carries the same keep_scale the forward applied.
    // Empty mask (dropout off this forward) -> identity.
    auto masked = [](const Tensor& g, const Tensor& m) {
        return m.size > 0 ? g.mul(m) : g;
    };
    auto backprop_ff_norm = [&](Tensor gradient) {
        if (saved_residual_.size == 0 || saved_ff_norm_.size == 0) {
            return gradient;
        }
        if (learnable_ff_norm_ && saved_ff_rms_.size > 0) {
            const int rows =
                static_cast<int>(saved_residual_.size / d_model);
            ff_norm_weight_.add_grad(
                gradient.mul(saved_ff_rms_)
                    .reshape({rows, d_model})
                    .sum(0));
            gradient = gradient.mul(ff_norm_weight_.data);
            return saved_residual_.rmsnorm_backward(
                gradient, saved_ff_rms_, 1e-5f);
        }
        return saved_residual_.rmsnorm_backward(
            gradient, saved_ff_norm_);
    };
    if (faithful_mamba_core_only_) {
        if ((!mamba_layer && !mamba3_layer) || saved_input_.size == 0 ||
            saved_core_rms_.size == 0) {
            throw std::logic_error(
                "faithful Mamba core backward is missing forward state");
        }
        Context local_context;
        Tensor core_grad = mamba_backward(
            masked(dy, saved_drop_core_),
            ctx ? *ctx : local_context);
        if (learnable_core_norm_) {
            const int rows = saved_input_.size / d_model;
            core_norm_weight_.add_grad(
                core_grad.mul(saved_core_rms_)
                    .reshape({rows, d_model})
                    .sum(0));
            core_grad = core_grad.mul(core_norm_weight_.data);
        }
        core_grad = saved_input_.rmsnorm_backward(
            core_grad, saved_core_rms_, core_norm_eps_);
        Tensor input_grad = dy.add(core_grad);
        if (audit_enabled) {
            const double latency_ms = audit_timer.elapsed_ms();
            audit_collector_->record_backward(
                layer_idx, audit_block_type(), dy, input_grad,
                latency_ms);
        }
        release_consumed_backward_state();
        return input_grad;
    }

    const Tensor dy_ff =
        hybrid_parallel_
            ? dy.mul(ffn_gate_.data)
            : use_attn_layerscale_
            ? dy.mul(attn_layerscale_.data)
            : dy;
    Tensor ff_grad;
    if (is_moe) {
        ff_grad = backward_moe(masked(dy_ff, saved_drop_moe_), ctx,
                               "L" + std::to_string(layer_idx), saved_ff_norm_);
        // CHRASS parallel: its grad w.r.t. saved_ff_norm adds to ff_grad
        // before the shared rmsnorm_backward.  Reshape dy + saved_ff_norm
        // to 2D, run chrass.backward, reshape result back.
        if (chrass_layer && saved_ff_norm_.size > 0) {
            const auto& s = saved_ff_norm_.shape;
            int total_lead = 1;
            std::vector<int> orig_shape;
            for (size_t i = 0; i < s.size(); ++i) {
                orig_shape.push_back(s[i]);
                if (i + 1 < s.size()) total_lead *= s[i];
            }
            Tensor dy_flat = dy_ff.reshape({total_lead, orig_shape.back()});
            Tensor x_flat  = saved_ff_norm_.reshape({total_lead, orig_shape.back()});
            Tensor c_grad_flat = chrass_layer->backward(dy_flat, x_flat);
            Tensor c_grad = c_grad_flat.reshape(orig_shape);
            ff_grad = ff_grad.add(c_grad);
        }
        ff_grad = backprop_ff_norm(ff_grad);
    } else if (kan_ffn) {
        ff_grad = kan_ffn->backward(masked(dy_ff, saved_drop_ff_out_));
        // CHRASS parallel (KAN path) — same injection as the dense FFN.
        if (chrass_layer && saved_ff_norm_.size > 0) {
            const auto& s = saved_ff_norm_.shape;
            int total_lead = 1;
            std::vector<int> orig_shape;
            for (size_t i = 0; i < s.size(); ++i) {
                orig_shape.push_back(s[i]);
                if (i + 1 < s.size()) total_lead *= s[i];
            }
            Tensor dy_flat = dy_ff.reshape({total_lead, orig_shape.back()});
            Tensor x_flat  = saved_ff_norm_.reshape({total_lead, orig_shape.back()});
            Tensor c_grad_flat = chrass_layer->backward(dy_flat, x_flat);
            Tensor c_grad = c_grad_flat.reshape(orig_shape);
            ff_grad = ff_grad.add(c_grad);
        }
        ff_grad = backprop_ff_norm(ff_grad);
    } else if (ffn_down && ffn_gate_up) {
        ff_grad = ffn_down->backward(masked(dy_ff, saved_drop_ff_out_));
        // Hidden dropout was applied AFTER squared_relu, BEFORE ffn_down, so its
        // mask multiplies the grad w.r.t. ff_hidden before the squared_relu VJP.
        ff_grad = masked(ff_grad, saved_drop_ff_hidden_);
        if (saved_ff_hidden_pre_.size > 0) {
            // LEARN S1: squared ReLU backward.  d(max(0,x)^2)/dx is:
            //   2 * max(0, x)   for x > 0
            //   0               for x <= 0
            // We use Tensor::squared_relu_backward which dispatches the
            // GPU kernel when both tensors live on the GPU and falls
            // back to a parallel OpenMP loop on CPU.  This replaces the
            // plain-ReLU backward that just zeroed gradient for x <= 0
            // and left it identity for x > 0.
            ff_grad = Tensor::squared_relu_backward(ff_grad, saved_ff_hidden_pre_);
        }
        ff_grad = ffn_gate_up->backward(ff_grad);
        // CHRASS parallel (FFN path)
        if (chrass_layer && saved_ff_norm_.size > 0) {
            const auto& s = saved_ff_norm_.shape;
            int total_lead = 1;
            std::vector<int> orig_shape;
            for (size_t i = 0; i < s.size(); ++i) {
                orig_shape.push_back(s[i]);
                if (i + 1 < s.size()) total_lead *= s[i];
            }
            Tensor dy_flat = dy_ff.reshape({total_lead, orig_shape.back()});
            Tensor x_flat  = saved_ff_norm_.reshape({total_lead, orig_shape.back()});
            Tensor c_grad_flat = chrass_layer->backward(dy_flat, x_flat);
            Tensor c_grad = c_grad_flat.reshape(orig_shape);
            ff_grad = ff_grad.add(c_grad);
        }
        ff_grad = backprop_ff_norm(ff_grad);
    } else {
        ff_grad = make_zero_like(dy);
    }

    // Regra da Cadeia para Conexão Residual Superior: d_residual = dy + d_ff
    Tensor residual_grad = dy.add(ff_grad);

    if (hybrid_parallel_) {
        const int rows = static_cast<int>(dy.size / d_model);
        if (saved_ls_ff_.size > 0) {
            ffn_gate_.add_grad(
                dy.mul(saved_ls_ff_)
                    .reshape({rows, d_model})
                    .sum(0));
        }
        if (saved_mamba_core_.size == 0 ||
            saved_attention_core_.size == 0) {
            throw std::logic_error(
                "parallel hybrid backward is missing branch activations");
        }
        mamba_gate_.add_grad(
            residual_grad.mul(saved_mamba_core_)
                .reshape({rows, d_model})
                .sum(0));
        attention_gate_.add_grad(
            residual_grad.mul(saved_attention_core_)
                .reshape({rows, d_model})
                .sum(0));

        Tensor mamba_grad = masked(
            residual_grad.mul(mamba_gate_.data),
            saved_drop_mamba_);
        Tensor attention_grad = masked(
            residual_grad.mul(attention_gate_.data),
            saved_drop_attention_);
        const Tensor mamba_mixer_grad = mamba_grad;
        const Tensor attention_mixer_grad = attention_grad;
        Context local_context;
        mamba_grad = mamba_backward(
            mamba_grad, ctx ? *ctx : local_context);
        attention_grad =
            attn_layer->backward(attention_grad, ctx);

        if (learnable_core_norm_ && saved_core_rms_.size > 0) {
            core_norm_weight_.add_grad(
                mamba_grad.mul(saved_core_rms_)
                    .reshape({rows, d_model})
                    .sum(0));
            mamba_grad =
                mamba_grad.mul(core_norm_weight_.data);
        }
        mamba_grad = saved_input_.rmsnorm_backward(
            mamba_grad,
            learnable_core_norm_ ? saved_core_rms_
                                 : saved_core_norm_,
            core_norm_eps_);

        if (learnable_attention_norm_ &&
            saved_attention_rms_.size > 0) {
            attention_norm_weight_.add_grad(
                attention_grad.mul(saved_attention_rms_)
                    .reshape({rows, d_model})
                    .sum(0));
            attention_grad =
                attention_grad.mul(attention_norm_weight_.data);
        }
        attention_grad = saved_input_.rmsnorm_backward(
            attention_grad,
            learnable_attention_norm_ ? saved_attention_rms_
                                      : saved_attention_norm_,
            core_norm_eps_);

        if (audit_enabled) {
            audit_collector_->record_hybrid_interaction(
                layer_idx, "backward", mamba_mixer_grad,
                attention_mixer_grad, dy_ff,
                mamba_grad, attention_grad, ff_grad);
        }
        Tensor input_grad =
            residual_grad.add(mamba_grad).add(attention_grad);
        if (audit_enabled) {
            const double latency_ms = audit_timer.elapsed_ms();
            audit_collector_->record_backward(
                layer_idx, audit_block_type(), dy, input_grad,
                latency_ms);
        }
        release_consumed_backward_state();
        return input_grad;
    }

    // The mixer output passed through dropout before the residual add, so the
    // mixer's input grad carries the core mask; the x-branch of the residual
    // (used for input_grad below) does NOT — keep residual_grad unmasked there.
    // LayerScale gamma grad: d/dgamma of (gamma*ff) and (gamma*core) summed over
    // rows = sum(dy .* ff_pre) + sum(residual_grad .* core_pre).  Computed here,
    // where residual_grad still holds the unscaled skip grad (the mixer grad
    // below multiplies a COPY by gamma; the residual skip stays unscaled).
    if (use_attn_layerscale_ && attn_layerscale_.data.size > 0) {
        const int width = attn_layerscale_.data.size;
        Tensor g = Tensor::zeros({width}, dy.get_device());
        if (saved_ls_ff_.size > 0) {
            const int rows = static_cast<int>(dy.size / width);
            g = g.add(dy.mul(saved_ls_ff_).reshape({rows, width}).sum(0));
        }
        if (saved_ls_core_.size > 0) {
            const int rows = static_cast<int>(residual_grad.size / width);
            g = g.add(
                residual_grad.mul(saved_ls_core_).reshape({rows, width}).sum(0));
        }
        attn_layerscale_.add_grad(g);
    }
    const Tensor core_scaled_grad =
        use_attn_layerscale_
            ? residual_grad.mul(attn_layerscale_.data)
            : residual_grad;
    const Tensor core_in_grad = masked(core_scaled_grad, saved_drop_core_);
    Tensor core_grad;
    if (is_ttt && ttt_layer) {
        core_grad = ttt_layer->backward(core_in_grad);
    } else if (is_attention && attn_layer) {
        core_grad = attn_layer->backward(core_in_grad, ctx);
    } else if (mamba_layer || mamba3_layer) {
        Context local_ctx;
        core_grad = mamba_backward(core_in_grad, ctx ? *ctx : local_ctx);
    } else {
        core_grad = make_zero_like(dy);
    }

    if (saved_input_.size > 0 && saved_core_norm_.size > 0) {
        if (learnable_core_norm_ && saved_core_rms_.size > 0) {
            const int rows = saved_input_.size / d_model;
            core_norm_weight_.add_grad(
                core_grad.mul(saved_core_rms_)
                    .reshape({rows, d_model})
                    .sum(0));
            core_grad = core_grad.mul(core_norm_weight_.data);
        }
        core_grad = saved_input_.rmsnorm_backward(
            core_grad,
            learnable_core_norm_ ? saved_core_rms_ : saved_core_norm_,
            core_norm_eps_);
    }

    // Regra da Cadeia Inferior: d_input = d_residual + d_core
    Tensor input_grad = residual_grad.add(core_grad);
    if (audit_enabled) {
        const double latency_ms = audit_timer.elapsed_ms();
        audit_collector_->record_backward(layer_idx, audit_block_type(), dy, input_grad,
                                          latency_ms);
    }
    release_consumed_backward_state();
    return input_grad;
}

void JambaBlock::release_consumed_backward_state() {
    saved_moe_grouped_training_ = false;
    if (grouped_moe_training_) grouped_moe_training_->discard_backward_state();
    saved_input_ = Tensor();
    saved_core_rms_ = Tensor();
    saved_core_norm_ = Tensor();
    saved_attention_rms_ = Tensor();
    saved_attention_norm_ = Tensor();
    saved_ff_rms_ = Tensor();
    saved_residual_ = Tensor();
    saved_ff_norm_ = Tensor();
    saved_ff_hidden_pre_ = Tensor();
    saved_ls_core_ = Tensor();
    saved_ls_ff_ = Tensor();
    saved_mamba_core_ = Tensor();
    saved_attention_core_ = Tensor();
    saved_drop_core_ = Tensor();
    saved_drop_mamba_ = Tensor();
    saved_drop_attention_ = Tensor();
    saved_drop_moe_ = Tensor();
    saved_drop_ff_hidden_ = Tensor();
    saved_drop_ff_out_ = Tensor();
    saved_moe_weights_ = Tensor();
    saved_moe_rows_.clear();
    saved_moe_scale_device_ = Tensor();
    saved_moe_permuted_output_ = Tensor();
    saved_moe_offsets_host_.clear();
    saved_moe_counts_host_.clear();
    saved_moe_n_active_ = 0;
    saved_moe_expert_out_.clear();
    // LEARN S1: clear per-expert pre-activations.  Default-construct
    // each Tensor releases its backing storage (unique_ptr in
    // Tensor::data_ptr); the vector itself stays sized at num_experts
    // until next forward overwrites it.
    saved_moe_pre_activations_.clear();
}

void JambaBlock::reset() {
    release_consumed_backward_state();
    if (mamba_layer) {
        mamba_layer->reset();
    }
    if (mamba3_layer) mamba3_layer->reset();
    if (attn_layer) {
        attn_layer->reset();
    }
    if (ttt_layer) {
        ttt_layer->reset();
    }
}

void JambaBlock::to(Device dev) {
    saved_moe_grouped_training_ = false;
    if (grouped_moe_training_) grouped_moe_training_->discard_backward_state();
    if (core_norm_weight_.data.size > 0) {
        core_norm_weight_.data = core_norm_weight_.data.to(dev);
    }
    if (attn_layerscale_.data.size > 0) {
        attn_layerscale_.data = attn_layerscale_.data.to(dev);
    }
    if (attention_norm_weight_.data.size > 0) {
        attention_norm_weight_.data =
            attention_norm_weight_.data.to(dev);
    }
    if (ff_norm_weight_.data.size > 0) {
        ff_norm_weight_.data = ff_norm_weight_.data.to(dev);
    }
    if (mamba_gate_.data.size > 0) {
        mamba_gate_.data = mamba_gate_.data.to(dev);
    }
    if (attention_gate_.data.size > 0) {
        attention_gate_.data = attention_gate_.data.to(dev);
    }
    if (ffn_gate_.data.size > 0) {
        ffn_gate_.data = ffn_gate_.data.to(dev);
    }
    if (attn_layer) {
        attn_layer->to(dev);
    }
    if (mamba_layer) {
        mamba_layer->to(dev);
    }
    if (mamba3_layer) mamba3_layer->to(dev);
    if (ttt_layer) {
        ttt_layer->to(dev);
    }
    if (router) {
        router->to(dev);
    }
    if (ffn_gate_up) {
        ffn_gate_up->to(dev);
    }
    if (ffn_down) {
        ffn_down->to(dev);
    }
    if (kan_ffn) {
        kan_ffn->to(dev);
    }
    for (auto& expert : expert_gate_up) {
        expert->to(dev);
    }
    for (auto& expert : expert_down) {
        expert->to(dev);
    }
}

std::vector<Parameter*> JambaBlock::parameters() {
    std::vector<Parameter*> params;
    if (core_norm_weight_.data.size > 0) {
        core_norm_weight_.assign_relative_name(
            hybrid_parallel_ ? "mamba.pre_norm.weight" : "norm.weight");
        params.push_back(&core_norm_weight_);
    }
    if (attention_norm_weight_.data.size > 0) {
        attention_norm_weight_.assign_relative_name("attn.pre_norm.weight");
        params.push_back(&attention_norm_weight_);
    }
    if (ff_norm_weight_.data.size > 0) {
        ff_norm_weight_.assign_relative_name("ffn.pre_norm.weight");
        params.push_back(&ff_norm_weight_);
    }
    if (attn_layerscale_.data.size > 0) {
        attn_layerscale_.assign_relative_name("attn.layerscale");
        params.push_back(&attn_layerscale_);
    }
    if (mamba_gate_.data.size > 0) {
        mamba_gate_.assign_relative_name("mamba.gate");
        params.push_back(&mamba_gate_);
    }
    if (attention_gate_.data.size > 0) {
        attention_gate_.assign_relative_name("attn.gate");
        params.push_back(&attention_gate_);
    }
    if (ffn_gate_.data.size > 0) {
        ffn_gate_.assign_relative_name("ffn.gate");
        params.push_back(&ffn_gate_);
    }
    if (attn_layer) {
        auto attn = attn_layer->parameters();
        prefix_parameter_names(attn, "attn.");
        params.insert(params.end(), attn.begin(), attn.end());
    }
    if (mamba_layer) {
        auto mamba = mamba_layer->parameters();
        prefix_parameter_names(mamba, "mamba.");
        params.insert(params.end(), mamba.begin(), mamba.end());
    }
    if (mamba3_layer) {
        auto mamba3 = mamba3_layer->parameters();
        prefix_parameter_names(mamba3, "mamba3.");
        params.insert(params.end(), mamba3.begin(), mamba3.end());
    }
    if (ttt_layer) {
        auto ttt = ttt_layer->parameters();
        prefix_parameter_names(ttt, "ttt.");
        params.insert(params.end(), ttt.begin(), ttt.end());
    }
    if (router) {
        auto router_params = router->parameters();
        prefix_parameter_names(router_params, "router.");
        params.insert(params.end(), router_params.begin(), router_params.end());
    }
    if (ffn_gate_up) {
        auto ff_up = ffn_gate_up->parameters();
        prefix_parameter_names(ff_up, "ffn_gate_up.");
        params.insert(params.end(), ff_up.begin(), ff_up.end());
    }
    if (ffn_down) {
        auto ff_down = ffn_down->parameters();
        prefix_parameter_names(ff_down, "ffn_down.");
        params.insert(params.end(), ff_down.begin(), ff_down.end());
    }
    if (kan_ffn) {
        auto kan_params = kan_ffn->parameters();
        prefix_parameter_names(kan_params, "kan_ffn.");
        params.insert(params.end(), kan_params.begin(), kan_params.end());
    }
    if (chrass_layer) {
        auto chrass_params = chrass_layer->parameters();
        prefix_parameter_names(chrass_params, "chrass.");
        params.insert(params.end(), chrass_params.begin(), chrass_params.end());
    }
    for (size_t index = 0; index < expert_gate_up.size(); ++index) {
        auto& expert = expert_gate_up[index];
        auto expert_params = expert->parameters();
        prefix_parameter_names(expert_params, "experts." + std::to_string(index) + ".up.");
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    for (size_t index = 0; index < expert_down.size(); ++index) {
        auto& expert = expert_down[index];
        auto expert_params = expert->parameters();
        prefix_parameter_names(expert_params, "experts." + std::to_string(index) + ".down.");
        params.insert(params.end(), expert_params.begin(), expert_params.end());
    }
    return params;
}

void JambaBlock::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (attn_layer) {
        attn_layer->collect_bitlinear_layers(out);
    }
    if (mamba_layer) {
        mamba_layer->collect_bitlinear_layers(out);
    }
    if (ttt_layer) {
        ttt_layer->collect_bitlinear_layers(out);
    }
    if (router) {
        router->collect_bitlinear_layers(out);
    }
    if (ffn_gate_up) {
        out.push_back(ffn_gate_up.get());
    }
    if (ffn_down) {
        out.push_back(ffn_down.get());
    }
    for (auto& expert : expert_gate_up) {
        out.push_back(expert.get());
    }
    for (auto& expert : expert_down) {
        out.push_back(expert.get());
    }
}

void JambaBlock::set_streaming_inference(bool enabled) {
    if (attn_layer) {
        attn_layer->set_streaming_mode(enabled);
    }
    if (mamba_layer) {
        mamba_layer->set_streaming_mode(enabled);
    }
    if (mamba3_layer) mamba3_layer->set_streaming_mode(enabled);
}

void JambaBlock::set_training_mode(bool enabled) {
    if (enabled != training_mode_) {
        saved_moe_grouped_training_ = false;
        if (grouped_moe_training_) grouped_moe_training_->discard_backward_state();
    }
    training_mode_ = enabled;
    if (mamba_layer) {
        mamba_layer->set_training_mode(enabled);
    }
    if (mamba3_layer) {mamba3_layer->cancel_pending();mamba3_layer->set_training_mode(enabled);}
    if (attn_layer) {
        attn_layer->set_training_mode(enabled);
    }
    if (ttt_layer) {
        ttt_layer->set_training_mode(enabled);
    }
}

void JambaBlock::set_batch_valid_lengths(const std::vector<int>& lengths) {
    active_batch_valid_lengths_ = lengths;
    if (attn_layer) {
        attn_layer->set_batch_valid_lengths(lengths);
    }
    if (ttt_layer) {
        ttt_layer->set_batch_valid_lengths(lengths);
    }
}

JambaBlockSessionSnapshot JambaBlock::snapshot_session_state(bool device_resident) const {
    JambaBlockSessionSnapshot snapshot;
    snapshot.has_attention = static_cast<bool>(attn_layer);
    snapshot.has_mamba = static_cast<bool>(mamba_layer);
    snapshot.has_mamba3 = static_cast<bool>(mamba3_layer);
    if(mamba3_layer) snapshot.mamba3=mamba3_layer->snapshot_streaming_state(device_resident);
    snapshot.has_ttt = static_cast<bool>(ttt_layer);
    if (attn_layer) {
        snapshot.attention = attn_layer->snapshot_cache();
    }
    if (mamba_layer) {
        snapshot.mamba = mamba_layer->snapshot_streaming_state(device_resident);
    }
    if (ttt_layer) {
        snapshot.ttt = ttt_layer->snapshot_state();
    }
    return snapshot;
}

std::vector<JambaBlockSessionSnapshot> JambaBlock::snapshot_session_state_batch(bool device_resident) const {
    const std::vector<AttentionCacheSnapshot> attention_snapshots =
        attn_layer ? attn_layer->snapshot_cache_batch()
                   : std::vector<AttentionCacheSnapshot>{};
    const std::vector<MambaStreamSnapshot> mamba_snapshots =
        mamba_layer ? mamba_layer->snapshot_streaming_state_batch(device_resident)
                    : std::vector<MambaStreamSnapshot>{};
    const auto mamba3_snapshots = mamba3_layer ? mamba3_layer->snapshot_streaming_state_batch(device_resident) : std::vector<Mamba3SessionSnapshot>{};
    size_t batch_size = (std::max)({attention_snapshots.size(), mamba_snapshots.size(), mamba3_snapshots.size()});
    if(!attention_snapshots.empty() && !mamba3_snapshots.empty() && attention_snapshots.size()!=mamba3_snapshots.size()) throw std::runtime_error("Mamba3/Attention session batch mismatch");
    if (!attention_snapshots.empty() && !mamba_snapshots.empty() &&
        attention_snapshots.size() != mamba_snapshots.size()) {
        throw std::runtime_error(
            "hybrid session snapshot branch batch sizes differ");
    }
    if (batch_size == 0 && ttt_layer && last_batch_size_ > 0) {
        batch_size = static_cast<size_t>(last_batch_size_);
    }
    if (batch_size == 0) {
        return {};
    }
    std::vector<JambaBlockSessionSnapshot> snapshots;
    snapshots.reserve(batch_size);
    const TTTSessionSnapshot shared_ttt =
        ttt_layer ? ttt_layer->snapshot_state() : TTTSessionSnapshot{};
    for (size_t batch = 0; batch < batch_size; ++batch) {
        JambaBlockSessionSnapshot snapshot;
        if (!attention_snapshots.empty()) {
            snapshot.has_attention = true;
            snapshot.attention = attention_snapshots[batch];
        }
        if(!mamba3_snapshots.empty()) {snapshot.has_mamba3=true;snapshot.mamba3=mamba3_snapshots[batch];}
        if (!mamba_snapshots.empty()) {
            snapshot.has_mamba = true;
            snapshot.mamba = mamba_snapshots[batch];
        }
        if (ttt_layer) {
            snapshot.has_ttt = true;
            snapshot.ttt = shared_ttt;
        }
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void JambaBlock::restore_session_state(const JambaBlockSessionSnapshot& snapshot) {
    if(mamba3_layer) {if(!snapshot.has_mamba3) throw std::invalid_argument("session lacks Mamba3 branch");mamba3_layer->restore_streaming_state(snapshot.mamba3);}
    if (attn_layer && snapshot.has_attention) {
        attn_layer->restore_cache(snapshot.attention);
    }
    if (mamba_layer && snapshot.has_mamba) {
        mamba_layer->restore_streaming_state(snapshot.mamba);
    }
    if (ttt_layer && snapshot.has_ttt) {
        ttt_layer->restore_state(snapshot.ttt);
    }
}

void JambaBlock::restore_session_state_batch(const std::vector<JambaBlockSessionSnapshot>& snapshots) {
    if(mamba3_layer) {
        std::vector<Mamba3SessionSnapshot> states;
        for(const auto& snapshot:snapshots) {if(!snapshot.has_mamba3) throw std::invalid_argument("batch session lacks Mamba3 branch");states.push_back(snapshot.mamba3);}
        mamba3_layer->restore_streaming_state_batch(states);
    }
    if (attn_layer) {
        std::vector<AttentionCacheSnapshot> attention_snapshots;
        attention_snapshots.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            if (!snapshot.has_attention) {
                throw std::invalid_argument(
                    "session snapshot lacks the hybrid Attention branch");
            }
            attention_snapshots.push_back(snapshot.attention);
        }
        attn_layer->restore_cache_batch(attention_snapshots);
    }
    if (mamba_layer) {
        std::vector<MambaStreamSnapshot> mamba_snapshots;
        mamba_snapshots.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            if (!snapshot.has_mamba) {
                throw std::invalid_argument(
                    "session snapshot lacks the hybrid Mamba branch");
            }
            mamba_snapshots.push_back(snapshot.mamba);
        }
        mamba_layer->restore_streaming_state_batch(mamba_snapshots);
    }
    if (ttt_layer) {
        if (snapshots.empty() || !snapshots.front().has_ttt) {
            throw std::invalid_argument(
                "session snapshot lacks the TTT branch");
        }
        ttt_layer->restore_state(snapshots.front().ttt);
    }
    if (!attn_layer && !mamba_layer && !mamba3_layer && !ttt_layer) {
        reset();
    }
}

Attention::Attention(int d, int n, int l, int n_kv, float rope_theta,
                     int sliding_window)
    : d_model(d),
      n_heads(n),
      n_kv_heads(n_kv),
      kv_group_size(n_kv > 0 ? n / n_kv : 0),
      head_dim(n > 0 ? d / n : 0),
      n_latents(l),
      q_down_proj(std::make_unique<BitLinear>(d, d)),
      kv_down_proj(
          std::make_unique<BitLinear>(d, 2 * n_kv_heads * head_dim)),
      out_proj(std::make_unique<BitLinear>(d, d)),
      max_seq_len(4096),
      sliding_window_(sliding_window),
      theta(rope_theta) {
    if (d <= 0 || n <= 0 || n_kv <= 0 || l <= 0 || d % n != 0 ||
        n_kv > n || n % n_kv != 0 || ((d / n) & 1) != 0 ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0f ||
        sliding_window <= 0) {
        throw std::invalid_argument(
            "Attention requires positive dimensions, d_model divisible by "
            "n_heads, an even head dimension, n_heads divisible by "
            "n_kv_heads, finite positive rope_theta, and a positive "
            "sliding_window");
    }
    // The serialized theta is passed directly to the exact RoPE kernels.
    precompute_freqs_cis();
    // Learned block-selection routing, initialised to identity so SSA
    // selection starts as the mean-key heuristic and is then trained.
    ssa_wsel_ = Parameter(Tensor::eye(head_dim), "attn.ssa_wsel");
}

void Attention::set_exact_linear_mode(bool enabled) {
    q_down_proj->set_exact_linear_mode(enabled);
    kv_down_proj->set_exact_linear_mode(enabled);
    out_proj->set_exact_linear_mode(enabled);
}

void Attention::precompute_freqs_cis() {
    // RoPE: pré-computa cos/sin para cada posição e dimensão
    // head_dim/2 pares de dimensões recebem rotações distintas
    const int half_dim = head_dim / 2;
    cos_cached.resize(max_seq_len * half_dim);
    sin_cached.resize(max_seq_len * half_dim);
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * i / head_dim);
            float angle = static_cast<float>(pos) * freq;
            cos_cached[pos * half_dim + i] = std::cos(angle);
            sin_cached[pos * half_dim + i] = std::sin(angle);
        }
    }
}

void Attention::ensure_freqs_capacity(int max_pos_exclusive) {
    if (max_pos_exclusive <= max_seq_len) {
        return;
    }
    int new_len = std::max(max_seq_len, 1);
    while (new_len < max_pos_exclusive) {
        new_len *= 2;
    }
    max_seq_len = new_len;
    precompute_freqs_cis();   // rebuilds cos_cached/sin_cached for the new length
    rope_gpu_uploaded_ = 0;   // force ensure_rope_gpu_cache() to re-upload
}

#ifdef USE_CUDA
namespace {

// Persistent, content-addressed valid-length staging. All attention layers in a
// model step receive the same mask, so uploading it once avoids a synchronous
// pageable H2D copy per hybrid layer. Default-stream ordering makes replacing
// changed contents safe with respect to kernels from the preceding layer.
int* attn_valid_device_values(const std::vector<int>& values) {
    if (values.empty()) return nullptr;
    struct Cache {
        cuda_detail::DeviceBuffer<int> buffer;
        std::vector<int> values;
        int* uploaded_pointer = nullptr;
    };
    thread_local Cache cache;
    int* pointer = cache.buffer.ensure(values.size());
    if (pointer == nullptr) return nullptr;
    if (pointer != cache.uploaded_pointer || cache.values != values) {
        gpu_runtime_check(
            cudaMemcpy(pointer, values.data(),
                       values.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
            "attention valid-length upload");
        record_gpu_transfer(
            Device::GPU, Device::CPU, values.size() * sizeof(int));
        cache.values = values;
        cache.uploaded_pointer = pointer;
    }
    return pointer;
}

// NSOS_ATTN_BWD_HOST=1 forces the host exact-cache backward (the A/B parity arm
// against the GPU path).  Read per call so one process can flip arms.
bool attn_bwd_force_host() {
    const char* e = std::getenv("NSOS_ATTN_BWD_HOST");
    return e != nullptr && e[0] == '1';
}

void attn_train_check_cuda(const char* op) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

}  // namespace
#endif  // USE_CUDA

// Uploads cos_cached/sin_cached to device buffers (lazily; re-uploads only when
// the host tables grew, e.g. after reserve_kv_cache).  No-op on CPU builds.
void Attention::ensure_rope_gpu_cache() {
#ifdef USE_CUDA
    const size_t n = cos_cached.size();
    if (n == 0 || (rope_gpu_uploaded_ == n &&
                   rope_cos_gpu_.size == static_cast<int>(n))) {
        return;
    }
    rope_cos_gpu_ =
        Tensor::uninitialized({static_cast<int>(n)}, Device::GPU);
    rope_sin_gpu_ =
        Tensor::uninitialized({static_cast<int>(n)}, Device::GPU);
    gpu_runtime_check(
        cudaMemcpy(rope_cos_gpu_.raw_data(), cos_cached.data(),
                   n * sizeof(float), cudaMemcpyHostToDevice),
        "RoPE cosine upload");
    record_gpu_transfer(Device::GPU, Device::CPU, n * sizeof(float));
    gpu_runtime_check(
        cudaMemcpy(rope_sin_gpu_.raw_data(), sin_cached.data(),
                   n * sizeof(float), cudaMemcpyHostToDevice),
        "RoPE sine upload");
    record_gpu_transfer(Device::GPU, Device::CPU, n * sizeof(float));
    rope_gpu_uploaded_ = n;
#endif
}

std::pair<Tensor, Tensor> Attention::apply_rope(const Tensor& q, const Tensor& k, int start_pos) {
    if (q.shape.size() != k.shape.size() || (q.shape.size() != 3 && q.shape.size() != 4)) {
        throw std::runtime_error("Attention::apply_rope expects rank-3 or rank-4 q/k tensors");
    }

    const bool batched = q.shape.size() == 4;
    const int batch = batched ? q.shape[0] : 1;
    const int seq_len = batched ? q.shape[1] : q.shape[0];
    const int q_heads = batched ? q.shape[2] : q.shape[1];
    const int k_heads = batched ? k.shape[2] : k.shape[1];
    const int half_dim = head_dim / 2;
    ensure_freqs_capacity(start_pos + seq_len);  // N7: grow tables, no silent clamp

    Tensor q_rot = q.clone();
    Tensor k_rot = k.clone();
    float* qd = q_rot.data();
    float* kd = k_rot.data();

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int pos = start_pos + s;  // table guaranteed to cover this (N7)
            for (int h = 0; h < q_heads; ++h) {
                const size_t q_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * q_heads + h) * head_dim
                        : (static_cast<size_t>(s) * q_heads + h) * head_dim;
                float* q_head = qd + q_offset;
                for (int i = 0; i < half_dim; ++i) {
                    float cos_v = cos_cached[pos * half_dim + i];
                    float sin_v = sin_cached[pos * half_dim + i];
                    float q0 = q_head[i],           q1 = q_head[i + half_dim];
                    q_head[i]            = q0 * cos_v - q1 * sin_v;
                    q_head[i + half_dim] = q0 * sin_v + q1 * cos_v;
                }
            }
            for (int h = 0; h < k_heads; ++h) {
                const size_t k_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * k_heads + h) * head_dim
                        : (static_cast<size_t>(s) * k_heads + h) * head_dim;
                float* k_head = kd + k_offset;
                for (int i = 0; i < half_dim; ++i) {
                    float cos_v = cos_cached[pos * half_dim + i];
                    float sin_v = sin_cached[pos * half_dim + i];
                    float k0 = k_head[i],           k1 = k_head[i + half_dim];
                    k_head[i]            = k0 * cos_v - k1 * sin_v;
                    k_head[i + half_dim] = k0 * sin_v + k1 * cos_v;
                }
            }
        }
    }
    return {q_rot, k_rot};
}

std::pair<Tensor, Tensor> Attention::apply_rope_backward(const Tensor& grad_q_rot,
                                                         const Tensor& grad_k_rot,
                                                         int start_pos) {
    if ((grad_q_rot.shape.size() != 3 && grad_q_rot.shape.size() != 4) ||
        (grad_k_rot.shape.size() != 3 && grad_k_rot.shape.size() != 4)) {
        throw std::runtime_error("RoPE backward expects rank-3 or rank-4 tensors");
    }

    const bool batched = grad_q_rot.shape.size() == 4;
    const int batch = batched ? grad_q_rot.shape[0] : 1;
    const int seq_len = batched ? grad_q_rot.shape[1] : grad_q_rot.shape[0];
    const int q_heads = batched ? grad_q_rot.shape[2] : grad_q_rot.shape[1];
    const int k_heads = batched ? grad_k_rot.shape[2] : grad_k_rot.shape[1];
    const int half_dim = head_dim / 2;
    ensure_freqs_capacity(start_pos + seq_len);  // N7: keep parity with forward tables

    Tensor grad_q = grad_q_rot.clone();
    Tensor grad_k = grad_k_rot.clone();
    float* qd = grad_q.data();
    float* kd = grad_k.data();

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int pos = start_pos + s;  // table guaranteed to cover this (N7)
            for (int h = 0; h < q_heads; ++h) {
                const size_t q_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * q_heads + h) * head_dim
                        : (static_cast<size_t>(s) * q_heads + h) * head_dim;
                float* q_head = qd + q_offset;
                for (int i = 0; i < half_dim; ++i) {
                    const float cos_v = cos_cached[pos * half_dim + i];
                    const float sin_v = sin_cached[pos * half_dim + i];
                    const float qr0 = q_head[i];
                    const float qr1 = q_head[i + half_dim];
                    q_head[i] = qr0 * cos_v + qr1 * sin_v;
                    q_head[i + half_dim] = -qr0 * sin_v + qr1 * cos_v;
                }
            }
            for (int h = 0; h < k_heads; ++h) {
                const size_t k_offset =
                    batched
                        ? ((static_cast<size_t>(b) * seq_len + s) * k_heads + h) * head_dim
                        : (static_cast<size_t>(s) * k_heads + h) * head_dim;
                float* k_head = kd + k_offset;
                for (int i = 0; i < half_dim; ++i) {
                    const float cos_v = cos_cached[pos * half_dim + i];
                    const float sin_v = sin_cached[pos * half_dim + i];
                    const float kr0 = k_head[i];
                    const float kr1 = k_head[i + half_dim];
                    k_head[i] = kr0 * cos_v + kr1 * sin_v;
                    k_head[i + half_dim] = -kr0 * sin_v + kr1 * cos_v;
                }
            }
        }
    }

    return {grad_q, grad_k};
}

Tensor Attention::forward_exact_gpu(const Tensor& q_flat,
                                    const Tensor& kv_flat,
                                    int batch_size, int seq_len,
                                    int kv_dim, float scale) {
#ifdef USE_CUDA
    saved_attn_lse_ = Tensor();
    saved_attn_output_ = Tensor();
    const size_t expected_tokens =
        static_cast<size_t>(batch_size > 0 ? batch_size : 0) *
        static_cast<size_t>(seq_len > 0 ? seq_len : 0);
    const size_t expected_q =
        expected_tokens * static_cast<size_t>(d_model);
    const size_t expected_kv =
        expected_tokens * static_cast<size_t>(2 * kv_dim);
    if (q_flat.get_device() != Device::GPU ||
        kv_flat.get_device() != Device::GPU ||
        batch_size <= 0 || seq_len <= 0 ||
        kv_dim != n_kv_heads * head_dim ||
        q_flat.size != expected_q ||
        kv_flat.size != expected_kv ||
        !std::isfinite(scale) || scale <= 0.0f ||
        static_cast<int>(saved_valid_lengths_.size()) != batch_size) {
        throw std::invalid_argument(
            "exact GPU attention forward received an invalid tensor/mask "
            "contract");
    }

    Tensor k_flat = Tensor::uninitialized(
        {batch_size, seq_len, kv_dim}, Device::GPU);
    Tensor v_flat = Tensor::uninitialized(
        {batch_size, seq_len, kv_dim}, Device::GPU);
    launch_kv_split(
        k_flat.raw_data(), v_flat.raw_data(), kv_flat.raw_data(),
        static_cast<long long>(batch_size) * seq_len, kv_dim);

    Tensor q_rot = q_flat.clone();
    Tensor k_rot = k_flat.clone();
    ensure_freqs_capacity(seq_len);
    ensure_rope_gpu_cache();
    launch_rope_apply(
        q_rot.raw_data(), rope_cos_gpu_.raw_data(),
        rope_sin_gpu_.raw_data(), batch_size, seq_len,
        n_heads, head_dim, 0, max_seq_len, +1);
    launch_rope_apply(
        k_rot.raw_data(), rope_cos_gpu_.raw_data(),
        rope_sin_gpu_.raw_data(), batch_size, seq_len,
        n_kv_heads, head_dim, 0, max_seq_len, +1);
    attn_train_check_cuda("exact attention split/RoPE forward");

    Tensor q_heads =
        q_rot.reshape({batch_size, seq_len, n_heads, head_dim});
    Tensor k_heads =
        k_rot.reshape({batch_size, seq_len, n_kv_heads, head_dim});
    Tensor v_heads =
        v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
    if (training_mode_) {
        saved_q_rot_ = q_heads;
        saved_k_rot_ = k_heads;
        saved_v_heads_ = v_heads;
    }
    saved_attn_probs_ = Tensor();

    std::vector<int> valid_host(static_cast<size_t>(batch_size), seq_len);
    for (int batch = 0; batch < batch_size; ++batch) {
        valid_host[static_cast<size_t>(batch)] = std::clamp(
            saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
    }
    int* device_valid = attn_valid_device_values(valid_host);
    if (device_valid == nullptr) {
        throw std::runtime_error(
            "exact attention forward valid-length allocation failed");
    }

    if (training_mode_ && training_policy::tiled_attention()) {
        Tensor output = Tensor::uninitialized({batch_size, seq_len, n_heads, head_dim}, Device::GPU);
        Tensor lse = Tensor::uninitialized({batch_size, seq_len, n_heads}, Device::GPU);
        if (!launch_attn_tiled_forward(q_heads.raw_data(), k_heads.raw_data(), v_heads.raw_data(),
            device_valid, output.raw_data(), lse.raw_data(), batch_size, seq_len, n_heads,
            n_kv_heads, head_dim, kv_group_size, sliding_window_, scale))
            throw std::runtime_error("Tiled attention training forward rejected shape or launch");
        saved_attn_lse_ = lse;
        saved_attn_output_ = output;
        return output.reshape({batch_size, seq_len, d_model});
    }
    const int batch_heads = batch_size * n_heads;
    Tensor query_permuted = Tensor::uninitialized(
        {batch_heads, seq_len, head_dim}, Device::GPU);
    Tensor key_transposed = Tensor::uninitialized(
        {batch_heads, head_dim, seq_len}, Device::GPU);
    Tensor value_permuted = Tensor::uninitialized(
        {batch_heads, seq_len, head_dim}, Device::GPU);
    launch_attn_gather_heads(
        query_permuted.raw_data(), q_heads.raw_data(),
        batch_size, seq_len, n_heads, head_dim, n_heads, 1, 0);
    launch_attn_gather_heads(
        key_transposed.raw_data(), k_heads.raw_data(),
        batch_size, seq_len, n_heads, head_dim, n_kv_heads,
        kv_group_size, 1);
    launch_attn_gather_heads(
        value_permuted.raw_data(), v_heads.raw_data(),
        batch_size, seq_len, n_heads, head_dim, n_kv_heads,
        kv_group_size, 0);
    attn_train_check_cuda("exact attention head gather forward");

    Tensor probabilities =
        query_permuted.matmul(key_transposed);
    launch_attn_masked_softmax(
        probabilities.raw_data(), device_valid, batch_size, n_heads,
        seq_len, scale, sliding_window_);
    Tensor output_permuted =
        probabilities.matmul(value_permuted);
    Tensor output_heads = Tensor::uninitialized(
        {batch_size, seq_len, n_heads, head_dim}, Device::GPU);
    launch_attn_unpermute_heads(
        output_heads.raw_data(), output_permuted.raw_data(),
        batch_size, n_heads, seq_len, head_dim);
    attn_train_check_cuda("exact attention softmax/value forward");
    return output_heads.reshape(
        {batch_size, seq_len, d_model});
#else
    (void)q_flat;
    (void)kv_flat;
    (void)batch_size;
    (void)seq_len;
    (void)kv_dim;
    (void)scale;
    throw std::runtime_error(
        "exact GPU attention forward requires a GPU build");
#endif
}

Tensor Attention::sparse_forward(const Tensor& input, Context* ctx) {
    (void)ctx;
    const Device original_device = input.get_device();

    Tensor project_input = input;
    int batch_size = 1;
    int seq_len = 1;
    if (input.shape.size() == 1) {
        project_input = input.reshape({1, 1, d_model});
        saved_valid_lengths_ = {1};
    } else if (input.shape.size() == 2) {
        seq_len = input.shape[0];
        project_input = input.reshape({1, seq_len, d_model});
        saved_valid_lengths_ = {seq_len};
    } else {
        batch_size = input.shape[0];
        seq_len = input.shape[1];
        saved_valid_lengths_ =
            normalize_valid_lengths(active_batch_valid_lengths_, batch_size, seq_len);
    }
    saved_input_rank_ = static_cast<int>(input.shape.size());

    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Tensor q_flat = q_down_proj->forward(project_input);
    Tensor kv_flat = kv_down_proj->forward(project_input);

#ifdef USE_CUDA
    if (original_device == Device::GPU && gpu_custom_kernels_supported() &&
        head_dim <= 256 && ssa_top_k_blocks_ >= 0 &&
        ssa_top_k_blocks_ <= 64) {
        Tensor k_flat_gpu = Tensor::uninitialized(
            {batch_size, seq_len, kv_dim}, Device::GPU);
        Tensor v_flat_gpu = Tensor::uninitialized(
            {batch_size, seq_len, kv_dim}, Device::GPU);
        launch_kv_split(k_flat_gpu.raw_data(), v_flat_gpu.raw_data(),
                        kv_flat.raw_data(),
                        static_cast<long long>(batch_size) * seq_len, kv_dim);
        Tensor q_rot_gpu = q_flat.clone();
        Tensor k_rot_gpu = k_flat_gpu.clone();
        ensure_rope_gpu_cache();
        launch_rope_apply(q_rot_gpu.raw_data(), rope_cos_gpu_.raw_data(),
                          rope_sin_gpu_.raw_data(), batch_size, seq_len,
                          n_heads, head_dim, 0, max_seq_len, +1);
        launch_rope_apply(k_rot_gpu.raw_data(), rope_cos_gpu_.raw_data(),
                          rope_sin_gpu_.raw_data(), batch_size, seq_len,
                          n_kv_heads, head_dim, 0, max_seq_len, +1);
        attn_train_check_cuda("SSA split/RoPE forward");

        saved_q_rot_ = q_rot_gpu.reshape(
            {batch_size, seq_len, n_heads, head_dim});
        saved_k_rot_ = k_rot_gpu.reshape(
            {batch_size, seq_len, n_kv_heads, head_dim});
        saved_v_heads_ = v_flat_gpu.reshape(
            {batch_size, seq_len, n_kv_heads, head_dim});
        saved_attn_probs_ = Tensor();

        const int BH = batch_size * n_heads;
        Tensor q_permuted = Tensor::uninitialized(
            {BH, seq_len, head_dim}, Device::GPU);
        Tensor k_permuted = Tensor::uninitialized(
            {BH, seq_len, head_dim}, Device::GPU);
        Tensor v_permuted = Tensor::uninitialized(
            {BH, seq_len, head_dim}, Device::GPU);
        launch_attn_gather_heads(
            q_permuted.raw_data(), saved_q_rot_.raw_data(), batch_size,
            seq_len, n_heads, head_dim, n_heads, 1, 0);
        launch_attn_gather_heads(
            k_permuted.raw_data(), saved_k_rot_.raw_data(), batch_size,
            seq_len, n_heads, head_dim, n_kv_heads, kv_group_size, 0);
        launch_attn_gather_heads(
            v_permuted.raw_data(), saved_v_heads_.raw_data(), batch_size,
            seq_len, n_heads, head_dim, n_kv_heads, kv_group_size, 0);

        Tensor route_permuted = q_permuted;
        if (ssa_wsel_.data.size > 0) {
            if (ssa_wsel_.data.get_device() != Device::GPU) {
                throw std::runtime_error(
                    "SSA selector parameter must share the GPU compute device");
            }
            route_permuted =
                matmul_nt(q_permuted, ssa_wsel_.data);
        }
        Tensor output_permuted =
            Tensor::zeros({BH, seq_len, head_dim}, Device::GPU);
        const int block_size = std::max(ssa_block_size_, 1);
        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len = std::clamp(
                saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            if (valid_len <= 0) continue;
            const int block_count =
                gpu::ceil_div_positive(valid_len, block_size);
            for (int head = 0; head < n_heads; ++head) {
                const int bh = batch * n_heads + head;
                const size_t offset =
                    static_cast<size_t>(bh) * seq_len * head_dim;
                Tensor block_means(
                    {block_count, head_dim}, Device::GPU);
                cuda::launch_sparse_block_means(
                    k_permuted.raw_data() + offset,
                    block_means.raw_data(), valid_len, head_dim, block_size);
                cuda::launch_sparse_selective_attention(
                    q_permuted.raw_data() + offset,
                    k_permuted.raw_data() + offset,
                    v_permuted.raw_data() + offset,
                    route_permuted.raw_data() + offset,
                    block_means.raw_data(), output_permuted.raw_data() + offset,
                    valid_len, head_dim, block_size, ssa_top_k_blocks_,
                    ssa_local_blocks_, ssa_sink_blocks_, scale);
            }
        }
        Tensor output_heads = Tensor::uninitialized(
            {batch_size, seq_len, n_heads, head_dim}, Device::GPU);
        launch_attn_unpermute_heads(
            output_heads.raw_data(), output_permuted.raw_data(), batch_size,
            n_heads, seq_len, head_dim);
        attn_train_check_cuda("SSA selective forward");
        Tensor projected = out_proj->forward(
            output_heads.reshape({batch_size, seq_len, d_model}));
        if (saved_input_rank_ == 1) return projected.reshape({d_model});
        if (saved_input_rank_ == 2) return projected.reshape({seq_len, d_model});
        return projected;
    }
#endif

    if (original_device == Device::GPU && strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU SSA requires custom kernels, head_dim <= 256 and "
            "top_k_blocks <= 64");
    }

    // Sparse attention math (block selection + softmax) runs on host pointers;
    // bring projections to CPU for the per-head extraction and for backward
    // (Attention::backward consumes saved_*_ as CPU tensors).
    Tensor q_host = q_flat.get_device() == Device::GPU ? q_flat.cpu() : q_flat;
    Tensor kv_host = kv_flat.get_device() == Device::GPU ? kv_flat.cpu() : kv_flat;

    Tensor k_flat({batch_size, seq_len, kv_dim}, Device::CPU);
    Tensor v_flat({batch_size, seq_len, kv_dim}, Device::CPU);
    {
        const float* kv_ptr = kv_host.data();
        float* k_ptr = k_flat.data();
        float* v_ptr = v_flat.data();
        const size_t token_stride = static_cast<size_t>(2 * kv_dim);
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int token = 0; token < seq_len; ++token) {
                const size_t in_off =
                    (static_cast<size_t>(batch) * seq_len + token) * token_stride;
                const size_t out_off =
                    (static_cast<size_t>(batch) * seq_len + token) *
                    static_cast<size_t>(kv_dim);
                std::memcpy(k_ptr + out_off, kv_ptr + in_off,
                            static_cast<size_t>(kv_dim) * sizeof(float));
                std::memcpy(v_ptr + out_off, kv_ptr + in_off + static_cast<size_t>(kv_dim),
                            static_cast<size_t>(kv_dim) * sizeof(float));
            }
        }
    }

    Tensor q_heads = q_host.reshape({batch_size, seq_len, n_heads, head_dim});
    Tensor k_heads = k_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
    Tensor v_heads = v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
    auto [q_rot, k_rot] = apply_rope(q_heads, k_heads, 0);

    saved_q_rot_ = q_rot;
    saved_k_rot_ = k_rot;
    saved_v_heads_ = v_heads;
    saved_attn_probs_ = Tensor();

    SparseAttentionConfig cfg;
    cfg.block_size = ssa_block_size_;
    cfg.top_k_blocks = ssa_top_k_blocks_;
    cfg.local_blocks = ssa_local_blocks_;
    cfg.sink_blocks = ssa_sink_blocks_;
    cfg.scale = scale;

    // Learned block-selection routing, materialised on the compute device so the
    // GPU kernel path can score blocks with it (empty/unmatched shape -> raw q).
    Tensor wsel_dev;
    const Tensor* wsel_ptr = nullptr;
    if (ssa_wsel_.data.size > 0) {
        if (original_device == Device::GPU) {
            wsel_dev = ssa_wsel_.data.get_device() == Device::GPU
                           ? ssa_wsel_.data
                           : ssa_wsel_.data.to(Device::GPU);
        } else {
            wsel_dev = ssa_wsel_.data.get_device() == Device::GPU
                           ? ssa_wsel_.data.cpu()
                           : ssa_wsel_.data;
        }
        wsel_ptr = &wsel_dev;
    }

    Tensor output_heads({batch_size, seq_len, n_heads, head_dim}, Device::CPU);
    std::fill_n(output_heads.data(),
                static_cast<size_t>(batch_size) * seq_len * n_heads * head_dim, 0.0f);
    float* out_ptr = output_heads.data();
    const float* q_ptr = q_rot.data();
    const float* k_ptr = k_rot.data();
    const float* v_ptr = v_heads.data();

    for (int batch = 0; batch < batch_size; ++batch) {
        const int valid_len =
            std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
        if (valid_len <= 0) continue;
        for (int head = 0; head < n_heads; ++head) {
            const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
            Tensor Qh({valid_len, head_dim}, Device::CPU);
            Tensor Kh({valid_len, head_dim}, Device::CPU);
            Tensor Vh({valid_len, head_dim}, Device::CPU);
            float* qh = Qh.data();
            float* kh = Kh.data();
            float* vh = Vh.data();
            for (int t = 0; t < valid_len; ++t) {
                const size_t q_off =
                    (((static_cast<size_t>(batch) * seq_len + t) * n_heads) + head) *
                    static_cast<size_t>(head_dim);
                const size_t kv_off =
                    (((static_cast<size_t>(batch) * seq_len + t) * n_kv_heads) + kv_head) *
                    static_cast<size_t>(head_dim);
                for (int dd = 0; dd < head_dim; ++dd) {
                    qh[static_cast<size_t>(t) * head_dim + dd] = q_ptr[q_off + dd];
                    kh[static_cast<size_t>(t) * head_dim + dd] = k_ptr[kv_off + dd];
                    vh[static_cast<size_t>(t) * head_dim + dd] = v_ptr[kv_off + dd];
                }
            }
            Tensor Oh;
            if (original_device == Device::GPU) {
                Oh = sparse_selective_attention(Qh.to(Device::GPU), Kh.to(Device::GPU),
                                                Vh.to(Device::GPU), cfg, nullptr, wsel_ptr)
                         .cpu();
            } else {
                Oh = sparse_selective_attention(Qh, Kh, Vh, cfg, nullptr, wsel_ptr);
            }
            const float* oh = Oh.data();
            for (int t = 0; t < valid_len; ++t) {
                const size_t o_off =
                    (((static_cast<size_t>(batch) * seq_len + t) * n_heads) + head) *
                    static_cast<size_t>(head_dim);
                for (int dd = 0; dd < head_dim; ++dd) {
                    out_ptr[o_off + dd] = oh[static_cast<size_t>(t) * head_dim + dd];
                }
            }
        }
    }

    Tensor output_2d = output_heads.reshape({batch_size, seq_len, d_model});
    Tensor projected_input =
        (original_device == Device::GPU) ? output_2d.to(Device::GPU) : output_2d;
    Tensor projected = out_proj->forward(projected_input);
    if (saved_input_rank_ == 1) {
        return projected.reshape({d_model});
    }
    if (saved_input_rank_ == 2) {
        return projected.reshape({seq_len, d_model});
    }
    return projected;
}

Tensor Attention::forward(const Tensor& input, Context* ctx) {
    (void)ctx;
    const auto rdna_policy = attention_training::policy();
    saved_rdna_training_ = false;
    if (rdna_policy != attention_training::Policy::FP32 && training_mode_) {
        if (input.get_device() != Device::GPU || streaming_inference_ || sparse_enabled_)
            throw std::invalid_argument("RDNA attention training requires dense GPU non-streaming execution");
#ifdef USE_CUDA
        const int rank = static_cast<int>(input.shape.size());
        if (rank < 1 || rank > 3)
            throw std::invalid_argument("RDNA attention input rank must be 1, 2 or 3");
        const int batch = rank == 3 ? input.shape[0] : 1;
        const int sequence = rank == 1 ? 1 : rank == 2 ? input.shape[0] : input.shape[1];
        ensure_freqs_capacity(sequence);
        ensure_rope_gpu_cache();
        if (!rdna_training_)
            rdna_training_ = std::make_unique<attention_training::Provider>(
                *q_down_proj, *kv_down_proj, *out_proj);
        attention_training::Config c;
        c.shape = {batch, sequence, n_heads, n_kv_heads, head_dim,
            sliding_window_ > 0 ? sliding_window_ : sequence,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            rdna_policy == attention_training::Policy::RdnaBF16
                ? attention_rdna::Precision::BF16 : attention_rdna::Precision::FP16};
        std::vector<int> valid = rank == 3 ? active_batch_valid_lengths_
                                         : std::vector<int>{sequence};
        // Model backward tickets enforce one live forward. Checkpoint replay
        // deliberately cancels the activation tape; sticky group status survives.
        rdna_training_->cancel_pending();
        Tensor result = rdna_training_->forward(input, c,
            rope_cos_gpu_.reshape({max_seq_len, head_dim / 2}),
            rope_sin_gpu_.reshape({max_seq_len, head_dim / 2}), valid);
        saved_rdna_training_ = true;
        return result;
#else
        throw std::runtime_error("RDNA attention requested without a GPU build");
#endif
    }
    if (rdna_training_) rdna_training_->cancel_pending();
    saved_attn_lse_ = Tensor();
    saved_attn_output_ = Tensor();
    const bool one_token = input.shape.size() == 1 ||
        (input.shape.size() == 2 && input.shape[0] == 1) ||
        (input.shape.size() == 3 && input.shape[1] == 1);
    if (compact_kv_ && (input.get_device() != Device::GPU || !streaming_inference_ ||
                       training_mode_ || sparse_enabled_ || !one_token))
        materialize_compact_kv();
    saved_q_rot_ = Tensor();
    saved_k_rot_ = Tensor();
    saved_v_heads_ = Tensor();
    saved_attn_probs_ = Tensor();
    saved_valid_lengths_.clear();
    // SSA: route non-streaming forward through the dedicated sparse path.  The
    // dense fast-paths below do not apply SSA (the batch/rank-3 path is
    // dense-only), so without this branch a sparse-enabled model silently runs
    // dense.  Streaming decode keeps the existing per-step path.
    if (sparse_enabled_ && !streaming_inference_ &&
        (input.shape.size() == 1 || input.shape.size() == 2 ||
         input.shape.size() == 3)) {
        return sparse_forward(input, ctx);
    }
    // K5: ALL training forwards take the exact path, which saves q_rot/k_rot/
    // v_heads + valid_lengths and pairs with the exact softmax-jacobian backward.
    // The old gate also required exact_training_path_; with it false, training
    // fell through to the dense forward that does NOT save the cache, so backward
    // used the jacobian-FREE fallback (no QK^T / probs gradient — attention could
    // not learn content-based recall). The serialized
    // use_exact_attention_training option is now compatibility metadata:
    // training is unconditionally exact. Inference (training_mode_==false)
    // still uses the validated dense/cached paths (no backward needed there).
    if (training_mode_ && !streaming_inference_ &&
        (input.shape.size() == 1 || input.shape.size() == 2 || input.shape.size() == 3)) {
        const Device original_device = input.get_device();
        Tensor project_input = input;
        int batch_size = 1;
        int seq_len = 1;
        if (input.shape.size() == 1) {
            project_input = input.reshape({1, 1, d_model});
            batch_size = 1;
            seq_len = 1;
            saved_valid_lengths_ = {1};
        } else if (input.shape.size() == 2) {
            seq_len = input.shape[0];
            project_input = input.reshape({1, seq_len, d_model});
            batch_size = 1;
            saved_valid_lengths_ = {seq_len};
        } else {
            batch_size = input.shape[0];
            seq_len = input.shape[1];
            saved_valid_lengths_ =
                normalize_valid_lengths(active_batch_valid_lengths_, batch_size, seq_len);
        }

        saved_input_rank_ = static_cast<int>(input.shape.size());
        const int kv_dim = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        Tensor q_flat = q_down_proj->forward(project_input);
        Tensor kv_flat = kv_down_proj->forward(project_input);
#ifdef USE_CUDA
        if (original_device == Device::GPU &&
            q_flat.get_device() == Device::GPU &&
            kv_flat.get_device() == Device::GPU &&
            batch_size > 0 &&
            seq_len > 0 &&
            gpu_custom_kernels_supported()) {
            Tensor attention_output = forward_exact_gpu(
                q_flat, kv_flat, batch_size, seq_len, kv_dim, scale);
            Tensor projected = out_proj->forward(attention_output);
            if (saved_input_rank_ == 1) {
                return projected.reshape({d_model});
            }
            if (saved_input_rank_ == 2) {
                return projected.reshape({seq_len, d_model});
            }
            return projected;
        }
#endif
        if (original_device == Device::GPU &&
            strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU exact attention forward has no eligible "
                "device path for the active sequence/configuration");
        }
        // CPU reference path. GPU training returned above after constructing
        // the same exact Q/K/V cache and valid-length mask on-device.
        Tensor output_heads;
        Tensor q_host = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
        Tensor kv_host = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;
        Tensor k_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        Tensor v_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        {
            const float* kv_ptr = kv_host.data();
            float* k_ptr = k_flat.data();
            float* v_ptr = v_flat.data();
            const size_t token_stride = static_cast<size_t>(2 * kv_dim);
            for (int batch = 0; batch < batch_size; ++batch) {
                for (int token = 0; token < seq_len; ++token) {
                    const size_t token_offset =
                        (static_cast<size_t>(batch) * seq_len + token) * token_stride;
                    const size_t out_offset =
                        (static_cast<size_t>(batch) * seq_len + token) *
                        static_cast<size_t>(kv_dim);
                    std::memcpy(
                        k_ptr + out_offset,
                        kv_ptr + token_offset,
                        static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(
                        v_ptr + out_offset,
                        kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                        static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }
        }

        Tensor q_heads = q_host.reshape({batch_size, seq_len, n_heads, head_dim});
        Tensor k_heads = k_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        Tensor v_heads = v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        auto [q_rot, k_rot] = apply_rope(q_heads, k_heads, 0);

        output_heads = Tensor::uninitialized(
            {batch_size, seq_len, n_heads, head_dim},
            q_rot.get_device());
        float* out_ptr = output_heads.data();
        const float* q_ptr = q_rot.data();
        const float* k_ptr = k_rot.data();
        const float* v_ptr = v_heads.data();
        std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> probs(static_cast<size_t>(seq_len), 0.0f);

        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len = std::clamp(
                saved_valid_lengths_[static_cast<size_t>(batch)],
                0, seq_len);
            for (int head = 0; head < n_heads; ++head) {
                    const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                    for (int i = 0; i < seq_len; ++i) {
                        const int first_key = std::max(0, i - sliding_window_ + 1);
                        const size_t out_row_offset =
                            (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                            static_cast<size_t>(head_dim);
                        if (i >= valid_len) {
                            std::fill_n(out_ptr + out_row_offset, head_dim, 0.0f);
                            continue;
                        }

                        float max_s = -1e30f;
                        for (int j = 0; j < seq_len; ++j) {
                            float score = -1e9f;
                            if (j >= first_key && j < valid_len && j <= i) {
                                float dot = 0.0f;
                                const size_t q_offset =
                                    (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                    static_cast<size_t>(head_dim);
                                const size_t k_offset =
                                    (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                     kv_head) *
                                    static_cast<size_t>(head_dim);
                                for (int dim = 0; dim < head_dim; ++dim) {
                                    dot += q_ptr[q_offset + dim] * k_ptr[k_offset + dim];
                                }
                                score = dot * scale;
                            }
                            scores[static_cast<size_t>(j)] = score;
                            max_s = std::max(max_s, score);
                        }

                        float sum_exp = 0.0f;
                        for (int j = 0; j < seq_len; ++j) {
                            float value =
                                (j >= first_key && j < valid_len && j <= i)
                                    ? std::exp(scores[static_cast<size_t>(j)] - max_s)
                                    : 0.0f;
                            probs[static_cast<size_t>(j)] = value;
                            sum_exp += value;
                        }
                        const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                        for (int j = 0; j < seq_len; ++j) {
                            probs[static_cast<size_t>(j)] *= inv_sum;
                        }

                        for (int dim = 0; dim < head_dim; ++dim) {
                            float acc = 0.0f;
                            for (int j = 0; j < valid_len; ++j) {
                                const size_t v_offset =
                                    (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                     kv_head) *
                                        static_cast<size_t>(head_dim) +
                                    static_cast<size_t>(dim);
                                acc += probs[static_cast<size_t>(j)] * v_ptr[v_offset];
                            }
                            out_ptr[out_row_offset + static_cast<size_t>(dim)] = acc;
                        }
                    }
                }
            }

        saved_q_rot_ = q_rot;
        saved_k_rot_ = k_rot;
        saved_v_heads_ = v_heads;
        saved_attn_probs_ = Tensor();
        Tensor output_2d =
            output_heads.reshape({batch_size, seq_len, d_model});
        Tensor projected_input =
            original_device == Device::GPU
                ? output_2d.to(Device::GPU)
                : output_2d;
        Tensor projected = out_proj->forward(projected_input);
        if (saved_input_rank_ == 1) {
            return projected.reshape({d_model});
        }
        if (saved_input_rank_ == 2) {
            return projected.reshape({seq_len, d_model});
        }
        return projected;
    }

    if (input.shape.size() == 3) {
        const Device original_device = input.get_device();
        const int batch_size = input.shape[0];
        const int seq_len = input.shape[1];
        const int kv_dim = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        Tensor q_flat = q_down_proj->forward(input);
        Tensor kv_flat = kv_down_proj->forward(input);

        if (streaming_inference_ && seq_len == 1) {
#ifdef USE_CUDA
            if (original_device == Device::GPU &&
                q_flat.get_device() == Device::GPU &&
                kv_flat.get_device() == Device::GPU && batch_size > 0 &&
                sparse_enabled_ && gpu_custom_kernels_supported() &&
                head_dim <= 256 && ssa_top_k_blocks_ >= 0 &&
                ssa_top_k_blocks_ <= 64) {
                ensure_kv_cache_capacity(
                    cached_tokens_ + 1, Device::GPU, batch_size);
                const size_t kv_row_stride =
                    static_cast<size_t>(2 * kv_dim);
                const size_t cache_row_stride =
                    static_cast<size_t>(cache_capacity_tokens_) * kv_dim;
                for (int batch = 0; batch < batch_size; ++batch) {
                    launch_gqa_append_kv_cache_kernel(
                        kv_flat.raw_data() +
                            static_cast<size_t>(batch) * kv_row_stride,
                        key_cache_buffer_.raw_data() +
                            static_cast<size_t>(batch) * cache_row_stride,
                        value_cache_buffer_.raw_data() +
                            static_cast<size_t>(batch) * cache_row_stride,
                        cached_tokens_, n_kv_heads, head_dim, theta);
                }
                Tensor q_rot_gpu = q_flat.clone();
                ensure_rope_gpu_cache();
                launch_rope_apply(
                    q_rot_gpu.raw_data(), rope_cos_gpu_.raw_data(),
                    rope_sin_gpu_.raw_data(), batch_size, 1, n_heads,
                    head_dim, cached_tokens_, max_seq_len, +1);
                Tensor q_heads = q_rot_gpu.reshape(
                    {batch_size, n_heads, 1, head_dim});
                Tensor route_heads = q_heads;
                if (ssa_wsel_.data.size > 0) {
                    if (ssa_wsel_.data.get_device() != Device::GPU) {
                        throw std::runtime_error(
                            "SSA selector parameter must share the GPU compute device");
                    }
                    route_heads =
                        matmul_nt(q_heads, ssa_wsel_.data);
                }
                const int total_tokens = cached_tokens_ + 1;
                const int block_size = std::max(ssa_block_size_, 1);
                const int block_count =
                    gpu::ceil_div_positive(total_tokens, block_size);
                Tensor output_heads = Tensor::zeros(
                    {batch_size, 1, n_heads, head_dim}, Device::GPU);
                for (int batch = 0; batch < batch_size; ++batch) {
                    Tensor k_permuted = Tensor::uninitialized(
                        {n_heads, total_tokens, head_dim}, Device::GPU);
                    Tensor v_permuted = Tensor::uninitialized(
                        {n_heads, total_tokens, head_dim}, Device::GPU);
                    launch_attn_gather_heads(
                        k_permuted.raw_data(),
                        key_cache_buffer_.raw_data() +
                            static_cast<size_t>(batch) * cache_row_stride,
                        1, total_tokens, n_heads, head_dim, n_kv_heads,
                        kv_group_size, 0);
                    launch_attn_gather_heads(
                        v_permuted.raw_data(),
                        value_cache_buffer_.raw_data() +
                            static_cast<size_t>(batch) * cache_row_stride,
                        1, total_tokens, n_heads, head_dim, n_kv_heads,
                        kv_group_size, 0);
                    for (int head = 0; head < n_heads; ++head) {
                        const size_t cache_offset =
                            static_cast<size_t>(head) * total_tokens * head_dim;
                        const size_t head_offset =
                            (static_cast<size_t>(batch) * n_heads + head) *
                            head_dim;
                        Tensor block_means(
                            {block_count, head_dim}, Device::GPU);
                        cuda::launch_sparse_block_means(
                            k_permuted.raw_data() + cache_offset,
                            block_means.raw_data(), total_tokens, head_dim,
                            block_size);
                        cuda::launch_sparse_selective_attention_decode(
                            q_heads.raw_data() + head_offset,
                            k_permuted.raw_data() + cache_offset,
                            v_permuted.raw_data() + cache_offset,
                            route_heads.raw_data() + head_offset,
                            block_means.raw_data(),
                            output_heads.raw_data() + head_offset,
                            total_tokens, head_dim, block_size,
                            ssa_top_k_blocks_, ssa_local_blocks_,
                            ssa_sink_blocks_, scale);
                    }
                }
                ++cached_tokens_;
                attn_train_check_cuda("batched SSA streaming decode");
                return out_proj->forward(
                    output_heads.reshape({batch_size, 1, d_model}));
            }
            if (original_device == Device::GPU &&
                q_flat.get_device() == Device::GPU &&
                kv_flat.get_device() == Device::GPU &&
                batch_size > 0 &&
                !sparse_enabled_ &&
                gpu_custom_kernels_supported()) {
                const bool compact = compact_kv_requested();
                if (compact) ensure_compact_kv_capacity(cached_tokens_ + 1, batch_size);
                else ensure_kv_cache_capacity(cached_tokens_ + 1, Device::GPU, batch_size);
                Tensor output_gpu = Tensor::uninitialized({batch_size, 1, d_model}, Device::GPU);
                launch_gqa_append_decode_batch(q_flat.raw_data(), kv_flat.raw_data(),
                    compact ? compact_kv_->keys() : key_cache_buffer_.raw_data(),
                    compact ? compact_kv_->values() : value_cache_buffer_.raw_data(),
                    output_gpu.raw_data(), batch_size, cache_capacity_tokens_, cached_tokens_,
                    n_heads, n_kv_heads, head_dim, kv_group_size, theta, sliding_window_, compact);
                ++cached_tokens_;

                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("Batched cached GQA attention kernel launch failed: ") +
                        cudaGetErrorString(status));
                }
                if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                    if (std::string(sync_env) == "1") {
                        status = cudaDeviceSynchronize();
                        if (status != cudaSuccess) {
                            throw std::runtime_error(
                                std::string("Batched cached GQA attention kernel sync failed: ") +
                                cudaGetErrorString(status));
                        }
                        record_gpu_device_synchronization();
                    }
                }

                return out_proj->forward(output_gpu);
            }
#endif

            if (sparse_enabled_) {
                throw std::runtime_error(
                    "Batched streaming SSA is unsupported; use single-sequence "
                    "decode or disable sparse attention");
            }
            if (original_device == Device::GPU &&
                strict_gpu_execution()) {
                throw std::runtime_error(
                    "Strict GPU batched streaming attention has no eligible "
                    "device path for the active sequence/configuration");
            }

            Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
            Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;
            Tensor k_flat({batch_size, 1, kv_dim}, Device::CPU);
            Tensor v_flat({batch_size, 1, kv_dim}, Device::CPU);
            {
                const float* kv_ptr = kv_cpu.data();
                float* k_ptr = k_flat.data();
                float* v_ptr = v_flat.data();
                const size_t token_stride = static_cast<size_t>(2 * kv_dim);
                for (int batch = 0; batch < batch_size; ++batch) {
                    const size_t token_offset =
                        static_cast<size_t>(batch) * token_stride;
                    const size_t out_offset = static_cast<size_t>(batch) *
                                              static_cast<size_t>(kv_dim);
                    std::memcpy(k_ptr + out_offset,
                                kv_ptr + token_offset,
                                static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(v_ptr + out_offset,
                                kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                                static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }

            Tensor q4 = q_cpu.reshape({batch_size, 1, n_heads, head_dim});
            Tensor k4 = k_flat.reshape({batch_size, 1, n_kv_heads, head_dim});
            Tensor v4 = v_flat.reshape({batch_size, 1, n_kv_heads, head_dim});
            auto [q_rot, k_rot] = apply_rope(q4, k4, cached_tokens_);
            append_kv_cache_batch_tokens(k_rot.data(), Device::CPU, v4.data(), Device::CPU, batch_size);

            Tensor output_cpu({batch_size, 1, d_model}, Device::CPU);
            float* out_ptr = output_cpu.data();
            const float* q_ptr = q_rot.data();
            for (int batch = 0; batch < batch_size; ++batch) {
                std::vector<float> token_scores(static_cast<size_t>(cached_tokens_), 0.0f);
                const int first_token = std::max(0, cached_tokens_ - sliding_window_);
                for (int head = 0; head < n_heads; ++head) {
                    const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                    float max_s = -1e30f;
                    for (int t = first_token; t < cached_tokens_; ++t) {
                        float dot = 0.0f;
                        const float* cached_key_token =
                            kv_cache_token_ptr(key_cache_buffer_, batch, t);
                        const size_t q_offset =
                            (((static_cast<size_t>(batch) * 1) * n_heads) + head) *
                            static_cast<size_t>(head_dim);
                        const size_t kv_offset =
                            static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                        for (int d = 0; d < head_dim; ++d) {
                            dot += q_ptr[q_offset + static_cast<size_t>(d)] *
                                   cached_key_token[kv_offset + static_cast<size_t>(d)];
                        }
                        token_scores[static_cast<size_t>(t)] = dot * scale;
                        max_s = std::max(max_s, token_scores[static_cast<size_t>(t)]);
                    }
                    float sum_exp = 0.0f;
                    for (int t = first_token; t < cached_tokens_; ++t) {
                        float& score_value = token_scores[static_cast<size_t>(t)];
                        score_value = std::exp(score_value - max_s);
                        sum_exp += score_value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int t = first_token; t < cached_tokens_; ++t) {
                        token_scores[static_cast<size_t>(t)] *= inv_sum;
                    }
                    const size_t out_offset =
                        (((static_cast<size_t>(batch) * 1) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    float* out_head = out_ptr + out_offset;
                    std::fill_n(out_head, head_dim, 0.0f);
                    const size_t kv_offset =
                        static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                    for (int t = first_token; t < cached_tokens_; ++t) {
                        const float* cached_value_token =
                            kv_cache_token_ptr(value_cache_buffer_, batch, t) + kv_offset;
                        const float score = token_scores[static_cast<size_t>(t)];
                        for (int d = 0; d < head_dim; ++d) {
                            out_head[d] += score * cached_value_token[static_cast<size_t>(d)];
                        }
                    }
                }
            }

            Tensor output =
                (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
            return out_proj->forward(output);
        }

        if (streaming_inference_) {
            throw std::runtime_error(
                "Attention::forward rank-3 batch path only supports streaming for single-token steps");
        }

#ifdef USE_CUDA
        if (original_device == Device::GPU &&
            q_flat.get_device() == Device::GPU &&
            kv_flat.get_device() == Device::GPU &&
            batch_size > 0 &&
            seq_len > 0 &&
            gpu_custom_kernels_supported()) {
            saved_input_rank_ = 3;
            saved_valid_lengths_ = normalize_valid_lengths(
                active_batch_valid_lengths_, batch_size, seq_len);
            Tensor attention_output = forward_exact_gpu(
                q_flat, kv_flat, batch_size, seq_len, kv_dim, scale);
            return out_proj->forward(attention_output);
        }
#endif

        if (original_device == Device::GPU &&
            strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU batched attention forward has no eligible "
                "device path for the active sequence/configuration");
        }

        Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
        Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;

        Tensor k_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        Tensor v_flat({batch_size, seq_len, kv_dim}, Device::CPU);
        {
            const float* kv_ptr = kv_cpu.data();
            float* k_ptr = k_flat.data();
            float* v_ptr = v_flat.data();
            const size_t token_stride = static_cast<size_t>(2 * kv_dim);
            for (int batch = 0; batch < batch_size; ++batch) {
                for (int token = 0; token < seq_len; ++token) {
                    const size_t token_offset =
                        (static_cast<size_t>(batch) * seq_len + token) * token_stride;
                    const size_t out_offset =
                        (static_cast<size_t>(batch) * seq_len + token) *
                        static_cast<size_t>(kv_dim);
                    std::memcpy(
                        k_ptr + out_offset,
                        kv_ptr + token_offset,
                        static_cast<size_t>(kv_dim) * sizeof(float));
                    std::memcpy(
                        v_ptr + out_offset,
                        kv_ptr + token_offset + static_cast<size_t>(kv_dim),
                        static_cast<size_t>(kv_dim) * sizeof(float));
                }
            }
        }

        Tensor q4 = q_cpu.reshape({batch_size, seq_len, n_heads, head_dim});
        Tensor k4 = k_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        Tensor v4 = v_flat.reshape({batch_size, seq_len, n_kv_heads, head_dim});
        auto [q_rot, k_rot] = apply_rope(q4, k4, 0);

        Tensor output_cpu({batch_size, seq_len, d_model}, Device::CPU);
        float* out_ptr = output_cpu.data();
        const float* q_ptr = q_rot.data();
        const float* k_ptr = k_rot.data();
        const float* v_ptr = v4.data();
        std::vector<float> scores(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len),
                                  0.0f);

        for (int batch = 0; batch < batch_size; ++batch) {
            for (int head = 0; head < n_heads; ++head) {
                const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                std::fill(scores.begin(), scores.end(), 0.0f);

                for (int i = 0; i < seq_len; ++i) {
                    const int first_key = std::max(0, i - sliding_window_ + 1);
                    float max_s = -1e30f;
                    for (int j = 0; j < seq_len; ++j) {
                        float score = -1e9f;
                        if (j >= first_key && j <= i) {
                            float dot = 0.0f;
                            const size_t q_offset =
                                (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                static_cast<size_t>(head_dim);
                            const size_t k_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                static_cast<size_t>(head_dim);
                            for (int dim = 0; dim < head_dim; ++dim) {
                                dot += q_ptr[q_offset + dim] * k_ptr[k_offset + dim];
                            }
                            score = dot * scale;
                        }
                        scores[static_cast<size_t>(i) * seq_len + j] = score;
                        max_s = std::max(max_s, score);
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        const float value =
                            (j >= first_key && j <= i)
                                ? std::exp(scores[static_cast<size_t>(i) * seq_len + j] - max_s)
                                : 0.0f;
                        scores[static_cast<size_t>(i) * seq_len + j] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int j = 0; j < seq_len; ++j) {
                        scores[static_cast<size_t>(i) * seq_len + j] *= inv_sum;
                    }

                    for (int dim = 0; dim < head_dim; ++dim) {
                        float acc = 0.0f;
                        for (int j = 0; j < seq_len; ++j) {
                            const size_t v_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                    static_cast<size_t>(head_dim) +
                                static_cast<size_t>(dim);
                            acc += scores[static_cast<size_t>(i) * seq_len + j] * v_ptr[v_offset];
                        }
                        const size_t out_offset =
                            (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                                static_cast<size_t>(head_dim) +
                            static_cast<size_t>(dim);
                        out_ptr[out_offset] = acc;
                    }
                }
            }
        }

        Tensor output =
            (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
        return out_proj->forward(output);
    }
    // input shape: [seq_len, d_model]
    // Inferimos seq_len a partir das dimensões do tensor
    const Device original_device = input.get_device();
    Tensor input_2d = input;
    if (input.shape.size() == 1) {
        input_2d = input.reshape({1, d_model});
    }
    const int seq_len = input_2d.shape[0];
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // 1. Projeções Q, K, V
    auto grouped = !training_mode_ && streaming_inference_ && seq_len == 1
        ? decode_projections_.forward(input_2d, {q_down_proj.get(), kv_down_proj.get()})
        : std::vector<Tensor>{};
    Tensor q_flat = grouped.empty() ? q_down_proj->forward(input_2d) : grouped[0];
    Tensor kv_flat = grouped.empty() ? kv_down_proj->forward(input_2d) : grouped[1];
#ifdef USE_CUDA
    if (original_device == Device::GPU &&
        q_flat.get_device() == Device::GPU &&
        kv_flat.get_device() == Device::GPU &&
        seq_len > 0 &&
        seq_len <= 4096 &&
        gpu_custom_kernels_supported()) {
        if (streaming_inference_ && seq_len == 1 && sparse_enabled_ &&
            head_dim <= 256 && ssa_top_k_blocks_ >= 0 &&
            ssa_top_k_blocks_ <= 64) {
            ensure_kv_cache_capacity(cached_tokens_ + 1, Device::GPU);
            launch_gqa_append_kv_cache_kernel(
                kv_flat.raw_data(), key_cache_buffer_.raw_data(),
                value_cache_buffer_.raw_data(), cached_tokens_, n_kv_heads,
                head_dim, theta);
            Tensor q_rot_gpu = q_flat.clone();
            ensure_rope_gpu_cache();
            launch_rope_apply(
                q_rot_gpu.raw_data(), rope_cos_gpu_.raw_data(),
                rope_sin_gpu_.raw_data(), 1, 1, n_heads, head_dim,
                cached_tokens_, max_seq_len, +1);
            const int total_tokens = cached_tokens_ + 1;
            Tensor k_permuted = Tensor::uninitialized(
                {n_heads, total_tokens, head_dim}, Device::GPU);
            Tensor v_permuted = Tensor::uninitialized(
                {n_heads, total_tokens, head_dim}, Device::GPU);
            launch_attn_gather_heads(
                k_permuted.raw_data(), key_cache_buffer_.raw_data(), 1,
                total_tokens, n_heads, head_dim, n_kv_heads, kv_group_size, 0);
            launch_attn_gather_heads(
                v_permuted.raw_data(), value_cache_buffer_.raw_data(), 1,
                total_tokens, n_heads, head_dim, n_kv_heads, kv_group_size, 0);
            Tensor q_heads = q_rot_gpu.reshape({n_heads, 1, head_dim});
            Tensor route_heads = q_heads;
            if (ssa_wsel_.data.size > 0) {
                if (ssa_wsel_.data.get_device() != Device::GPU) {
                    throw std::runtime_error(
                        "SSA selector parameter must share the GPU compute device");
                }
                route_heads = matmul_nt(q_heads, ssa_wsel_.data);
            }
            Tensor output_heads =
                Tensor::zeros({1, 1, n_heads, head_dim}, Device::GPU);
            const int block_size = std::max(ssa_block_size_, 1);
            const int block_count =
                gpu::ceil_div_positive(total_tokens, block_size);
            for (int head = 0; head < n_heads; ++head) {
                const size_t cache_offset =
                    static_cast<size_t>(head) * total_tokens * head_dim;
                const size_t head_offset =
                    static_cast<size_t>(head) * head_dim;
                Tensor block_means(
                    {block_count, head_dim}, Device::GPU);
                cuda::launch_sparse_block_means(
                    k_permuted.raw_data() + cache_offset,
                    block_means.raw_data(), total_tokens, head_dim, block_size);
                cuda::launch_sparse_selective_attention_decode(
                    q_heads.raw_data() + head_offset,
                    k_permuted.raw_data() + cache_offset,
                    v_permuted.raw_data() + cache_offset,
                    route_heads.raw_data() + head_offset,
                    block_means.raw_data(),
                    output_heads.raw_data() + head_offset, total_tokens,
                    head_dim, block_size, ssa_top_k_blocks_, ssa_local_blocks_,
                    ssa_sink_blocks_, scale);
            }
            ++cached_tokens_;
            attn_train_check_cuda("SSA streaming decode");
            Tensor projected = out_proj->forward(
                output_heads.reshape({1, d_model}));
            return input.shape.size() == 1
                       ? projected.reshape({d_model})
                       : projected;
        }
        if (streaming_inference_ && seq_len == 1 && !sparse_enabled_) {
            if (compact_kv_requested()) {
                ensure_compact_kv_capacity(cached_tokens_ + 1, 1);
                Tensor result = Tensor::uninitialized({1, d_model}, Device::GPU);
                launch_gqa_append_decode_batch(q_flat.raw_data(), kv_flat.raw_data(),
                    compact_kv_->keys(), compact_kv_->values(), result.raw_data(), 1,
                    cache_capacity_tokens_, cached_tokens_, n_heads, n_kv_heads, head_dim,
                    kv_group_size, theta, sliding_window_, 1);
                attn_train_check_cuda("compact KV decode");
                ++cached_tokens_;
                Tensor projected = out_proj->forward(result);
                return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
            }
            ensure_kv_cache_capacity(cached_tokens_ + 1, Device::GPU);
            Tensor output_gpu = Tensor::uninitialized(
                {seq_len, d_model}, Device::GPU);
            launch_gqa_append_kv_cache_kernel(
                kv_flat.raw_data(),
                key_cache_buffer_.raw_data(),
                value_cache_buffer_.raw_data(),
                cached_tokens_,
                n_kv_heads,
                head_dim,
                theta);
            launch_gqa_cached_attention_decode_kernel(
                q_flat.raw_data(),
                key_cache_buffer_.raw_data(),
                value_cache_buffer_.raw_data(),
                output_gpu.raw_data(),
                cached_tokens_ + 1,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_group_size,
                theta,
                sliding_window_);
            ++cached_tokens_;

            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("GQA cached attention kernel launch failed: ") +
                    cudaGetErrorString(status));
            }
            if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                if (std::string(sync_env) == "1") {
                    status = cudaDeviceSynchronize();
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("GQA cached attention kernel sync failed: ") +
                            cudaGetErrorString(status));
                    }
                    record_gpu_device_synchronization();
                }
            }

            Tensor projected = out_proj->forward(output_gpu);
            return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
        }

        if (!streaming_inference_ ||
            (streaming_inference_ && seq_len > 1 && !sparse_enabled_)) {
            Tensor output_gpu = Tensor::uninitialized(
                {seq_len, d_model}, Device::GPU);

            // A multi-token streaming call is a prefill: like the validated
            // CPU path below, it starts a new session and leaves every
            // RoPE-rotated K/V row resident on the compute device for the
            // subsequent one-token decode calls.  Do not call
            // clear_kv_cache() here because reserve_kv_cache() may already
            // have provisioned the correctly sized device allocation.
            if (streaming_inference_) {
                cached_tokens_ = 0;
                ensure_kv_cache_capacity(seq_len, Device::GPU);
                for (int token_index = 0; token_index < seq_len; ++token_index) {
                    launch_gqa_append_kv_cache_kernel(
                        kv_flat.raw_data() +
                            static_cast<size_t>(token_index) *
                                static_cast<size_t>(2 * kv_dim),
                        key_cache_buffer_.raw_data(),
                        value_cache_buffer_.raw_data(),
                        token_index,
                        n_kv_heads,
                        head_dim,
                        theta);
                }
            }

            launch_gqa_causal_attention_kernel(
                q_flat.raw_data(),
                kv_flat.raw_data(),
                output_gpu.raw_data(),
                seq_len,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_group_size,
                theta,
                sliding_window_);

            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("GQA causal attention kernel launch failed: ") +
                    cudaGetErrorString(status));
            }
            if (const char* sync_env = std::getenv("NSOS_CUDA_SYNC")) {
                if (std::string(sync_env) == "1") {
                    status = cudaDeviceSynchronize();
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("GQA causal attention kernel sync failed: ") +
                            cudaGetErrorString(status));
                    }
                    record_gpu_device_synchronization();
                }
            }
            if (streaming_inference_) {
                cached_tokens_ = seq_len;
            }

            Tensor projected = out_proj->forward(output_gpu);
            return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
        }
    }
#endif
    if (original_device == Device::GPU && streaming_inference_ &&
        sparse_enabled_ && strict_gpu_execution()) {
        throw std::runtime_error(
            "GPU streaming SSA has no validated device kernel; strict GPU "
            "execution refuses a host fallback");
    }
    Tensor q_cpu = (q_flat.get_device() == Device::GPU) ? q_flat.cpu() : q_flat;
    Tensor kv_cpu = (kv_flat.get_device() == Device::GPU) ? kv_flat.cpu() : kv_flat;

    // 2. Separar K e V do kv_flat
    //    kv_flat layout: [seq, d_model | d_model]
    Tensor k_flat({seq_len, kv_dim}, Device::CPU);
    Tensor v_flat({seq_len, kv_dim}, Device::CPU);
    {
        const float* kv_ptr = kv_cpu.data();
        float* k_ptr = k_flat.data();
        float* v_ptr = v_flat.data();
        for (int s = 0; s < seq_len; ++s) {
            // k é a primeira metade, v é a segunda
            std::memcpy(k_ptr + s * kv_dim, kv_ptr + s * 2 * kv_dim,
                        static_cast<size_t>(kv_dim) * sizeof(float));
            std::memcpy(v_ptr + s * kv_dim, kv_ptr + s * 2 * kv_dim + kv_dim,
                        static_cast<size_t>(kv_dim) * sizeof(float));
        }
    }

    // 3. Reshape para [seq, n_heads, head_dim]
    Tensor q3 = q_cpu.reshape({seq_len, n_heads, head_dim});
    Tensor k3 = k_flat.reshape({seq_len, n_kv_heads, head_dim});
    Tensor v3 = v_flat.reshape({seq_len, n_kv_heads, head_dim});

    // 4. Aplicar RoPE em Q e K
    auto [q_rot, k_rot] = apply_rope(q3, k3, streaming_inference_ ? cached_tokens_ : 0);

    // 5. Scored dot-product attention com máscara causal
    //    Resultado: [seq, d_model]
    Tensor output_cpu({seq_len, d_model}, Device::CPU);
    float* out_ptr = output_cpu.data();

    const float* q_rot_ptr = q_rot.data();
    const float* k_rot_ptr = k_rot.data();
    const float* v_ptr = v3.data();

    if (streaming_inference_ && seq_len == 1) {
        append_kv_cache_token(k_rot_ptr, Device::CPU, v_ptr, Device::CPU);

        std::vector<float> token_scores(static_cast<size_t>(cached_tokens_), 0.0f);
        const int first_token = std::max(0, cached_tokens_ - sliding_window_);
        for (int h = 0; h < n_heads; ++h) {
            const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
            // SSA (opt-in, default OFF): select top-k content blocks + local
            // window + sinks over the paged KV cache and mask out the rest so
            // the softmax below attends only to the selected tokens.  Block
            // means are recomputed per step (correct; an incremental-mean
            // cache is the documented speedup follow-up).
            std::vector<char> ssa_active;
            if (sparse_enabled_ && cached_tokens_ > 0) {
                const int bs = std::max(ssa_block_size_, 1);
                const int nb = gpu::ceil_div_positive(cached_tokens_, bs);
                const int cur = nb - 1;
                const size_t kvo =
                    static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                std::vector<float> bscore(static_cast<size_t>(nb), 0.0f);
                std::vector<char> bsel(static_cast<size_t>(nb), 0);
                for (int b = 0; b < nb; ++b) {
                    const int bstart = b * bs;
                    const int bend = std::min((b + 1) * bs, cached_tokens_);
                    const float inv_cnt =
                        1.0f / static_cast<float>(std::max(bend - bstart, 1));
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float mean_k = 0.0f;
                        for (int t = bstart; t < bend; ++t)
                            mean_k += kv_cache_token_ptr(key_cache_buffer_, t)[kvo + d];
                        acc += q_rot_ptr[h * head_dim + d] * (mean_k * inv_cnt);
                    }
                    bscore[static_cast<size_t>(b)] = acc * scale;
                }
                for (int b = 0; b < std::min(std::max(ssa_sink_blocks_, 0), nb); ++b)
                    bsel[static_cast<size_t>(b)] = 1;
                for (int b = std::max(0, cur - std::max(ssa_local_blocks_, 0) + 1);
                     b <= cur; ++b)
                    bsel[static_cast<size_t>(b)] = 1;
                if (ssa_top_k_blocks_ > 0) {
                    std::vector<std::pair<float, int>> cand;
                    for (int b = 0; b < nb; ++b)
                        if (!bsel[static_cast<size_t>(b)])
                            cand.emplace_back(bscore[static_cast<size_t>(b)], b);
                    const int kk =
                        std::min(ssa_top_k_blocks_, static_cast<int>(cand.size()));
                    std::partial_sort(
                        cand.begin(), cand.begin() + kk, cand.end(),
                        [](const std::pair<float, int>& a,
                           const std::pair<float, int>& b) { return a.first > b.first; });
                    for (int z = 0; z < kk; ++z)
                        bsel[static_cast<size_t>(cand[static_cast<size_t>(z)].second)] = 1;
                }
                ssa_active.assign(static_cast<size_t>(cached_tokens_), 0);
                for (int b = 0; b < nb; ++b)
                    if (bsel[static_cast<size_t>(b)])
                        for (int t = b * bs;
                             t < std::min((b + 1) * bs, cached_tokens_); ++t)
                            ssa_active[static_cast<size_t>(t)] = 1;
            }
            float max_s = -1e30f;
            for (int t = first_token; t < cached_tokens_; ++t) {
                if (!ssa_active.empty() && !ssa_active[static_cast<size_t>(t)]) {
                    token_scores[static_cast<size_t>(t)] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                const float* cached_key_token = kv_cache_token_ptr(key_cache_buffer_, t);
                const size_t kv_offset =
                    static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_rot_ptr[h * head_dim + d] * cached_key_token[kv_offset + d];
                }
                token_scores[static_cast<size_t>(t)] = dot * scale;
                max_s = std::max(max_s, token_scores[static_cast<size_t>(t)]);
            }
            float sum_exp = 0.0f;
            for (int t = first_token; t < cached_tokens_; ++t) {
                float& score_value = token_scores[static_cast<size_t>(t)];
                score_value = std::exp(score_value - max_s);
                sum_exp += score_value;
            }
            const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
            for (int t = first_token; t < cached_tokens_; ++t) {
                token_scores[static_cast<size_t>(t)] *= inv_sum;
            }
            float* out_head = out_ptr + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            std::fill_n(out_head, head_dim, 0.0f);
            const size_t kv_offset =
                static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
            for (int t = first_token; t < cached_tokens_; ++t) {
                const float* cached_value_token =
                    kv_cache_token_ptr(value_cache_buffer_, t) + kv_offset;
                const float score = token_scores[static_cast<size_t>(t)];
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += score * cached_value_token[static_cast<size_t>(d)];
                }
            }
        }
    } else {
        if (streaming_inference_) {
            clear_kv_cache();
            for (int token_index = 0; token_index < seq_len; ++token_index) {
                append_kv_cache_token(k_rot_ptr + token_index * kv_dim,
                                      Device::CPU,
                                      v_ptr + token_index * kv_dim,
                                      Device::CPU);
            }
        }
        std::vector<float> scores(seq_len * seq_len, 0.0f);
        for (int h = 0; h < n_heads; ++h) {
        if (sparse_enabled_) {
          // SSA: content-dependent block selection for this head (subq.ai).
          // Reuses the validated sparse_selective_attention component.  With
          // top_k covering all causal blocks this is identical to the dense
          // path below; with a smaller top_k it skips unselected blocks.
          SparseAttentionConfig ssa_cfg;
          ssa_cfg.block_size = ssa_block_size_;
          ssa_cfg.top_k_blocks = ssa_top_k_blocks_;
          ssa_cfg.local_blocks = ssa_local_blocks_;
          ssa_cfg.sink_blocks = ssa_sink_blocks_;
          ssa_cfg.scale = scale;
          const int kv_head_s = std::min(h / kv_group_size, n_kv_heads - 1);
          Tensor Qh({seq_len, head_dim}, Device::CPU);
          Tensor Kh({seq_len, head_dim}, Device::CPU);
          Tensor Vh({seq_len, head_dim}, Device::CPU);
          float* qp = Qh.data();
          float* kp = Kh.data();
          float* vp = Vh.data();
          for (int t = 0; t < seq_len; ++t) {
            for (int dd = 0; dd < head_dim; ++dd) {
              qp[t * head_dim + dd] =
                  q_rot_ptr[t * n_heads * head_dim + h * head_dim + dd];
              kp[t * head_dim + dd] =
                  k_rot_ptr[t * n_kv_heads * head_dim + kv_head_s * head_dim + dd];
              vp[t * head_dim + dd] =
                  v_ptr[t * n_kv_heads * head_dim + kv_head_s * head_dim + dd];
            }
          }
          Tensor Oh = sparse_selective_attention(Qh, Kh, Vh, ssa_cfg, nullptr,
                                                 &ssa_wsel_.data);
          const float* op = Oh.data();
          for (int t = 0; t < seq_len; ++t) {
            for (int dd = 0; dd < head_dim; ++dd) {
              out_ptr[t * n_heads * head_dim + h * head_dim + dd] =
                  op[t * head_dim + dd];
            }
          }
          continue;
        }
        // Extrair cabeça h de q_rot, k_rot, v3: cada um [seq, head_dim]
        // Calcular scores: [seq, seq] = Q_h @ K_h^T * scale
        std::fill(scores.begin(), scores.end(), 0.0f);
        const float* qh = q_rot.data() + h * head_dim;   // offset na dim de cabeça

        for (int i = 0; i < seq_len; ++i) {
            const int first_key = std::max(0, i - sliding_window_ + 1);
            for (int j = 0; j < seq_len; ++j) {
                if (j < first_key || j > i) {
                    // Máscara causal: tokens futuros recebem -inf
                    scores[i * seq_len + j] = -1e9f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    // Stride: [seq, n_heads, head_dim] → posição head h, pos seq i
                    const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
                    dot += q_rot_ptr[i * n_heads * head_dim + h * head_dim + d]
                         * k_rot_ptr[j * n_kv_heads * head_dim + kv_head * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }

            // Softmax sobre eixo j (linha i)
            float max_s = *std::max_element(scores.data() + i * seq_len,
                                             scores.data() + i * seq_len + seq_len);
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_s);
                sum_exp += scores[i * seq_len + j];
            }
            float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
            for (int j = 0; j < seq_len; ++j)
                scores[i * seq_len + j] *= inv_sum;

            // Acumular: out[i, h] = sum_j(attn[i,j] * V[j, h])
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    const int kv_head = std::min(h / kv_group_size, n_kv_heads - 1);
                    acc += scores[i * seq_len + j]
                         * v_ptr[j * n_kv_heads * head_dim + kv_head * head_dim + d];
                }
                // Layout de saída: [seq, n_heads, head_dim]
                out_ptr[i * n_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }

    // 6. Reshape [seq, n_heads, head_dim] → [seq, d_model] e projetar
    }
    Tensor output = (original_device == Device::GPU) ? output_cpu.to(Device::GPU) : output_cpu;
    Tensor output2d = output.reshape({seq_len, d_model});
    Tensor projected = out_proj->forward(output2d);
    return input.shape.size() == 1 ? projected.reshape({d_model}) : projected;
}

#ifdef USE_CUDA
// Device-resident exact-cache backward.  Mirrors the host loop in
// Attention::backward EXACTLY (same masks: j<=i, j<valid_b, i<valid_b; same
// softmax 1e-9 guard; scale folded into dS; RoPE-transpose at start_pos=0;
// rows beyond valid_b produce zero grads), but as 5 batched-cuBLAS GEMM
// families + glue kernels from attention_train_kernels.cu.  The GEMMs run
// through Tensor::matmul, so the BF16 Tensor-Core mode applies here too.
// Derivation per (b,h,i):  P = softmax(mask(scale·Q·Kᵀ))
//   dP = dO·Vᵀ ; dS = scale·P⊙(dP − Σⱼ P·dP) ; dQ = dS·K ; dK = dSᵀ·Q ;
//   dV = Pᵀ·dO ; GQA: dK/dV reduzidos por grupo de query-heads.
Tensor Attention::backward_sparse_gpu(const Tensor& dy, int batch_size,
                                      int seq_len, int kv_dim, float scale) {
#ifdef USE_CUDA
    Tensor grad_out = out_proj->backward(dy);
    Tensor grad_heads =
        grad_out.reshape({batch_size, seq_len, n_heads, head_dim});
    const int BH = batch_size * n_heads;
    Tensor q_permuted = Tensor::uninitialized(
        {BH, seq_len, head_dim}, Device::GPU);
    Tensor k_permuted = Tensor::uninitialized(
        {BH, seq_len, head_dim}, Device::GPU);
    Tensor v_permuted = Tensor::uninitialized(
        {BH, seq_len, head_dim}, Device::GPU);
    Tensor go_permuted = Tensor::uninitialized(
        {BH, seq_len, head_dim}, Device::GPU);
    launch_attn_gather_heads(
        q_permuted.raw_data(), saved_q_rot_.raw_data(), batch_size, seq_len,
        n_heads, head_dim, n_heads, 1, 0);
    launch_attn_gather_heads(
        k_permuted.raw_data(), saved_k_rot_.raw_data(), batch_size, seq_len,
        n_heads, head_dim, n_kv_heads, kv_group_size, 0);
    launch_attn_gather_heads(
        v_permuted.raw_data(), saved_v_heads_.raw_data(), batch_size, seq_len,
        n_heads, head_dim, n_kv_heads, kv_group_size, 0);
    launch_attn_gather_heads(
        go_permuted.raw_data(), grad_heads.raw_data(), batch_size, seq_len,
        n_heads, head_dim, n_heads, 1, 0);

    Tensor route_permuted = q_permuted;
    if (ssa_wsel_.data.size > 0) {
        if (ssa_wsel_.data.get_device() != Device::GPU) {
            throw std::runtime_error(
                "SSA selector parameter must share the GPU compute device");
        }
        route_permuted = matmul_nt(q_permuted, ssa_wsel_.data);
    }
    Tensor dq_permuted =
        Tensor::zeros({BH, seq_len, head_dim}, Device::GPU);
    Tensor dk_permuted =
        Tensor::zeros({BH, seq_len, head_dim}, Device::GPU);
    Tensor dv_permuted =
        Tensor::zeros({BH, seq_len, head_dim}, Device::GPU);
    const int block_size = std::max(ssa_block_size_, 1);
    const bool deterministic =
        determinism::deterministic_reductions_enabled();
    for (int batch = 0; batch < batch_size; ++batch) {
        const int valid_len = std::clamp(
            saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
        if (valid_len <= 0) continue;
        const int block_count =
            gpu::ceil_div_positive(valid_len, block_size);
        for (int head = 0; head < n_heads; ++head) {
            const int bh = batch * n_heads + head;
            const size_t offset =
                static_cast<size_t>(bh) * seq_len * head_dim;
            Tensor block_means(
                {block_count, head_dim}, Device::GPU);
            cuda::launch_sparse_block_means(
                k_permuted.raw_data() + offset, block_means.raw_data(),
                valid_len, head_dim, block_size);
            cuda::launch_sparse_selective_attention_backward(
                q_permuted.raw_data() + offset,
                k_permuted.raw_data() + offset,
                v_permuted.raw_data() + offset,
                route_permuted.raw_data() + offset,
                block_means.raw_data(), go_permuted.raw_data() + offset,
                dq_permuted.raw_data() + offset,
                dk_permuted.raw_data() + offset,
                dv_permuted.raw_data() + offset, valid_len, head_dim,
                block_size, ssa_top_k_blocks_, ssa_local_blocks_,
                ssa_sink_blocks_, scale, deterministic);
        }
    }

    Tensor grad_q_rot = Tensor::uninitialized(
        {batch_size, seq_len, n_heads, head_dim}, Device::GPU);
    Tensor grad_k_rot = Tensor::uninitialized(
        {batch_size, seq_len, n_kv_heads, head_dim}, Device::GPU);
    Tensor grad_v = Tensor::uninitialized(
        {batch_size, seq_len, n_kv_heads, head_dim}, Device::GPU);
    launch_attn_unpermute_heads(
        grad_q_rot.raw_data(), dq_permuted.raw_data(), batch_size, n_heads,
        seq_len, head_dim);
    launch_attn_reduce_group(
        grad_k_rot.raw_data(), dk_permuted.raw_data(), batch_size, seq_len,
        n_kv_heads, head_dim, n_heads, kv_group_size);
    launch_attn_reduce_group(
        grad_v.raw_data(), dv_permuted.raw_data(), batch_size, seq_len,
        n_kv_heads, head_dim, n_heads, kv_group_size);

    ensure_rope_gpu_cache();
    launch_rope_apply(
        grad_q_rot.raw_data(), rope_cos_gpu_.raw_data(),
        rope_sin_gpu_.raw_data(), batch_size, seq_len, n_heads, head_dim, 0,
        max_seq_len, -1);
    launch_rope_apply(
        grad_k_rot.raw_data(), rope_cos_gpu_.raw_data(),
        rope_sin_gpu_.raw_data(), batch_size, seq_len, n_kv_heads, head_dim, 0,
        max_seq_len, -1);
    Tensor grad_q_input =
        grad_q_rot.reshape({batch_size, seq_len, d_model});
    Tensor grad_kv_input = Tensor::uninitialized(
        {batch_size, seq_len, 2 * kv_dim}, Device::GPU);
    launch_kv_concat(
        grad_kv_input.raw_data(), grad_k_rot.raw_data(), grad_v.raw_data(),
        static_cast<long long>(batch_size) * seq_len, kv_dim);
    attn_train_check_cuda("SSA selective backward");

    Tensor grad_q = q_down_proj->backward(grad_q_input);
    Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
    Tensor grad_total = grad_q.add(grad_kv);
    if (saved_input_rank_ == 1) return grad_total.reshape({d_model});
    if (saved_input_rank_ == 2) return grad_total.reshape({seq_len, d_model});
    return grad_total;
#else
    (void)dy;
    (void)batch_size;
    (void)seq_len;
    (void)kv_dim;
    (void)scale;
    throw std::runtime_error("GPU SSA backward requires a GPU build");
#endif
}

Tensor Attention::backward_exact_gpu(const Tensor& dy, int batch_size,
                                     int seq_len, int kv_dim, float scale) {
    const int B = batch_size;
    const int S = seq_len;
    const int H = n_heads;
    const int KV = n_kv_heads;
    const int hd = head_dim;
    const int group = kv_group_size;
    const int BH = B * H;

    Tensor grad_out = out_proj->backward(dy);  // [B,S,d_model] on GPU
    Tensor grad_heads = grad_out.reshape({B, S, H, hd});

    // Per-batch valid lengths (clamped exactly like the host) -> device.
    std::vector<int> valid_host(static_cast<size_t>(B), S);
    for (int b = 0; b < B; ++b) {
        valid_host[static_cast<size_t>(b)] =
            std::clamp(saved_valid_lengths_[static_cast<size_t>(b)], 0, S);
    }
    int* dvalid = attn_valid_device_values(valid_host);
    if (dvalid == nullptr) {
        throw std::runtime_error(
            "attention backward: valid-length buffer allocation failed");
    }

    // Head gather/expand: Q,dO permute [B,S,H,hd]->[B,H,S,hd]; K,V expanded to
    if (saved_attn_lse_.size > 0) {
        Tensor delta = Tensor::uninitialized({B, S, H}, Device::GPU);
        Tensor dq = Tensor::uninitialized({B, S, H, hd}, Device::GPU);
        Tensor dk = Tensor::uninitialized({B, S, KV, hd}, Device::GPU);
        Tensor dv = Tensor::uninitialized({B, S, KV, hd}, Device::GPU);
        if (!launch_attn_tiled_backward(saved_q_rot_.raw_data(), saved_k_rot_.raw_data(),
            saved_v_heads_.raw_data(), saved_attn_output_.raw_data(), grad_heads.raw_data(),
            saved_attn_lse_.raw_data(), dvalid, delta.raw_data(), dq.raw_data(), dk.raw_data(),
            dv.raw_data(), B, S, H, KV, hd, group, sliding_window_, scale))
            throw std::runtime_error("Tiled attention training backward rejected shape or launch");
        ensure_rope_gpu_cache();
        launch_rope_apply(dq.raw_data(), rope_cos_gpu_.raw_data(), rope_sin_gpu_.raw_data(),
            B, S, H, hd, 0, max_seq_len, -1);
        launch_rope_apply(dk.raw_data(), rope_cos_gpu_.raw_data(), rope_sin_gpu_.raw_data(),
            B, S, KV, hd, 0, max_seq_len, -1);
        Tensor dkv = Tensor::uninitialized({B, S, 2 * kv_dim}, Device::GPU);
        launch_kv_concat(dkv.raw_data(), dk.raw_data(), dv.raw_data(), static_cast<long long>(B)*S, kv_dim);
        attn_train_check_cuda("tiled attention backward/RoPE");
        Tensor result = q_down_proj->backward(dq.reshape({B, S, d_model})).add(kv_down_proj->backward(dkv));
        if (saved_input_rank_ == 1) return result.reshape({d_model});
        if (saved_input_rank_ == 2) return result.reshape({S, d_model});
        return result;
    }
    // Head gather/expand: Q,dO permute [B,S,H,hd]->[B,H,S,hd]; K,V expanded to
    // per-query-head (GQA kv_head = min(h/group, KV-1)), K/V also produced
    // pre-transposed [B,H,hd,S] for the scores/dP GEMMs.
    Tensor Qp = Tensor::uninitialized({B, H, S, hd}, Device::GPU);
    Tensor Op = Tensor::uninitialized({B, H, S, hd}, Device::GPU);
    Tensor Kp = Tensor::uninitialized({B, H, S, hd}, Device::GPU);
    Tensor KpT = Tensor::uninitialized({B, H, hd, S}, Device::GPU);
    Tensor VpT = Tensor::uninitialized({B, H, hd, S}, Device::GPU);
    launch_attn_gather_heads(Qp.raw_data(), saved_q_rot_.raw_data(), B, S, H, hd, H, 1, 0);
    launch_attn_gather_heads(Op.raw_data(), grad_heads.raw_data(), B, S, H, hd, H, 1, 0);
    launch_attn_gather_heads(Kp.raw_data(), saved_k_rot_.raw_data(), B, S, H, hd, KV, group, 0);
    launch_attn_gather_heads(KpT.raw_data(), saved_k_rot_.raw_data(), B, S, H, hd, KV, group, 1);
    launch_attn_gather_heads(VpT.raw_data(), saved_v_heads_.raw_data(), B, S, H, hd, KV, group, 1);
    attn_train_check_cuda("launch_attn_gather_heads");

    // P = masked_softmax(scale * Q Kᵀ)   (in place over the scores buffer)
    Tensor scores = Qp.reshape({BH, S, hd}).matmul(KpT.reshape({BH, hd, S}));
    KpT = Tensor();
    launch_attn_masked_softmax(scores.raw_data(), dvalid, B, H, S, scale,
                               sliding_window_);
    // dP = dO Vᵀ ; dS = scale * P ⊙ (dP − rowdot)
    Tensor dP = Op.reshape({BH, S, hd}).matmul(VpT.reshape({BH, hd, S}));
    VpT = Tensor();
    Tensor dS = Tensor::uninitialized({BH, S, S}, Device::GPU);
    launch_attn_softmax_backward(dS.raw_data(), scores.raw_data(), dP.raw_data(),
                                 B, H, S, scale);
    dP = Tensor();
    attn_train_check_cuda("attn softmax fwd/bwd");

    // dQ = dS K ; dK(per-q-head) = dSᵀ Q ; dV(per-q-head) = Pᵀ dO
    Tensor dQp = dS.matmul(Kp.reshape({BH, S, hd}));
    Kp = Tensor();
    Tensor dS_T = Tensor::uninitialized({BH, S, S}, Device::GPU);
    launch_batched_transpose_last2(dS_T.raw_data(), dS.raw_data(), BH, S, S);
    dS = Tensor();
    Tensor P_T = Tensor::uninitialized({BH, S, S}, Device::GPU);
    launch_batched_transpose_last2(P_T.raw_data(), scores.raw_data(), BH, S, S);
    scores = Tensor();
    attn_train_check_cuda("attn transposes");
    Tensor dKp = dS_T.matmul(Qp.reshape({BH, S, hd}));
    dS_T = Tensor();
    Qp = Tensor();
    Tensor dVp = P_T.matmul(Op.reshape({BH, S, hd}));
    P_T = Tensor();
    Op = Tensor();

    // Back to model layouts (+ GQA group reduction for K/V).
    Tensor grad_q_rot = Tensor::uninitialized({B, S, H, hd}, Device::GPU);
    Tensor grad_k_rot = Tensor::uninitialized({B, S, KV, hd}, Device::GPU);
    Tensor grad_v = Tensor::uninitialized({B, S, KV, hd}, Device::GPU);
    launch_attn_unpermute_heads(grad_q_rot.raw_data(), dQp.raw_data(), B, H, S, hd);
    launch_attn_reduce_group(grad_k_rot.raw_data(), dKp.raw_data(), B, S, KV, hd, H, group);
    launch_attn_reduce_group(grad_v.raw_data(), dVp.raw_data(), B, S, KV, hd, H, group);
    attn_train_check_cuda("attn unpermute/reduce");

    // RoPE transpose-rotation (identical math to apply_rope_backward, pos
    // clamp included); V carries no rotation, matching the host.
    ensure_rope_gpu_cache();
    launch_rope_apply(grad_q_rot.raw_data(), rope_cos_gpu_.raw_data(),
                      rope_sin_gpu_.raw_data(), B, S, H, hd, 0, max_seq_len, -1);
    launch_rope_apply(grad_k_rot.raw_data(), rope_cos_gpu_.raw_data(),
                      rope_sin_gpu_.raw_data(), B, S, KV, hd, 0, max_seq_len, -1);
    attn_train_check_cuda("launch_rope_apply(bwd)");

    // Assemble projection grads: [B,S,H,hd] is layout-identical to
    // [B,S,d_model]; KV halves interleave into [B,S,2*kv_dim].
    Tensor grad_q_input = grad_q_rot.reshape({B, S, d_model});
    Tensor grad_kv_input = Tensor::uninitialized({B, S, 2 * kv_dim}, Device::GPU);
    launch_kv_concat(grad_kv_input.raw_data(), grad_k_rot.raw_data(),
                     grad_v.raw_data(), static_cast<long long>(B) * S, kv_dim);
    attn_train_check_cuda("launch_kv_concat");

    Tensor grad_q = q_down_proj->backward(grad_q_input);
    Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
    Tensor grad_total = grad_q.add(grad_kv);
    if (saved_input_rank_ == 1) {
        return grad_total.reshape({d_model});
    }
    if (saved_input_rank_ == 2) {
        return grad_total.reshape({S, d_model});
    }
    return grad_total;
}
#endif  // USE_CUDA

Tensor Attention::backward(const Tensor& dy, Context* ctx) {
    (void)ctx;
    if (saved_rdna_training_) {
        if (!rdna_training_) throw std::logic_error("RDNA attention missing owning provider");
        return rdna_training_->backward(dy);
    }
    const bool has_exact_cache = saved_q_rot_.size > 0 &&
                                 saved_k_rot_.size > 0 &&
                                 saved_v_heads_.size > 0 &&
                                 !saved_valid_lengths_.empty();
    if (has_exact_cache) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int batch_size = saved_q_rot_.shape[0];
        const int seq_len = saved_q_rot_.shape[1];
        const int kv_dim = n_kv_heads * head_dim;

#ifdef USE_CUDA
        // GPU exact-cache backward: keeps the whole attention backward
        // device-resident.  The host loop below remains as the reference
        // implementation and the NSOS_ATTN_BWD_HOST=1 parity arm.
        if (sparse_enabled_ && !attn_bwd_force_host() &&
            gpu_custom_kernels_supported() &&
            saved_q_rot_.get_device() == Device::GPU &&
            saved_k_rot_.get_device() == Device::GPU &&
            saved_v_heads_.get_device() == Device::GPU &&
            head_dim <= 256 && ssa_top_k_blocks_ >= 0 &&
            ssa_top_k_blocks_ <= 64 && head_dim % 2 == 0) {
            return backward_sparse_gpu(
                dy, batch_size, seq_len, kv_dim, scale);
        }
        if (!sparse_enabled_ && !attn_bwd_force_host() &&
            gpu_custom_kernels_supported() &&
            saved_q_rot_.get_device() == Device::GPU &&
            saved_k_rot_.get_device() == Device::GPU &&
            saved_v_heads_.get_device() == Device::GPU &&
            head_dim % 2 == 0) {
            return backward_exact_gpu(dy, batch_size, seq_len, kv_dim, scale);
        }
#endif

        if (saved_q_rot_.get_device() == Device::GPU &&
            strict_gpu_execution()) {
            throw std::runtime_error(
                "Strict GPU attention backward has no eligible device path for "
                "the active configuration");
        }

        Tensor grad_out = out_proj->backward(dy);
        Tensor grad_out_host = (grad_out.get_device() == Device::GPU) ? grad_out.cpu() : grad_out;
        Tensor grad_heads = grad_out_host.reshape({batch_size, seq_len, n_heads, head_dim});

        Tensor q_host =
            (saved_q_rot_.get_device() == Device::GPU) ? saved_q_rot_.cpu() : saved_q_rot_;
        Tensor k_host =
            (saved_k_rot_.get_device() == Device::GPU) ? saved_k_rot_.cpu() : saved_k_rot_;
        Tensor v_host =
            (saved_v_heads_.get_device() == Device::GPU) ? saved_v_heads_.cpu() : saved_v_heads_;
        Tensor grad_q_rot({batch_size, seq_len, n_heads, head_dim}, Device::CPU);
        Tensor grad_k_rot({batch_size, seq_len, n_kv_heads, head_dim}, Device::CPU);
        Tensor grad_v({batch_size, seq_len, n_kv_heads, head_dim}, Device::CPU);

        float* grad_q_ptr = grad_q_rot.data();
        float* grad_k_ptr = grad_k_rot.data();
        float* grad_v_ptr = grad_v.data();
        const float* q_ptr = q_host.data();
        const float* k_ptr = k_host.data();
        const float* v_ptr = v_host.data();
        const float* grad_head_ptr = grad_heads.data();
        std::vector<float> d_probs(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
        std::vector<float> probs(static_cast<size_t>(seq_len), 0.0f);

        if (sparse_enabled_) {
            SparseAttentionConfig sparse_cfg;
            sparse_cfg.block_size = ssa_block_size_;
            sparse_cfg.top_k_blocks = ssa_top_k_blocks_;
            sparse_cfg.local_blocks = ssa_local_blocks_;
            sparse_cfg.sink_blocks = ssa_sink_blocks_;
            sparse_cfg.scale = scale;

            for (int batch = 0; batch < batch_size; ++batch) {
                const int valid_len = std::clamp(
                    saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
                if (valid_len <= 0) continue;
                for (int head = 0; head < n_heads; ++head) {
                    const int kv_head =
                        std::min(head / kv_group_size, n_kv_heads - 1);
                    Tensor Qh({valid_len, head_dim}, Device::CPU);
                    Tensor Kh({valid_len, head_dim}, Device::CPU);
                    Tensor Vh({valid_len, head_dim}, Device::CPU);
                    Tensor dOh({valid_len, head_dim}, Device::CPU);
                    for (int token = 0; token < valid_len; ++token) {
                        const size_t q_offset =
                            (((static_cast<size_t>(batch) * seq_len + token) *
                              n_heads) + head) * head_dim;
                        const size_t kv_offset =
                            (((static_cast<size_t>(batch) * seq_len + token) *
                              n_kv_heads) + kv_head) * head_dim;
                        std::memcpy(Qh.data() + static_cast<size_t>(token) * head_dim,
                                    q_ptr + q_offset,
                                    static_cast<size_t>(head_dim) * sizeof(float));
                        std::memcpy(Kh.data() + static_cast<size_t>(token) * head_dim,
                                    k_ptr + kv_offset,
                                    static_cast<size_t>(head_dim) * sizeof(float));
                        std::memcpy(Vh.data() + static_cast<size_t>(token) * head_dim,
                                    v_ptr + kv_offset,
                                    static_cast<size_t>(head_dim) * sizeof(float));
                        std::memcpy(dOh.data() + static_cast<size_t>(token) * head_dim,
                                    grad_head_ptr + q_offset,
                                    static_cast<size_t>(head_dim) * sizeof(float));
                    }

                    SparseAttentionBackwardResult sparse_grad =
                        sparse_selective_attention_backward(
                            Qh, Kh, Vh, sparse_cfg, dOh, &ssa_wsel_.data);
                    const float* dq_sparse = sparse_grad.dQ.data();
                    const float* dk_sparse = sparse_grad.dK.data();
                    const float* dv_sparse = sparse_grad.dV.data();
                    for (int token = 0; token < valid_len; ++token) {
                        const size_t q_offset =
                            (((static_cast<size_t>(batch) * seq_len + token) *
                              n_heads) + head) * head_dim;
                        const size_t kv_offset =
                            (((static_cast<size_t>(batch) * seq_len + token) *
                              n_kv_heads) + kv_head) * head_dim;
                        for (int dim = 0; dim < head_dim; ++dim) {
                            const size_t local =
                                static_cast<size_t>(token) * head_dim + dim;
                            grad_q_ptr[q_offset + dim] = dq_sparse[local];
                            grad_k_ptr[kv_offset + dim] += dk_sparse[local];
                            grad_v_ptr[kv_offset + dim] += dv_sparse[local];
                        }
                    }
                }
            }
        } else {
        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len =
                std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            for (int head = 0; head < n_heads; ++head) {
                const int kv_head = std::min(head / kv_group_size, n_kv_heads - 1);
                for (int i = 0; i < valid_len; ++i) {
                    const int first_key = std::max(0, i - sliding_window_ + 1);
                    const size_t q_offset =
                        (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    const size_t grad_out_offset =
                        (((static_cast<size_t>(batch) * seq_len + i) * n_heads) + head) *
                        static_cast<size_t>(head_dim);

                    float max_s = -1e30f;
                    for (int j = 0; j < seq_len; ++j) {
                        float score = -1e9f;
                        if (j >= first_key && j < valid_len && j <= i) {
                            float dot = 0.0f;
                            const size_t k_offset =
                                (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) +
                                 kv_head) *
                                static_cast<size_t>(head_dim);
                            for (int dim = 0; dim < head_dim; ++dim) {
                                dot += q_ptr[q_offset + static_cast<size_t>(dim)] *
                                       k_ptr[k_offset + static_cast<size_t>(dim)];
                            }
                            score = dot * scale;
                        }
                        scores[static_cast<size_t>(j)] = score;
                        max_s = std::max(max_s, score);
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        const float value =
                            (j >= first_key && j < valid_len && j <= i)
                                ? std::exp(scores[static_cast<size_t>(j)] - max_s)
                                : 0.0f;
                        probs[static_cast<size_t>(j)] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
                    for (int j = 0; j < seq_len; ++j) {
                        probs[static_cast<size_t>(j)] *= inv_sum;
                    }

                    float row_dot = 0.0f;
                    for (int j = first_key; j < valid_len; ++j) {
                        const size_t v_offset =
                            (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) + kv_head) *
                            static_cast<size_t>(head_dim);
                        float dot = 0.0f;
                        for (int dim = 0; dim < head_dim; ++dim) {
                            dot += grad_head_ptr[grad_out_offset + static_cast<size_t>(dim)] *
                                   v_ptr[v_offset + static_cast<size_t>(dim)];
                        }
                        d_probs[static_cast<size_t>(j)] = dot;
                        row_dot += dot * probs[static_cast<size_t>(j)];
                    }

                    for (int j = first_key; j < valid_len; ++j) {
                        const float prob = probs[static_cast<size_t>(j)];
                        const float d_score = prob * (d_probs[static_cast<size_t>(j)] - row_dot);
                        const size_t k_offset =
                            (((static_cast<size_t>(batch) * seq_len + j) * n_kv_heads) + kv_head) *
                            static_cast<size_t>(head_dim);
                        for (int dim = 0; dim < head_dim; ++dim) {
                            const size_t dim_offset = static_cast<size_t>(dim);
                            grad_q_ptr[q_offset + dim_offset] +=
                                scale * d_score * k_ptr[k_offset + dim_offset];
                            grad_k_ptr[k_offset + dim_offset] +=
                                scale * d_score * q_ptr[q_offset + dim_offset];
                            grad_v_ptr[k_offset + dim_offset] +=
                                prob * grad_head_ptr[grad_out_offset + dim_offset];
                        }
                    }
                }
            }
        }
        }

        auto [grad_q_pre_rope, grad_k_pre_rope] = apply_rope_backward(grad_q_rot, grad_k_rot, 0);

        Tensor grad_q_input_cpu({batch_size, seq_len, d_model}, Device::CPU);
        Tensor grad_kv_input_cpu({batch_size, seq_len, 2 * kv_dim}, Device::CPU);
        float* grad_q_input_ptr = grad_q_input_cpu.data();
        float* grad_kv_input_ptr = grad_kv_input_cpu.data();
        const float* grad_q_pre_ptr = grad_q_pre_rope.data();
        const float* grad_k_pre_ptr = grad_k_pre_rope.data();
        const float* grad_v_pre_ptr = grad_v.data();

        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len =
                std::clamp(saved_valid_lengths_[static_cast<size_t>(batch)], 0, seq_len);
            for (int token = 0; token < valid_len; ++token) {
                const size_t q_token_offset =
                    (static_cast<size_t>(batch) * seq_len + token) * static_cast<size_t>(d_model);
                for (int head = 0; head < n_heads; ++head) {
                    const size_t head_offset =
                        (((static_cast<size_t>(batch) * seq_len + token) * n_heads) + head) *
                        static_cast<size_t>(head_dim);
                    const size_t out_offset = q_token_offset +
                                              static_cast<size_t>(head) *
                                                  static_cast<size_t>(head_dim);
                    std::memcpy(grad_q_input_ptr + out_offset,
                                grad_q_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                }

                const size_t kv_token_offset =
                    (static_cast<size_t>(batch) * seq_len + token) *
                    static_cast<size_t>(2 * kv_dim);
                for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                    const size_t head_offset =
                        (((static_cast<size_t>(batch) * seq_len + token) * n_kv_heads) + kv_head) *
                        static_cast<size_t>(head_dim);
                    const size_t k_out_offset = kv_token_offset +
                                                static_cast<size_t>(kv_head) *
                                                    static_cast<size_t>(head_dim);
                    const size_t v_out_offset = k_out_offset + static_cast<size_t>(kv_dim);
                    std::memcpy(grad_kv_input_ptr + k_out_offset,
                                grad_k_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                    std::memcpy(grad_kv_input_ptr + v_out_offset,
                                grad_v_pre_ptr + head_offset,
                                static_cast<size_t>(head_dim) * sizeof(float));
                }
            }
        }

        Tensor grad_q_input = grad_q_input_cpu;
        Tensor grad_kv_input = grad_kv_input_cpu;
        if (grad_out.get_device() == Device::GPU) {
            grad_q_input = grad_q_input.to(Device::GPU);
            grad_kv_input = grad_kv_input.to(Device::GPU);
        }

        Tensor grad_q = q_down_proj->backward(grad_q_input);
        Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
        Tensor grad_total = grad_q.add(grad_kv);
        if (saved_input_rank_ == 1) {
            return grad_total.reshape({d_model});
        }
        if (saved_input_rank_ == 2) {
            return grad_total.reshape({seq_len, d_model});
        }
        return grad_total;
    }
    {
    Tensor grad_out = out_proj->backward(dy);
    Tensor grad_q = q_down_proj->backward(grad_out);

    const int kv_dim = n_kv_heads * head_dim;
    const int rank = static_cast<int>(grad_out.shape.size());
    const int rows = grad_out.size / std::max(d_model, 1);
    Tensor grad_heads = grad_out.reshape({rows, n_heads, head_dim});
    Tensor grad_kv_proj = Tensor::zeros({rows, 2 * kv_dim}, grad_out.get_device());

    const float* grad_head_ptr = grad_heads.data();
    float* grad_kv_ptr = grad_kv_proj.data();
    for (int row = 0; row < rows; ++row) {
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            const int start_head = kv_head * kv_group_size;
            const int end_head = std::min(start_head + kv_group_size, n_heads);
            const int group_count = std::max(end_head - start_head, 1);
            for (int dim = 0; dim < head_dim; ++dim) {
                float accum = 0.0f;
                for (int head = start_head; head < end_head; ++head) {
                    const size_t offset =
                        ((static_cast<size_t>(row) * n_heads) + head) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    accum += grad_head_ptr[offset];
                }
                const float shared_grad = accum / static_cast<float>(group_count);
                const size_t k_offset =
                    static_cast<size_t>(row) * static_cast<size_t>(2 * kv_dim) +
                    static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim) +
                    static_cast<size_t>(dim);
                const size_t v_offset = k_offset + static_cast<size_t>(kv_dim);
                grad_kv_ptr[k_offset] = shared_grad;
                grad_kv_ptr[v_offset] = shared_grad;
            }
        }
    }

    Tensor grad_kv_input = grad_kv_proj;
    if (rank == 3) {
        grad_kv_input = grad_kv_proj.reshape({dy.shape[0], dy.shape[1], 2 * kv_dim});
    } else if (rank == 1) {
        grad_kv_input = grad_kv_proj.reshape({2 * kv_dim});
    }

    Tensor grad_kv = kv_down_proj->backward(grad_kv_input);
    return grad_q.add(grad_kv);
    }
    // Gradiente através da projeção de saída
    Tensor grad_out = out_proj->backward(dy);
    // Propagar gradiente através das projeções Q e KV
    // Para MHA completo, o gradiente se divide entre n_heads
    Tensor grad_q  = q_down_proj->backward(grad_out);
    Tensor grad_kv = kv_down_proj->backward(grad_out);
    // Somar contribuições (chain rule simplificada: grad passa pelas duas projeções)
    return grad_q.add(grad_kv.slice(0, 0, grad_q.shape[0]));
}

void Attention::to(Device dev) {
    if (rdna_training_) rdna_training_->cancel_pending();
    rdna_training_.reset();
    saved_rdna_training_ = false;
    if (compact_kv_ && dev != Device::GPU) materialize_compact_kv();
    if (key_cache_buffer_.size && key_cache_buffer_.get_device() != dev)
        key_cache_buffer_ = key_cache_buffer_.to(dev);
    if (value_cache_buffer_.size && value_cache_buffer_.get_device() != dev)
        value_cache_buffer_ = value_cache_buffer_.to(dev);
    if (q_down_proj) {
        q_down_proj->to(dev);
    }
    if (kv_down_proj) {
        kv_down_proj->to(dev);
    }
    if (out_proj) {
        out_proj->to(dev);
    }
    if (ssa_wsel_.data.get_device() != dev) {
        ssa_wsel_.data = ssa_wsel_.data.to(dev);
        ssa_wsel_.mark_updated();
    }
    if (ssa_wsel_.grad.size > 0 && ssa_wsel_.grad.get_device() != dev) {
        ssa_wsel_.grad = ssa_wsel_.grad.to(dev);
    }
}

std::vector<Parameter*> Attention::parameters() {
    std::vector<Parameter*> params;
    if (q_down_proj) {
        auto q = q_down_proj->parameters();
        prefix_parameter_names(q, "q_down_proj.");
        params.insert(params.end(), q.begin(), q.end());
    }
    if (kv_down_proj) {
        auto kv = kv_down_proj->parameters();
        prefix_parameter_names(kv, "kv_down_proj.");
        params.insert(params.end(), kv.begin(), kv.end());
    }
    if (out_proj) {
        auto out = out_proj->parameters();
        prefix_parameter_names(out, "out_proj.");
        params.insert(params.end(), out.begin(), out.end());
    }
    // Learned routing state is a first-class parameter: the Trainer owns its
    // update and serializers must preserve it with the rest of the attention.
    params.push_back(&ssa_wsel_);
    return params;
}

float Attention::accumulate_selector_distill_grad(float gradient_weight) {
    // Learned block selection trained by distilling the DENSE attention's
    // per-block MASS into the scorer: the selector learns to predict which
    // blocks dense attention attends to, so hard top-k by (Wsel @ q).block_mean
    // picks the right blocks.  Cross-entropy(target_mass, softmax(scores)) with
    // the exact gradient (pred - mass); self-contained SGD on ssa_wsel_ to keep
    // it decoupled from the main optimizer.  Returns the CE loss (0 if the
    // exact-training forward's per-head tensors are unavailable).
    if (!sparse_enabled_) {
        return 0.0f;  // selector only drives selection when sparse attention is opt-in
    }
    if (saved_q_rot_.size == 0 || saved_k_rot_.size == 0) {
        return 0.0f;
    }
    const auto& qs = saved_q_rot_.shape.dims;  // [B, S, nH, hd]
    const auto& ks = saved_k_rot_.shape.dims;  // [B, S, nKV, hd]
    if (qs.size() != 4 || ks.size() != 4) {
        return 0.0f;
    }
    const int batch_size = qs[0];
    const int S = qs[1];
    const int nH = qs[2];
    const int d = qs[3];
    const int nKV = ks[2];
    if (S <= 0 || d != head_dim) {
        return 0.0f;
    }
    const int Bsz = std::max(ssa_block_size_, 1);
    const int nb = gpu::ceil_div_positive(S, Bsz);
    if (nb < 2) {
        return 0.0f;  // selection only meaningful with >= 2 blocks
    }

#ifdef USE_CUDA
    if (saved_q_rot_.get_device() == Device::GPU &&
        saved_k_rot_.get_device() == Device::GPU &&
        ssa_wsel_.data.get_device() == Device::GPU &&
        gpu_custom_kernels_supported() && d <= 256 && nb <= 512) {
        const int BH = batch_size * nH;
        Tensor q_permuted = Tensor::uninitialized(
            {BH, S, d}, Device::GPU);
        Tensor k_permuted = Tensor::uninitialized(
            {BH, S, d}, Device::GPU);
        launch_attn_gather_heads(
            q_permuted.raw_data(), saved_q_rot_.raw_data(), batch_size, S,
            nH, d, nH, 1, 0);
        launch_attn_gather_heads(
            k_permuted.raw_data(), saved_k_rot_.raw_data(), batch_size, S,
            nH, d, nKV, std::max(kv_group_size, 1), 0);
        Tensor selector_grad = Tensor::zeros({d, d}, Device::GPU);
        Tensor statistics = Tensor::zeros({2}, Device::GPU);
        const float selector_scale =
            1.0f / std::sqrt(static_cast<float>(d));
        const bool deterministic =
            determinism::deterministic_reductions_enabled();
        for (int batch = 0; batch < batch_size; ++batch) {
            const int valid_len = saved_valid_lengths_.empty()
                                      ? S
                                      : std::clamp(
                                            saved_valid_lengths_[static_cast<size_t>(batch)],
                                            0, S);
            const int valid_blocks =
                gpu::ceil_div_positive(valid_len, Bsz);
            if (valid_blocks < 2) continue;
            for (int head = 0; head < nH; ++head) {
                const int bh = batch * nH + head;
                const size_t offset = static_cast<size_t>(bh) * S * d;
                Tensor block_means(
                    {valid_blocks, d}, Device::GPU);
                cuda::launch_sparse_block_means(
                    k_permuted.raw_data() + offset,
                    block_means.raw_data(), valid_len, d, Bsz);
                cuda::launch_sparse_selector_distill(
                    q_permuted.raw_data() + offset,
                    k_permuted.raw_data() + offset,
                    block_means.raw_data(), ssa_wsel_.data.raw_data(),
                    selector_grad.raw_data(), statistics.raw_data(),
                    statistics.raw_data() + 1, valid_len, d, Bsz,
                    selector_scale, deterministic);
            }
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("SSA selector distillation kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        Tensor statistics_host = statistics.cpu();
        const float loss_sum = statistics_host.data()[0];
        const float count = statistics_host.data()[1];
        if (!(count > 0.0f)) return 0.0f;
        const float effective_weight =
            std::isfinite(gradient_weight) ? gradient_weight : 0.0f;
        if (effective_weight != 0.0f) {
            launch_scale_inplace_kernel(
                selector_grad.raw_data(), effective_weight / count,
                selector_grad.size);
            const cudaError_t scale_status = cudaGetLastError();
            if (scale_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("SSA selector gradient scale failed: ") +
                    cudaGetErrorString(scale_status));
            }
            ssa_wsel_.add_grad(selector_grad);
        }
        return loss_sum / count;
    }
#endif

    if (saved_q_rot_.get_device() == Device::GPU &&
        strict_gpu_execution()) {
        throw std::runtime_error(
            "Strict GPU selector distillation requires head_dim <= 256 and "
            "at most 512 blocks");
    }

    Tensor q_host = saved_q_rot_.get_device() == Device::GPU
                        ? saved_q_rot_.cpu()
                        : saved_q_rot_;
    Tensor k_host = saved_k_rot_.get_device() == Device::GPU
                        ? saved_k_rot_.cpu()
                        : saved_k_rot_;
    Tensor w_host = ssa_wsel_.data.get_device() == Device::GPU
                        ? ssa_wsel_.data.cpu()
                        : ssa_wsel_.data;
    const float* qp = q_host.data();
    const float* kp = k_host.data();
    const float* W = w_host.data();
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    Tensor dWsel({d, d}, Device::CPU);  // zero-filled
    float* dw = dWsel.data();
    double loss_sum = 0.0;
    long count = 0;

    std::vector<float> bmean(static_cast<size_t>(nb) * d, 0.0f);
    std::vector<float> attn(static_cast<size_t>(S), 0.0f);
    std::vector<float> mass(static_cast<size_t>(nb), 0.0f);
    std::vector<float> score(static_cast<size_t>(nb), 0.0f);
    std::vector<float> pred(static_cast<size_t>(nb), 0.0f);
    std::vector<float> qsel(static_cast<size_t>(d), 0.0f);

    for (int batch = 0; batch < batch_size; ++batch) {
      const int valid_len = saved_valid_lengths_.empty()
                                ? S
                                : std::clamp(
                                      saved_valid_lengths_[static_cast<size_t>(batch)],
                                      0, S);
      const int valid_nb = gpu::ceil_div_positive(valid_len, Bsz);
      if (valid_nb < 2) continue;
      for (int h = 0; h < nH; ++h) {
        const int kvh = std::min(h / std::max(kv_group_size, 1), nKV - 1);
        std::fill(bmean.begin(), bmean.end(), 0.0f);
        for (int b = 0; b < valid_nb; ++b) {
            const int st = b * Bsz;
            const int en = std::min((b + 1) * Bsz, valid_len);
            for (int j = st; j < en; ++j) {
                const size_t kb =
                    (((static_cast<size_t>(batch) * S + j) * nKV) + kvh) * d;
                for (int c = 0; c < d; ++c)
                    bmean[static_cast<size_t>(b) * d + c] += kp[kb + c];
            }
            const float inv = 1.0f / static_cast<float>(std::max(en - st, 1));
            for (int c = 0; c < d; ++c) bmean[static_cast<size_t>(b) * d + c] *= inv;
        }
        for (int i = 0; i < valid_len; ++i) {
            const int ncand = i / Bsz + 1;  // causal: blocks 0..i/Bsz visible
            if (ncand < 2) continue;
            const size_t qb =
                (((static_cast<size_t>(batch) * S + i) * nH) + h) * d;
            // TARGET: dense attention's per-block mass over j <= i.
            float mx = -1e30f;
            for (int j = 0; j <= i; ++j) {
                const size_t kb =
                    (((static_cast<size_t>(batch) * S + j) * nKV) + kvh) * d;
                float dv = 0.0f;
                for (int c = 0; c < d; ++c) dv += qp[qb + c] * kp[kb + c];
                attn[static_cast<size_t>(j)] = dv * scale;
                mx = std::max(mx, attn[static_cast<size_t>(j)]);
            }
            float sm = 0.0f;
            for (int j = 0; j <= i; ++j) {
                attn[static_cast<size_t>(j)] = std::exp(attn[static_cast<size_t>(j)] - mx);
                sm += attn[static_cast<size_t>(j)];
            }
            const float invsm = 1.0f / (sm + 1e-20f);
            for (int b = 0; b < ncand; ++b) mass[static_cast<size_t>(b)] = 0.0f;
            for (int j = 0; j <= i; ++j)
                mass[static_cast<size_t>(j / Bsz)] += attn[static_cast<size_t>(j)] * invsm;
            // PREDICTED: softmax over candidate blocks of (Wsel @ q) . block_mean.
            for (int a = 0; a < d; ++a) {
                float acc = 0.0f;
                for (int c = 0; c < d; ++c)
                    acc += W[static_cast<size_t>(a) * d + c] * qp[qb + c];
                qsel[static_cast<size_t>(a)] = acc;
            }
            float smx = -1e30f;
            for (int b = 0; b < ncand; ++b) {
                float sc = 0.0f;
                for (int c = 0; c < d; ++c)
                    sc += qsel[static_cast<size_t>(c)] * bmean[static_cast<size_t>(b) * d + c];
                score[static_cast<size_t>(b)] = sc;
                smx = std::max(smx, sc);
            }
            float ssm = 0.0f;
            for (int b = 0; b < ncand; ++b) {
                pred[static_cast<size_t>(b)] = std::exp(score[static_cast<size_t>(b)] - smx);
                ssm += pred[static_cast<size_t>(b)];
            }
            const float invssm = 1.0f / (ssm + 1e-20f);
            for (int b = 0; b < ncand; ++b) {
                pred[static_cast<size_t>(b)] *= invssm;
                loss_sum += -static_cast<double>(mass[static_cast<size_t>(b)]) *
                            std::log(pred[static_cast<size_t>(b)] + 1e-20f);
            }
            ++count;
            // CE+softmax grad: dL/dscore[b] = pred[b] - mass[b].
            // dq_sel[a] = sum_b (pred-mass)[b] * block_mean[b][a];
            // dWsel[a][c] += dq_sel[a] * q[c].
            for (int a = 0; a < d; ++a) {
                float dqa = 0.0f;
                for (int b = 0; b < ncand; ++b)
                    dqa += (pred[static_cast<size_t>(b)] - mass[static_cast<size_t>(b)]) *
                           bmean[static_cast<size_t>(b) * d + a];
                for (int c = 0; c < d; ++c)
                    dw[static_cast<size_t>(a) * d + c] += dqa * qp[qb + c];
            }
        }
      }
    }
    if (count == 0) {
        return 0.0f;
    }
    const float invc = 1.0f / static_cast<float>(count);
    const float effective_weight = std::isfinite(gradient_weight)
                                       ? gradient_weight
                                       : 0.0f;
    if (effective_weight != 0.0f) {
        for (int i = 0; i < dWsel.size; ++i) {
            dw[i] *= invc * effective_weight;
        }
        ssa_wsel_.add_grad(dWsel);
    }
    return static_cast<float>(loss_sum / count);
}

void Attention::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (q_down_proj) out.push_back(q_down_proj.get());
    if (kv_down_proj) out.push_back(kv_down_proj.get());
    if (out_proj) out.push_back(out_proj.get());
}

bool Attention::compact_kv_requested() const {
    if (compact_kv_) return true;  // dtype is stable for the lifetime of a session
    const char* value = std::getenv("NSOS_GPU_KV_DTYPE");
    if (value && std::string(value) != "fp32" && std::string(value) != "fp16")
        throw std::invalid_argument("NSOS_GPU_KV_DTYPE must be fp32 or fp16");
    return !training_mode_ && !sparse_enabled_ && value && std::string(value) == "fp16";
}

void Attention::materialize_compact_kv() {
    if (!compact_kv_) return;
    auto buffers = compact_kv_->materialize(cached_tokens_);
    key_cache_buffer_ = std::move(buffers.first);
    value_cache_buffer_ = std::move(buffers.second);
    compact_kv_.reset();
}

void Attention::ensure_compact_kv_capacity(int tokens, int batch) {
    if (tokens <= 0 || batch <= 0) throw std::invalid_argument("Invalid compact KV request");
    if (compact_kv_ && compact_kv_->batch() != batch)
        throw std::logic_error("Use restore_session_batch to change a live compact KV batch");
    int capacity = std::max({tokens, cache_capacity_tokens_, cache_page_tokens_});
    if (compact_kv_) {
        if (tokens > cache_capacity_tokens_ && cache_capacity_tokens_ <= std::numeric_limits<int>::max() / 2)
            capacity = std::max(capacity, cache_capacity_tokens_ * 2);
        if (capacity != cache_capacity_tokens_ || compact_kv_.use_count() > 1)
            compact_kv_ = compact_kv_->resize(capacity, cached_tokens_);
    } else if (key_cache_buffer_.size) {
        if (cached_batch_size_ != batch) throw std::logic_error("Compact KV batch does not match FP32 cache");
        compact_kv_ = GpuKvCache::from_float(key_cache_buffer_, value_cache_buffer_, batch,
            cache_capacity_tokens_, n_kv_heads * head_dim, cached_tokens_);
        if (capacity != cache_capacity_tokens_) compact_kv_ = compact_kv_->resize(capacity, cached_tokens_);
    } else {
        if (cached_tokens_ != 0) throw std::logic_error("Live KV state has no storage");
        compact_kv_ = std::make_shared<GpuKvCache>(batch, capacity, n_kv_heads * head_dim);
    }
    cache_capacity_tokens_ = capacity;
    cached_batch_size_ = batch;
    key_cache_buffer_ = Tensor();
    value_cache_buffer_ = Tensor();
}

void Attention::clear_kv_cache() {
    compact_kv_.reset();
    cached_tokens_ = 0;
    cached_batch_size_ = 0;
    cache_capacity_tokens_ = 0;
    key_cache_buffer_ = Tensor();
    value_cache_buffer_ = Tensor();
}

void Attention::ensure_kv_cache_capacity(int required_tokens, Device device, int batch_size) {
    materialize_compact_kv();
    const int token_width = n_kv_heads * head_dim;
    const int normalized_batch = std::max(batch_size, 1);
    if (required_tokens <= cache_capacity_tokens_ &&
        cached_batch_size_ == normalized_batch &&
        key_cache_buffer_.size > 0 &&
        value_cache_buffer_.size > 0 &&
        key_cache_buffer_.get_device() == device &&
        value_cache_buffer_.get_device() == device &&
        key_cache_buffer_.data_ptr.use_count() == 1 &&
        value_cache_buffer_.data_ptr.use_count() == 1) {
        return;
    }

    const int target_capacity = std::max(
        required_tokens,
        cache_capacity_tokens_ > 0 ? cache_capacity_tokens_ + cache_page_tokens_ : cache_page_tokens_);
    Tensor next_key({normalized_batch, target_capacity, token_width}, device);
    Tensor next_value({normalized_batch, target_capacity, token_width}, device);

    if (cached_tokens_ > 0 && key_cache_buffer_.size > 0 && value_cache_buffer_.size > 0) {
        const int copy_batch = std::min(std::max(cached_batch_size_, 1), normalized_batch);
        // The buffers are [batch, capacity, token_width].  The OLD buffer's
        // per-sequence stride is the OLD capacity (== cache_capacity_tokens_
        // here, still holding the pre-grow value); the NEW buffer's is
        // target_capacity.  Because they differ on a grow, a single contiguous
        // copy of copy_batch*cached_tokens_ scatters rows>=1 into the wrong
        // region (corrupting every sequence after the first in batched decode).
        // Copy each sequence's live tokens row-by-row with the correct strides.
        const size_t old_row_stride =
            static_cast<size_t>(cache_capacity_tokens_) * token_width;
        const size_t new_row_stride =
            static_cast<size_t>(target_capacity) * token_width;
        const size_t row_bytes = static_cast<size_t>(cached_tokens_) *
                                 static_cast<size_t>(token_width) * sizeof(float);
        float* dst_key = next_key.raw_data();
        float* dst_value = next_value.raw_data();
        const float* src_key = key_cache_buffer_.raw_data();
        const float* src_value = value_cache_buffer_.raw_data();
        for (int b = 0; b < copy_batch; ++b) {
            const size_t dst_off = static_cast<size_t>(b) * new_row_stride;
            const size_t src_off = static_cast<size_t>(b) * old_row_stride;
            copy_float_bytes_device_safe(dst_key + dst_off, next_key.get_device(),
                                         src_key + src_off, key_cache_buffer_.get_device(),
                                         row_bytes);
            copy_float_bytes_device_safe(dst_value + dst_off, next_value.get_device(),
                                         src_value + src_off, value_cache_buffer_.get_device(),
                                         row_bytes);
        }
    }

    key_cache_buffer_ = next_key;
    value_cache_buffer_ = next_value;
    cached_batch_size_ = normalized_batch;
    cache_capacity_tokens_ = target_capacity;
}

void Attention::append_kv_cache_token(const float* key_ptr,
                                      Device key_device,
                                      const float* value_ptr,
                                      Device value_device) {
    const int token_width = n_kv_heads * head_dim;
    if (token_width <= 0) {
        return;
    }

    ensure_kv_cache_capacity(cached_tokens_ + 1,
                             key_cache_buffer_.size > 0 ? key_cache_buffer_.get_device()
                                                        : Device::CPU,
                             1);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    float* key_dst =
        key_cache_buffer_.raw_data() + static_cast<size_t>(cached_tokens_) * token_width;
    float* value_dst =
        value_cache_buffer_.raw_data() + static_cast<size_t>(cached_tokens_) * token_width;
    if (cached_batch_size_ > 1) {
        key_dst = key_cache_buffer_.raw_data() + batch_stride +
                  static_cast<size_t>(cached_tokens_) * token_width;
        value_dst = value_cache_buffer_.raw_data() + batch_stride +
                    static_cast<size_t>(cached_tokens_) * token_width;
    }
    copy_float_bytes_device_safe(key_dst,
                                 key_cache_buffer_.get_device(),
                                 key_ptr,
                                 key_device,
                                 static_cast<size_t>(token_width) * sizeof(float));
    copy_float_bytes_device_safe(value_dst,
                                 value_cache_buffer_.get_device(),
                                 value_ptr,
                                 value_device,
                                 static_cast<size_t>(token_width) * sizeof(float));
    ++cached_tokens_;
}

void Attention::append_kv_cache_batch_tokens(const float* key_ptr,
                                             Device key_device,
                                             const float* value_ptr,
                                             Device value_device,
                                             int batch_size) {
    const int token_width = n_kv_heads * head_dim;
    const int normalized_batch = std::max(batch_size, 1);
    if (token_width <= 0) {
        return;
    }
    ensure_kv_cache_capacity(cached_tokens_ + 1,
                             key_cache_buffer_.size > 0 ? key_cache_buffer_.get_device()
                                                        : Device::CPU,
                             normalized_batch);
    const size_t token_bytes = static_cast<size_t>(token_width) * sizeof(float);
    const size_t src_stride = static_cast<size_t>(token_width);
    const size_t dst_batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    for (int batch = 0; batch < normalized_batch; ++batch) {
        float* key_dst = key_cache_buffer_.raw_data() +
                         static_cast<size_t>(batch) * dst_batch_stride +
                         static_cast<size_t>(cached_tokens_) * static_cast<size_t>(token_width);
        float* value_dst = value_cache_buffer_.raw_data() +
                           static_cast<size_t>(batch) * dst_batch_stride +
                           static_cast<size_t>(cached_tokens_) * static_cast<size_t>(token_width);
        const float* key_src = key_ptr + static_cast<size_t>(batch) * src_stride;
        const float* value_src = value_ptr + static_cast<size_t>(batch) * src_stride;
        copy_float_bytes_device_safe(key_dst,
                                     key_cache_buffer_.get_device(),
                                     key_src,
                                     key_device,
                                     token_bytes);
        copy_float_bytes_device_safe(value_dst,
                                     value_cache_buffer_.get_device(),
                                     value_src,
                                     value_device,
                                     token_bytes);
    }
    ++cached_tokens_;
}

const float* Attention::kv_cache_token_ptr(
    const Tensor& cache,
    int token_index) const {
    const size_t token_width = static_cast<size_t>(n_kv_heads * head_dim);
    return cache.data() + static_cast<size_t>(token_index) * token_width;
}

const float* Attention::kv_cache_token_ptr(
    const Tensor& cache,
    int batch_index,
    int token_index) const {
    const size_t token_width = static_cast<size_t>(n_kv_heads * head_dim);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    return cache.data() + static_cast<size_t>(batch_index) * batch_stride +
           static_cast<size_t>(token_index) * token_width;
}

void Attention::set_streaming_mode(bool enabled) {
    streaming_inference_ = enabled;
    clear_kv_cache();
}

void Attention::reset() {
    if (rdna_training_) rdna_training_->cancel_pending();
    saved_rdna_training_ = false;
    clear_kv_cache();
}

AttentionCacheSnapshot Attention::snapshot_cache() const {
    AttentionCacheSnapshot snapshot;
    snapshot.enabled = streaming_inference_;
    snapshot.cached_tokens = cached_tokens_;
    snapshot.cache_page_tokens = cache_page_tokens_;
    snapshot.cache_capacity_tokens = cache_capacity_tokens_;
    if (compact_kv_) {
        snapshot.compact = cached_batch_size_ <= 1 ? compact_kv_ : compact_kv_->row(0, cached_tokens_);
        return snapshot;
    }
    if (cached_batch_size_ <= 1) {
        snapshot.key_cache = key_cache_buffer_;
        snapshot.value_cache = value_cache_buffer_;
    } else if (key_cache_buffer_.size > 0 && value_cache_buffer_.size > 0) {
        const int token_width = n_kv_heads * head_dim;
        snapshot.key_cache = Tensor({cache_capacity_tokens_, token_width}, key_cache_buffer_.get_device());
        snapshot.value_cache =
            Tensor({cache_capacity_tokens_, token_width}, value_cache_buffer_.get_device());
        const size_t bytes = static_cast<size_t>(cache_capacity_tokens_) *
                             static_cast<size_t>(token_width) * sizeof(float);
        copy_float_bytes_device_safe(snapshot.key_cache.raw_data(),
                                     snapshot.key_cache.get_device(),
                                     key_cache_buffer_.raw_data(),
                                     key_cache_buffer_.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(snapshot.value_cache.raw_data(),
                                     snapshot.value_cache.get_device(),
                                     value_cache_buffer_.raw_data(),
                                     value_cache_buffer_.get_device(),
                                     bytes);
    }
    return snapshot;
}

std::vector<AttentionCacheSnapshot> Attention::snapshot_cache_batch() const {
    std::vector<AttentionCacheSnapshot> snapshots;
    if (compact_kv_) {
        for (int b = 0; b < cached_batch_size_; ++b) {
            AttentionCacheSnapshot snapshot;
            snapshot.enabled = streaming_inference_;
            snapshot.cached_tokens = cached_tokens_;
            snapshot.cache_page_tokens = cache_page_tokens_;
            snapshot.cache_capacity_tokens = cache_capacity_tokens_;
            snapshot.compact = compact_kv_->row(b, cached_tokens_);
            snapshots.push_back(std::move(snapshot));
        }
        return snapshots;
    }
    if (cached_batch_size_ <= 0 || key_cache_buffer_.size == 0 || value_cache_buffer_.size == 0) {
        return snapshots;
    }
    const int token_width = n_kv_heads * head_dim;
    const size_t bytes = static_cast<size_t>(cache_capacity_tokens_) *
                         static_cast<size_t>(token_width) * sizeof(float);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    snapshots.reserve(static_cast<size_t>(cached_batch_size_));
    for (int batch = 0; batch < cached_batch_size_; ++batch) {
        AttentionCacheSnapshot snapshot;
        snapshot.enabled = streaming_inference_;
        snapshot.cached_tokens = cached_tokens_;
        snapshot.cache_page_tokens = cache_page_tokens_;
        snapshot.cache_capacity_tokens = cache_capacity_tokens_;
        snapshot.key_cache =
            Tensor({cache_capacity_tokens_, token_width}, key_cache_buffer_.get_device());
        snapshot.value_cache =
            Tensor({cache_capacity_tokens_, token_width}, value_cache_buffer_.get_device());
        copy_float_bytes_device_safe(snapshot.key_cache.raw_data(),
                                     snapshot.key_cache.get_device(),
                                     key_cache_buffer_.raw_data() +
                                         static_cast<size_t>(batch) * batch_stride,
                                     key_cache_buffer_.get_device(),
                                     bytes);
        copy_float_bytes_device_safe(snapshot.value_cache.raw_data(),
                                     snapshot.value_cache.get_device(),
                                     value_cache_buffer_.raw_data() +
                                         static_cast<size_t>(batch) * batch_stride,
                                     value_cache_buffer_.get_device(),
                                     bytes);
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void Attention::restore_cache(const AttentionCacheSnapshot& snapshot) {
    if (snapshot.cached_tokens < 0 || snapshot.cache_capacity_tokens < snapshot.cached_tokens)
        throw std::invalid_argument("Invalid attention snapshot length");
    if (snapshot.compact && (snapshot.compact->batch() != 1 ||
        snapshot.compact->capacity() != snapshot.cache_capacity_tokens ||
        snapshot.compact->width() != n_kv_heads * head_dim))
        throw std::invalid_argument("Incompatible compact attention snapshot");
    compact_kv_ = snapshot.compact;
    streaming_inference_ = snapshot.enabled;
    cached_tokens_ = snapshot.cached_tokens;
    cache_page_tokens_ = std::max(snapshot.cache_page_tokens, 1);
    cache_capacity_tokens_ = std::max(snapshot.cache_capacity_tokens, 0);
    cached_batch_size_ = 1;
    key_cache_buffer_ = snapshot.key_cache;
    value_cache_buffer_ = snapshot.value_cache;
}

void Attention::restore_cache_batch(const std::vector<AttentionCacheSnapshot>& snapshots) {
    if (snapshots.empty()) {
        clear_kv_cache();
        return;
    }
    const auto& first = snapshots.front();
    if (std::any_of(snapshots.begin(), snapshots.end(), [](const auto& s) { return bool(s.compact); })) {
        const bool all_compact = std::all_of(snapshots.begin(), snapshots.end(), [](const auto& s) { return bool(s.compact); });
        if (!all_compact) {
            auto expanded = snapshots;
            for (auto& snapshot : expanded) if (snapshot.compact) {
                auto values = snapshot.compact->materialize(snapshot.cached_tokens);
                snapshot.key_cache = std::move(values.first);
                snapshot.value_cache = std::move(values.second);
                snapshot.compact.reset();
            }
            restore_cache_batch(expanded);
            return;
        }
        int capacity = 0;
        for (const auto& s : snapshots) {
            if (s.enabled != first.enabled || s.cached_tokens != first.cached_tokens ||
                s.cached_tokens < 0 || s.cache_capacity_tokens < s.cached_tokens ||
                s.compact->batch() != 1 || s.compact->width() != n_kv_heads * head_dim ||
                s.compact->capacity() != s.cache_capacity_tokens)
                throw std::invalid_argument("Incompatible compact attention batch");
            capacity = std::max(capacity, s.cache_capacity_tokens);
        }
        auto next = std::make_shared<GpuKvCache>(int(snapshots.size()), capacity, n_kv_heads * head_dim);
        for (size_t b = 0; b < snapshots.size(); ++b)
            next->copy_row_from(int(b), *snapshots[b].compact, 0, first.cached_tokens);
        compact_kv_ = std::move(next);
        streaming_inference_ = first.enabled;
        cached_tokens_ = first.cached_tokens;
        cached_batch_size_ = int(snapshots.size());
        cache_capacity_tokens_ = capacity;
        cache_page_tokens_ = std::max(first.cache_page_tokens, 1);
        key_cache_buffer_ = Tensor();
        value_cache_buffer_ = Tensor();
        return;
    }
    compact_kv_.reset();
    if (first.cached_tokens < 0 ||
        first.cache_capacity_tokens < first.cached_tokens) {
        throw std::runtime_error("Invalid attention cache snapshot metadata");
    }
    const int token_width = n_kv_heads * head_dim;
    int target_capacity = first.cache_capacity_tokens;
    for (const auto& snapshot : snapshots) {
        if (snapshot.enabled != first.enabled ||
            snapshot.cached_tokens != first.cached_tokens) {
            throw std::runtime_error(
                "Batched attention restore requires equal sequence lengths");
        }
        if (snapshot.cached_tokens < 0 ||
            snapshot.cache_capacity_tokens < snapshot.cached_tokens) {
            throw std::runtime_error("Invalid attention cache snapshot metadata");
        }
        const uint64_t required_elements =
            static_cast<uint64_t>(snapshot.cache_capacity_tokens) *
            static_cast<uint64_t>(token_width);
        if (static_cast<uint64_t>(snapshot.key_cache.size) < required_elements ||
            static_cast<uint64_t>(snapshot.value_cache.size) < required_elements) {
            throw std::runtime_error("Attention cache snapshot buffer is truncated");
        }
        target_capacity = std::max(target_capacity, snapshot.cache_capacity_tokens);
    }

    streaming_inference_ = first.enabled;
    cached_tokens_ = first.cached_tokens;
    cache_page_tokens_ = std::max(first.cache_page_tokens, 1);
    cache_capacity_tokens_ = target_capacity;
    cached_batch_size_ = static_cast<int>(snapshots.size());
    const Device cache_device =
        first.key_cache.size > 0 ? first.key_cache.get_device() : Device::CPU;
    key_cache_buffer_ = Tensor({cached_batch_size_, cache_capacity_tokens_, token_width}, cache_device);
    value_cache_buffer_ =
        Tensor({cached_batch_size_, cache_capacity_tokens_, token_width}, cache_device);
    const size_t live_bytes = static_cast<size_t>(cached_tokens_) *
                              static_cast<size_t>(token_width) * sizeof(float);
    const size_t batch_stride =
        static_cast<size_t>(cache_capacity_tokens_) * static_cast<size_t>(token_width);
    for (size_t batch = 0; batch < snapshots.size(); ++batch) {
        copy_float_bytes_device_safe(key_cache_buffer_.raw_data() + batch * batch_stride,
                                     key_cache_buffer_.get_device(),
                                      snapshots[batch].key_cache.raw_data(),
                                     snapshots[batch].key_cache.get_device(),
                                     live_bytes);
        copy_float_bytes_device_safe(value_cache_buffer_.raw_data() + batch * batch_stride,
                                     value_cache_buffer_.get_device(),
                                      snapshots[batch].value_cache.raw_data(),
                                     snapshots[batch].value_cache.get_device(),
                                     live_bytes);
    }
}

MoERouter::MoERouter(int d_model_value, int n, int k)
    : num_experts(n),
      top_k(k),
      aux_loss_coef(0.01f),
      expert_loads(n > 0 ? static_cast<size_t>(n) : 0U, 0.0f),
      gate(std::make_unique<BitLinear>(d_model_value, n > 0 ? n : 1, false)) {
  if (d_model_value <= 0 || n <= 0 || k <= 0 || k > n) {
    throw std::invalid_argument(
        "MoERouter requires d_model>0, num_experts>0 and top_k in "
        "[1, num_experts]");
  }
  // Opt-in full-precision router (NSOS_MOE_FP_ROUTER=1).  Routing is sensitive:
  // under QAT a ternary gate yields coarse, unstable assignments (risk of expert
  // collapse).  Marking the gate quantization-sensitive keeps it on the float
  // reference path through QAT (the scheduler honors this flag — see
  // bitlinear.h set_quantization_sensitive).  Default OFF preserves the
  // historical ternary-under-QAT behavior byte-for-byte.  Pairs with the
  // differentiable Switch aux-loss (NSOS_MOE_SWITCH_AUX, trainer.cpp).
  // K2: full-precision router is ON by default — a ternary gate yields coarse,
  // unstable routing under QAT (expert collapse).  NSOS_MOE_FP_ROUTER=0 forces
  // the ternary gate (historical behavior) if ever needed.
  if (const char* e = std::getenv("NSOS_MOE_FP_ROUTER"); e == nullptr || e[0] != '0') {
    gate->set_quantization_sensitive(true);
  }
}

void MoERouter::begin_aux_accumulation() {
    aux_accumulation_active_ = true;
    aux_forward_records_.clear();
    accumulated_expert_loads_.assign(static_cast<size_t>(num_experts), 0.0f);
    pending_expert_loads_device_ = Tensor();
    accumulated_expert_loads_device_ = Tensor();
}

void MoERouter::materialize_expert_loads() {
    if (pending_expert_loads_device_.size == 0) return;
    const Tensor host = pending_expert_loads_device_.cpu();
    expert_loads.assign(host.data(), host.data() + num_experts);
    pending_expert_loads_device_ = Tensor();
}

void MoERouter::finalize_aux_accumulation(bool materialize_loads) {
    if (aux_accumulation_active_ && accumulated_expert_loads_device_.size > 0) {
        // Preserve final step telemetry without a mandatory per-layer fence.
        // Audit callers can materialize this after all gradient work is queued.
        pending_expert_loads_device_ = accumulated_expert_loads_device_;
    } else if (aux_accumulation_active_ &&
        accumulated_expert_loads_.size() == expert_loads.size()) {
        expert_loads = accumulated_expert_loads_;
    }
    aux_accumulation_active_ = false;
    aux_forward_records_.clear();
    accumulated_expert_loads_.clear();
    accumulated_expert_loads_device_ = Tensor();
    if (materialize_loads) materialize_expert_loads();
}

void MoERouter::cancel_aux_accumulation() {
    aux_accumulation_active_ = false;
    aux_forward_records_.clear();
    accumulated_expert_loads_.clear();
    pending_expert_loads_device_ = Tensor();
    accumulated_expert_loads_device_ = Tensor();
}

std::pair<Tensor, Tensor> MoERouter::forward(
    const Tensor& x, const std::vector<uint8_t>& valid_rows) {
    if (x.shape.empty() || x.shape.back() <= 0) {
        throw std::invalid_argument("MoERouter::forward requires a non-empty feature dimension");
    }
    const int rows = x.size / x.shape.back();
    if (!valid_rows.empty()) {
        if (valid_rows.size() != static_cast<size_t>(rows)) {
            throw std::invalid_argument("MoERouter valid-row mask size mismatch");
        }
        if (std::any_of(valid_rows.begin(), valid_rows.end(),
                        [](uint8_t value) { return value > 1U; })) {
            throw std::invalid_argument("MoERouter valid-row mask must contain only 0 or 1");
        }
    }
    Tensor logits = gate->forward(x);
    Tensor weights = logits.softmax(-1);

    // Save the PRE-mask softmax probabilities on their original device for the
    // Switch aux loss.  Captured before top-k masking so it is the true routing
    // distribution p[i,e]. The clone is required because weights is mutated
    // Switch aux path is enabled — which is a TRAINING path, so at inference we
    // below. At inference the training-only retention is skipped entirely.
    if (gate->training_mode()) {
        saved_probs_ = weights.clone();
        if (aux_accumulation_active_) {
            aux_forward_records_.push_back(
                AuxForwardRecord{saved_probs_, x.clone(), valid_rows});
        }
    } else {
        saved_probs_ = Tensor();
    }

    pending_expert_loads_device_ = Tensor();
    std::fill(expert_loads.begin(), expert_loads.end(), 0.0f);
    if (rows <= 0 || num_experts <= 0) {
        return {logits, weights};
    }

#ifdef USE_CUDA
    // GPU fast-path: avoid the GPU→CPU→GPU round-trip on `weights`
    // (which used to fire on every routed token in every MoE layer at
    // every step).  We:
    //   * Apply the top-k mask + renormalization in-place on the device.
    //   * Accumulate per-expert loads into a small device buffer.
    //   * Issue a single tiny D2H copy of expert_loads at the end.
    //
    // Eligibility: weights must already live on the device, and the
    // expected layout is contiguous [rows, num_experts].
    if (weights.get_device() == Device::GPU) {
        // Top-k mask + renormalize in-place.  No-op when top_k >=
        // num_experts (kernel takes the renormalize-only branch).
        const int eff_top_k =
            (top_k > 0 && top_k < num_experts) ? top_k : num_experts;
        launch_moe_topk_mask_kernel(weights.raw_data(), rows, num_experts,
                                    eff_top_k);

        // Padding is semantically absent from routing, dispatch and load
        // statistics. Zero invalid rows on the device before either consumer.
        if (!valid_rows.empty()) {
            uint8_t* valid_rows_gpu =
                moe_valid_rows_scratch(valid_rows.size());
            const cudaError_t copy_status = cudaMemcpy(
                valid_rows_gpu, valid_rows.data(), valid_rows.size(),
                cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("MoE valid-row mask upload failed: ") +
                    cudaGetErrorString(copy_status));
            }
            record_gpu_transfer(
                Device::GPU, Device::CPU, valid_rows.size());
            launch_moe_zero_invalid_rows_kernel(
                weights.raw_data(), valid_rows_gpu, rows, num_experts);
            const cudaError_t launch_status = cudaGetLastError();
            if (launch_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("MoE valid-row mask kernel failed: ") +
                    cudaGetErrorString(launch_status));
            }
        }

        // Per-expert load accumulation: only training consumers read
        // expert_loads (aux regularization, layer-audit router records) — at
        // inference the D2H below is a per-token synchronous copy that costs
        // decode latency and aborts CUDA-graph capture, so skip it and leave
        // expert_loads zeroed (already reset above).
        if (gate->training_mode()) {
            // Fresh buffer rather than one reused across calls so the device
            // memset is safe regardless of previous async work.
            Tensor loads_gpu = Tensor::zeros(
                std::vector<int>{num_experts}, Device::GPU);
            if (training_policy::ordered_moe()) {
                launch_moe_load_ordered_kernel(weights.raw_data(), loads_gpu.raw_data(), rows, num_experts);
            } else {
                launch_moe_load_accumulate_kernel(weights.raw_data(),
                                              loads_gpu.raw_data(),
                                              rows, num_experts);
            }
            gpu_runtime_check(cudaGetLastError(), "MoE load reduction");
            if (aux_accumulation_active_) {
                pending_expert_loads_device_ = loads_gpu;
                if (accumulated_expert_loads_device_.size == 0)
                    accumulated_expert_loads_device_ = loads_gpu.clone();
                else
                    accumulated_expert_loads_device_.add_inplace_(loads_gpu);
                return {logits, weights};
            }
            Tensor loads_host = loads_gpu.cpu();
            const float* loads_ptr = loads_host.data();
            for (int e = 0; e < num_experts; ++e) {
                expert_loads[e] = loads_ptr[e];
            }
            if (aux_accumulation_active_ &&
                accumulated_expert_loads_.size() == expert_loads.size()) {
                for (size_t e = 0; e < expert_loads.size(); ++e) {
                    accumulated_expert_loads_[e] += expert_loads[e];
                }
            }
        }
        return {logits, weights};
    }
#endif

    // CPU path (unchanged) — kept for builds without CUDA and for the
    // case where weights happen to live on the host (e.g. eval scripts).
    Tensor weights_host = weights;
    float* weight_ptr = weights_host.data();
    if (top_k > 0 && top_k < num_experts) {
        std::vector<int> ranked(static_cast<size_t>(num_experts));
        std::vector<char> keep(static_cast<size_t>(num_experts), 0);
        std::iota(ranked.begin(), ranked.end(), 0);
        for (int row = 0; row < rows; ++row) {
            std::fill(keep.begin(), keep.end(), 0);
            std::partial_sort(
                ranked.begin(),
                ranked.begin() + top_k,
                ranked.end(),
                [&](int lhs, int rhs) {
                    const float lhs_value =
                        weight_ptr[row * num_experts + lhs];
                    const float rhs_value =
                        weight_ptr[row * num_experts + rhs];
                    return lhs_value > rhs_value ||
                           (lhs_value == rhs_value && lhs < rhs);
                });
            float selected_sum = 0.0f;
            for (int rank = 0; rank < top_k; ++rank) {
                keep[static_cast<size_t>(ranked[rank])] = 1;
                selected_sum += weight_ptr[row * num_experts + ranked[rank]];
            }
            const float inv_selected = 1.0f / std::max(selected_sum, 1e-9f);
            for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                float& value = weight_ptr[row * num_experts + expert_idx];
                value = keep[static_cast<size_t>(expert_idx)] ? (value * inv_selected) : 0.0f;
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        if (!valid_rows.empty() && valid_rows[static_cast<size_t>(row)] == 0U) {
            std::fill_n(weight_ptr + static_cast<size_t>(row) * num_experts,
                        num_experts, 0.0f);
            continue;
        }
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            expert_loads[expert_idx] += weight_ptr[row * num_experts + expert_idx];
        }
    }

    if (aux_accumulation_active_ &&
        accumulated_expert_loads_.size() == expert_loads.size()) {
        for (size_t e = 0; e < expert_loads.size(); ++e) {
            accumulated_expert_loads_[e] += expert_loads[e];
        }
    }

    weights = weights_host;
    return {logits, weights};
}

Tensor MoERouter::backward(const Tensor& grad_logits) {
    return gate ? gate->backward(grad_logits) : grad_logits;
}

void MoERouter::to(Device dev) {
    if (gate && gate->weight.data.get_device() != dev) {
        cancel_aux_accumulation();
        saved_probs_ = Tensor();
    }
    if (gate) {
        gate->to(dev);
    }
}

std::vector<Parameter*> MoERouter::parameters() {
    std::vector<Parameter*> params;
    if (gate) {
        auto gate_params = gate->parameters();
        prefix_parameter_names(gate_params, "gate.");
        params.insert(params.end(), gate_params.begin(), gate_params.end());
    }
    if (shared_expert_gate) {
        auto shared_gate_params = shared_expert_gate->parameters();
        prefix_parameter_names(shared_gate_params, "shared_expert_gate.");
        params.insert(params.end(), shared_gate_params.begin(), shared_gate_params.end());
    }
    if (shared_expert_up) {
        auto shared_up_params = shared_expert_up->parameters();
        prefix_parameter_names(shared_up_params, "shared_expert_up.");
        params.insert(params.end(), shared_up_params.begin(), shared_up_params.end());
    }
    if (shared_expert_down) {
        auto shared_down_params = shared_expert_down->parameters();
        prefix_parameter_names(shared_down_params, "shared_expert_down.");
        params.insert(params.end(), shared_down_params.begin(), shared_down_params.end());
    }
    return params;
}

void MoERouter::collect_bitlinear_layers(std::vector<BitLinear*>& out) {
    if (gate) out.push_back(gate.get());
    if (shared_expert_gate) out.push_back(shared_expert_gate.get());
    if (shared_expert_up) out.push_back(shared_expert_up.get());
    if (shared_expert_down) out.push_back(shared_expert_down.get());
}

// (compute_aux_loss removido — zero call sites; o aux loss vivo é o Switch
// diferenciável em switch_aux_grad_logits/accumulate_switch_aux_grad.)

Tensor MoERouter::switch_aux_grad_logits(const Tensor& probs, int top_k,
                                         float coef, float* out_loss) {
    Tensor loss;
    Tensor grad = switch_aux_grad_logits_device(probs, top_k, coef,
                                                out_loss ? &loss : nullptr);
    if (out_loss) *out_loss = loss.cpu().data()[0];
    return grad;
}

Tensor MoERouter::switch_aux_grad_logits_device(const Tensor& probs, int top_k,
                                                float coef, Tensor* out_loss) {
    // probs: [T, N] pre-mask softmax (host or device).  Returns grad wrt the
    // router logits [T, N].  L = coef·N·Σ_e f_e·P_e, with f_e the hard dispatch
    // fraction (stop-grad) and P_e = mean_i p[i,e].
    if (probs.shape.empty()) {
        throw std::invalid_argument("Switch aux probabilities require rank >= 1");
    }
    if (!std::isfinite(coef)) {
        throw std::invalid_argument("Switch aux coefficient must be finite");
    }
    const int N = probs.shape.back();
    const int T = N > 0 ? probs.size / N : 0;
    if (out_loss) {
        *out_loss = Tensor::zeros({1}, probs.get_device());
    }
    if (T <= 0 || N <= 0) {
        return Tensor::zeros({std::max(T, 0), std::max(N, 0)},
                             probs.get_device());
    }
    const int k = std::clamp(top_k, 1, N);

#ifdef USE_CUDA
    if (probs.get_device() == Device::GPU) {
        const bool deterministic =
            determinism::deterministic_reductions_enabled();
        Tensor counts = deterministic ? Tensor::uninitialized({N}, Device::GPU) : Tensor::zeros({N}, Device::GPU);
        Tensor probability_sums = deterministic ? Tensor::uninitialized({N}, Device::GPU) : Tensor::zeros({N}, Device::GPU);
        Tensor grad = Tensor::uninitialized({T, N}, Device::GPU);
        launch_moe_switch_aux_stats_kernel(
            probs.raw_data(), counts.raw_data(), probability_sums.raw_data(),
            T, N, k, deterministic);
        launch_moe_switch_aux_grad_kernel(
            probs.raw_data(), counts.raw_data(), grad.raw_data(), T, N, coef);
        if (out_loss != nullptr) {
            launch_moe_switch_aux_loss_kernel(
                counts.raw_data(), probability_sums.raw_data(),
                out_loss->raw_data(), T, N, coef, deterministic);
        }
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Switch aux GPU kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        return grad;
    }
#endif

    Tensor grad = Tensor::zeros({T, N}, Device::CPU);
    const float* p = probs.data();
    float* gz = grad.data();

    // Hard dispatch fraction f_e = (1/T) Σ_i 1[e ∈ topk(i)]  (treated as const).
    std::vector<double> f(static_cast<size_t>(N), 0.0);
    std::vector<int> ranked(static_cast<size_t>(N));
    for (int i = 0; i < T; ++i) {
        std::iota(ranked.begin(), ranked.end(), 0);
        std::partial_sort(
            ranked.begin(), ranked.begin() + k, ranked.end(),
            [&](int a, int b) {
                const float lhs = p[static_cast<size_t>(i) * N + a];
                const float rhs = p[static_cast<size_t>(i) * N + b];
                return lhs > rhs || (lhs == rhs && a < b);
            });
        for (int r = 0; r < k; ++r) {
            f[static_cast<size_t>(ranked[static_cast<size_t>(r)])] += 1.0;
        }
    }
    for (int e = 0; e < N; ++e) {
        f[static_cast<size_t>(e)] /= static_cast<double>(T);
    }

    if (out_loss) {
        std::vector<double> P(static_cast<size_t>(N), 0.0);
        for (int i = 0; i < T; ++i) {
            for (int e = 0; e < N; ++e) {
                P[static_cast<size_t>(e)] += p[static_cast<size_t>(i) * N + e];
            }
        }
        double L = 0.0;
        for (int e = 0; e < N; ++e) {
            P[static_cast<size_t>(e)] /= static_cast<double>(T);
            L += f[static_cast<size_t>(e)] * P[static_cast<size_t>(e)];
        }
        out_loss->data()[0] = static_cast<float>(static_cast<double>(coef) * N * L);
    }

    // grad_z[i,e] = (coef·N/T)·p[i,e]·(f_e − Σ_e' f_e'·p[i,e']).
    const double scale = static_cast<double>(coef) * static_cast<double>(N) /
                         static_cast<double>(T);
    for (int i = 0; i < T; ++i) {
        double dot = 0.0;
        for (int e = 0; e < N; ++e) {
            dot += f[static_cast<size_t>(e)] * p[static_cast<size_t>(i) * N + e];
        }
        for (int e = 0; e < N; ++e) {
            const size_t idx = static_cast<size_t>(i) * N + e;
            gz[idx] = static_cast<float>(scale * p[idx] *
                                         (f[static_cast<size_t>(e)] - dot));
        }
    }
    return grad;
}

float MoERouter::accumulate_switch_aux_grad(float coef) {
    const Tensor loss = accumulate_switch_aux_grad_device(coef);
    materialize_expert_loads();
    return loss.cpu().data()[0];
}

Tensor MoERouter::accumulate_switch_aux_grad_device(float coef) {
    const Device loss_device = gate ? gate->weight.data.get_device() : Device::CPU;
    if (!std::isfinite(coef)) {
        cancel_aux_accumulation();
        throw std::invalid_argument("Switch aux coefficient must be finite");
    }
    if (!gate || coef == 0.0f) {
        cancel_aux_accumulation();
        return Tensor::zeros({1}, loss_device);
    }

    // Exact once-per-step objective across every bucket/micro-chunk.  Building
    // the global probability matrix is important: averaging independently
    // computed f_e * P_e products is not equivalent to computing the Switch
    // objective from the complete batch. Inputs and probabilities remain on
    // their original device; one aggregate gate replay produces one exact VJP.
    if (!aux_forward_records_.empty()) {
        const Device device = aux_forward_records_.front().probs.get_device();
        const int input_dim = aux_forward_records_.front().input.shape.empty()
                                  ? 0
                                  : aux_forward_records_.front().input.shape.back();
        if (input_dim <= 0) {
            cancel_aux_accumulation();
            throw std::runtime_error("Switch aux record has no input feature dimension");
        }
        int total_rows = 0;
        for (const auto& record : aux_forward_records_) {
            if (record.probs.get_device() != device ||
                record.input.get_device() != device) {
                cancel_aux_accumulation();
                throw std::runtime_error("Switch aux records span multiple devices");
            }
            if (record.probs.size > 0 && num_experts > 0) {
                if (record.probs.size % num_experts != 0) {
                    cancel_aux_accumulation();
                    throw std::runtime_error("Switch aux probability shape is inconsistent");
                }
                const int record_rows = record.probs.size / num_experts;
                if (record.input.size != record_rows * input_dim) {
                    cancel_aux_accumulation();
                    throw std::runtime_error("Switch aux input/probability row mismatch");
                }
                if (!record.valid_rows.empty() &&
                    record.valid_rows.size() != static_cast<size_t>(record_rows)) {
                    cancel_aux_accumulation();
                    throw std::runtime_error("Switch aux valid-row mask mismatch");
                }
                if (record.valid_rows.empty()) {
                    if (record_rows > std::numeric_limits<int>::max() - total_rows) {
                        cancel_aux_accumulation();
                        throw std::length_error("Switch aux aggregate exceeds row indexing limit");
                    }
                    total_rows += record_rows;
                } else {
                    const int added_rows = static_cast<int>(std::count(
                        record.valid_rows.begin(), record.valid_rows.end(),
                        static_cast<uint8_t>(1U)));
                    if (added_rows > std::numeric_limits<int>::max() - total_rows) {
                        cancel_aux_accumulation();
                        throw std::length_error("Switch aux aggregate exceeds row indexing limit");
                    }
                    total_rows += added_rows;
                }
            }
        }
        if (total_rows <= 0) {
            cancel_aux_accumulation();
            return Tensor::zeros({1}, loss_device);
        }
        if (total_rows > std::numeric_limits<int>::max() / std::max(input_dim, num_experts)) {
            cancel_aux_accumulation();
            throw std::length_error("Switch aux aggregate exceeds tensor indexing limit");
        }

        Tensor all_probs = Tensor::uninitialized({total_rows, num_experts}, device);
        Tensor all_input = Tensor::uninitialized({total_rows, input_dim}, device);
        int row_offset = 0;
        for (const auto& record : aux_forward_records_) {
            const int rows = record.probs.size / num_experts;
            if (rows <= 0) continue;
            const int valid_count = record.valid_rows.empty()
                                        ? rows
                                        : static_cast<int>(std::count(
                                              record.valid_rows.begin(),
                                              record.valid_rows.end(),
                                              static_cast<uint8_t>(1U)));
            if (valid_count == 0) {
                continue;
            }
            if (record.valid_rows.empty()) {
                copy_tensor_bytes(
                    all_probs.raw_data() + static_cast<size_t>(row_offset) * num_experts,
                    device, record.probs.raw_data(), device,
                    static_cast<size_t>(rows) * num_experts * sizeof(float));
                copy_tensor_bytes(
                    all_input.raw_data() + static_cast<size_t>(row_offset) * input_dim,
                    device, record.input.raw_data(), device,
                    static_cast<size_t>(rows) * input_dim * sizeof(float));
            } else if (device == Device::CPU) {
                for (int row = 0, destination_row = 0; row < rows; ++row) {
                    if (record.valid_rows[static_cast<size_t>(row)] == 0U) continue;
                    std::memcpy(
                        all_probs.data() +
                            static_cast<size_t>(row_offset + destination_row) * num_experts,
                        record.probs.data() + static_cast<size_t>(row) * num_experts,
                        static_cast<size_t>(num_experts) * sizeof(float));
                    std::memcpy(
                        all_input.data() +
                            static_cast<size_t>(row_offset + destination_row) * input_dim,
                        record.input.data() + static_cast<size_t>(row) * input_dim,
                        static_cast<size_t>(input_dim) * sizeof(float));
                    ++destination_row;
                }
            } else {
#ifdef USE_CUDA
                std::vector<int> valid_indices;
                valid_indices.reserve(static_cast<size_t>(valid_count));
                for (int row = 0; row < rows; ++row) {
                    if (record.valid_rows[static_cast<size_t>(row)] != 0U) {
                        valid_indices.push_back(row);
                    }
                }
                int* indices_gpu =
                    moe_valid_indices_scratch(valid_indices.size());
                const cudaError_t copy_status = cudaMemcpy(
                    indices_gpu, valid_indices.data(),
                    valid_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
                if (copy_status != cudaSuccess) {
                    cancel_aux_accumulation();
                    throw std::runtime_error(
                        std::string("Switch aux index upload failed: ") +
                        cudaGetErrorString(copy_status));
                }
                record_gpu_transfer(
                    Device::GPU, Device::CPU,
                    valid_indices.size() * sizeof(int));
                launch_moe_gather_rows_kernel(
                    record.probs.raw_data(), indices_gpu,
                    all_probs.raw_data() +
                        static_cast<size_t>(row_offset) * num_experts,
                    valid_count, num_experts);
                launch_moe_gather_rows_kernel(
                    record.input.raw_data(), indices_gpu,
                    all_input.raw_data() +
                        static_cast<size_t>(row_offset) * input_dim,
                    valid_count, input_dim);
                const cudaError_t launch_status = cudaGetLastError();
                if (launch_status != cudaSuccess) {
                    cancel_aux_accumulation();
                    throw std::runtime_error(
                        std::string("Switch aux row gather failed: ") +
                        cudaGetErrorString(launch_status));
                }
#else
                cancel_aux_accumulation();
                throw std::runtime_error("GPU Switch aux requires a GPU build");
#endif
            }
            row_offset += valid_count;
        }
        if (row_offset != total_rows) {
            cancel_aux_accumulation();
            throw std::runtime_error("Switch aux aggregation row-count mismatch");
        }

        Tensor gate_logits = gate->forward(all_input);
        Tensor loss;
        Tensor all_grad = switch_aux_grad_logits_device(all_probs, top_k, coef, &loss);
        if (all_grad.size == gate_logits.size &&
            all_grad.shape != gate_logits.shape) {
            all_grad = all_grad.reshape(gate_logits.shape.dims);
        }
        if (gate->weight.data.get_device() != all_grad.get_device()) {
            all_grad = all_grad.to(gate->weight.data.get_device());
        }
        (void)gate->backward(all_grad);
        finalize_aux_accumulation(false);
        return loss;
    }

    if (saved_probs_.size == 0) {
        cancel_aux_accumulation();
        return Tensor::zeros({1}, loss_device);
    }
    Tensor loss;
    Tensor grad_z = switch_aux_grad_logits_device(saved_probs_, top_k, coef, &loss);
    // Backprop the aux grad through the gate (accumulates the gate's weight
    // grads).  The returned input-gradient is intentionally discarded: the
    // load-balancing signal shapes the router, the standard Switch treatment.
    if (gate->weight.data.size > 0 &&
        gate->weight.data.get_device() != grad_z.get_device()) {
        grad_z = grad_z.to(gate->weight.data.get_device());
    }
    (void)gate->backward(grad_z);
    finalize_aux_accumulation(false);
    return loss;
}

Tensor MoERouter::router_grad_logits(const Tensor& probs, const Tensor& g_w,
                                     int top_k) {
    // probs,g_w: [T,N] host.  Returns dL/dlogits [T,N] from backprop through
    // renorm(top-k(softmax)): for the kept set K with S=sum_{e in K} p_e,
    //   w_e = p_e/S
    //   g_p_j = (1/S)(g_w_j - sum_{e in K} g_w_e w_e)   (j in K, else 0)
    //   g_z_i = p_i (g_p_i - sum_j p_j g_p_j)           (softmax jacobian)
    if (probs.shape.empty() || g_w.shape.empty()) {
        throw std::invalid_argument("Router VJP requires rank >= 1 tensors");
    }
    if (probs.size != g_w.size || probs.shape.back() != g_w.shape.back()) {
        throw std::invalid_argument("Router VJP probability/gradient shape mismatch");
    }
    if (probs.get_device() != g_w.get_device()) {
        throw std::invalid_argument("Router VJP tensors must share a device");
    }
    const int N = probs.shape.back();
    const int T = N > 0 ? probs.size / N : 0;
    if (T <= 0 || N <= 0) {
        return Tensor::zeros({std::max(T, 0), std::max(N, 0)},
                             probs.get_device());
    }
    const int k = std::clamp(top_k, 1, N);

#ifdef USE_CUDA
    if (probs.get_device() == Device::GPU) {
        Tensor grad = Tensor::zeros({T, N}, Device::GPU);
        launch_moe_router_logits_grad_kernel(
            probs.raw_data(), g_w.raw_data(), grad.raw_data(), T, N, k);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("MoE router VJP kernel failed: ") +
                cudaGetErrorString(launch_status));
        }
        return grad;
    }
#endif

    Tensor grad = Tensor::zeros({T, N}, Device::CPU);
    const float* p = probs.data();
    const float* gw = g_w.data();
    float* gz = grad.data();
    std::vector<int> ranked(static_cast<size_t>(N));
    std::vector<float> gp(static_cast<size_t>(N));
    for (int i = 0; i < T; ++i) {
        const size_t base = static_cast<size_t>(i) * N;
        std::iota(ranked.begin(), ranked.end(), 0);
        std::partial_sort(
            ranked.begin(), ranked.begin() + k, ranked.end(),
            [&](int a, int b) {
                return p[base + a] > p[base + b] ||
                       (p[base + a] == p[base + b] && a < b);
            });
        double S = 0.0;
        for (int r = 0; r < k; ++r) S += p[base + ranked[static_cast<size_t>(r)]];
        if (S <= 0.0) continue;
        // weighted sum_{e in K} g_w_e * w_e
        double gw_dot_w = 0.0;
        for (int r = 0; r < k; ++r) {
            const int e = ranked[static_cast<size_t>(r)];
            const double w_e = p[base + e] / S;
            gw_dot_w += gw[base + e] * w_e;
        }
        std::fill(gp.begin(), gp.end(), 0.0f);
        for (int r = 0; r < k; ++r) {
            const int j = ranked[static_cast<size_t>(r)];
            gp[static_cast<size_t>(j)] =
                static_cast<float>((gw[base + j] - gw_dot_w) / S);
        }
        // softmax backward: g_z_i = p_i (g_p_i - sum_j p_j g_p_j)
        double p_dot_gp = 0.0;
        for (int j = 0; j < N; ++j) p_dot_gp += p[base + j] * gp[static_cast<size_t>(j)];
        for (int idx = 0; idx < N; ++idx) {
            gz[base + idx] = static_cast<float>(
                p[base + idx] * (gp[static_cast<size_t>(idx)] - p_dot_gp));
        }
    }
    return grad;
}

Tensor MoERouter::accumulate_task_router_grad(const Tensor& g_w) {
    if (!gate || saved_probs_.size == 0 || g_w.size == 0) {
        return Tensor();
    }
    // The ordered deterministic expert path builds its small routing VJP on
    // the host; the router probabilities still belong to the GPU forward.
    // Move only that VJP to the cache's device, never the cached probabilities
    // or expert activations to host to satisfy the device contract.
    Tensor routed_grad = g_w.get_device() == saved_probs_.get_device()
                             ? g_w
                             : g_w.to(saved_probs_.get_device());
    Tensor g_logits = router_grad_logits(saved_probs_, routed_grad, top_k);
    if (gate->weight.data.size > 0 &&
        gate->weight.data.get_device() != g_logits.get_device()) {
        g_logits = g_logits.to(gate->weight.data.get_device());
    }
    // Accumulate the gate's weight gradients and return its input VJP for the
    // block to add to the expert branch (the routing gradient is not dropped).
    return gate->backward(g_logits);
}

std::vector<int> D2FDecoder::generate(
    const std::vector<int>& prompt,
    int length,
    Context* ctx,
    float temperature,
    float top_p,
    int   top_k,
    int   eos_token_id)
{
    std::vector<int> output = prompt;
    const bool use_streaming =
        model != nullptr && model->supports_streaming_inference() && !prompt.empty();
    const bool restore_training_mode = model != nullptr && model->training_mode();

    // RNG local, seed por tempo para não repetir;
    const uint64_t rng_sequence =
        hash_token_sequence(prompt) ^
        (static_cast<uint64_t>(std::max(length, 0)) << 32) ^
        static_cast<uint64_t>(std::max(eos_token_id, 0));
    auto seeded_rng =
        determinism::DeterminismManager::instance().get_rng_for_operation(
            "jamba_decoder", "generate", rng_sequence);
    std::mt19937 rng(static_cast<uint32_t>(seeded_rng()));

    try {
        if (model != nullptr) {
            model->set_training_mode(false);
        }
        if (use_streaming) {
            model->reset_session();
            model->set_streaming_inference(true);
        }

        Tensor logits;
        if (!output.empty()) {
            logits = model->forward_ids(output, ctx);
        }

        for (int i = 0; i < length; ++i) {
            // Abort check
            if (ctx && ctx->abort_signal &&
                ctx->abort_signal->load(std::memory_order_relaxed)) {
                throw AbortException();
            }

            // 1. Forward pass: obtém logits sobre vocabulário inteiro
            if (i > 0 || output.empty()) {
                const std::vector<int> step_tokens =
                    (use_streaming && !output.empty()) ? std::vector<int>{output.back()} : output;
                logits = model->forward_ids(step_tokens, ctx);
            }
            if (logits.size == 0) break;

            // logits shape: [seq_len, vocab_size] → pegar último token
            Tensor host_logits = logits.get_device() == Device::GPU ? logits.cpu() : logits;
            const int vocab_size = host_logits.shape.back();
            const int last_offset = host_logits.size - vocab_size;
            const float* raw = host_logits.data() + last_offset;

            // 2. Aplicar temperatura (escala os logits)
            const float temp = std::max(temperature, 1e-6f);
            std::vector<float> scaled(vocab_size);
            for (int v = 0; v < vocab_size; ++v)
                scaled[v] = raw[v] / temp;

            // 3. Top-k filtering: manter apenas os k maiores logits
            if (top_k > 0 && top_k < vocab_size) {
                // Encontrar o k-ésimo maior valor
                std::vector<float> sorted_vals = scaled;
                std::nth_element(sorted_vals.begin(),
                                 sorted_vals.begin() + (vocab_size - top_k),
                                 sorted_vals.end());
                float cutoff = sorted_vals[vocab_size - top_k];
                for (int v = 0; v < vocab_size; ++v)
                    if (scaled[v] < cutoff) scaled[v] = -1e9f;
            }

            // 4. Converter para probabilidades (softmax numericamente estável)
            float max_val = *std::max_element(scaled.begin(), scaled.end());
            std::vector<float> probs(vocab_size);
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; ++v) {
                probs[v] = std::exp(scaled[v] - max_val);
                sum_exp += probs[v];
            }
            float inv = 1.0f / std::max(sum_exp, 1e-9f);
            for (int v = 0; v < vocab_size; ++v) probs[v] *= inv;

            // 5. Top-p (nucleus) filtering: manter tokens até p acumulado
            if (top_p < 1.0f && top_p > 0.0f) {
                // Ordenar por probabilidade decrescente
                std::vector<int> indices(vocab_size);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(),
                          [&](int a, int b){ return probs[a] > probs[b]; });
                float cumulative = 0.0f;
                for (int idx : indices) {
                    if (cumulative >= top_p) probs[idx] = 0.0f;
                    else cumulative += probs[idx];
                }
                // Renormalizar após filtro
                float new_sum = 0.0f;
                for (float p : probs) new_sum += p;
                inv = 1.0f / std::max(new_sum, 1e-9f);
                for (float& p : probs) p *= inv;
            }

            // 6. Sample categórico
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next_token = dist(rng);
            output.push_back(next_token);

            // 7. Critério de parada: EOS token
            if (next_token == eos_token_id) break;
        }
    } catch (const AbortException& e) {
        std::cerr << "[NSOS:Decoder] " << e.what() << "\n";
        // Retorna tokens gerados até o momento do cancelamento
    }
    if (use_streaming) {
        model->set_streaming_inference(false);
    }
    if (model != nullptr && restore_training_mode) {
        model->set_training_mode(true);
    }
    return output;
}

} // namespace nsos
