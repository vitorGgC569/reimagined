#pragma once
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include "nsos_config.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

// Mixed-precision GEMM control. FP16 Tensor Cores require sm_70+; native BF16
// Tensor Cores require sm_80+ (Ampere). Requests are validated against the
// active device before GEMM; unsupported modes fail loudly.
// 0 = FP32 (default, bit-parity with CPU), 1 = BF16, 2 = FP16.  Master weights
// and optimizer state remain FP32; only GEMM inputs are cast.  Default-OFF;
// initialized from NSOS_MIXED_PRECISION env for back-compat.  See tensor.cpp.
void set_matmul_precision_mode(int mode);
int matmul_precision_mode();
// When enabled, any Tensor operation that would execute a host fallback for a
// GPU tensor throws instead of migrating Unified Memory silently.
void set_strict_gpu_execution(bool enabled);
bool strict_gpu_execution();

// (auditoria #8) Cópia de bytes UNIFICADA entre TUs — implementação única em
// tensor.cpp: D2D opt-in async no stream 0; H2D/D2H síncronos (lifetime do
// buffer host).  trainer.cpp e jamba.cpp tinham cópias locais com semânticas
// divergentes (uma 100% síncrona).
void copy_tensor_bytes(float* dst, Device dst_device, const float* src,
                       Device src_device, size_t bytes, bool async_d2d = false);

// mean(|x|) — o estatístico "absmean" do BitNet b1.58 (regra canônica do
// weight-scale ternário).  Kernel GPU quando o tensor é device-resident
// (uma D2H de 4 bytes); loop host caso contrário.
class Tensor;
float tensor_abs_mean(const Tensor& t);

// C = A · Bᵀ com B [n, k] rank-2 SEM materializar a transposta.  No GPU usa
// cublas OP_T direto (a leitura integral mostrou que TODO forward de BitLinear
// fazia weight.transpose() — uma cópia completa da matriz por camada por token
// — só para alimentar um matmul que o cuBLAS resolveria com um flag).  CPU e
// modo de precisão mista mantêm o caminho antigo (transpose + matmul) — o CPU
// está bom como está e o BF16 é opt-in minoritário.
Tensor matmul_nt(const Tensor& a, const Tensor& b_rowmajor);

// (auditoria #25/#26) Observabilidade e controle do pool GPU.
struct PoolStats {
    size_t cached_bytes = 0;   // soma das free-lists
    size_t live_bytes = 0;     // blocos atualmente possuídos por Tensors
    size_t bins = 0;           // nº de size-classes em cache
};
PoolStats pool_stats();
void release_cached_memory();  // devolve TODO o cache ao driver (trim)
// CUDA-graph capture guard: begin makes the pool
// capture-safe (no cudaFree/trim/memGetInfo; capture-time buffers are
// quarantined on free); end stops tracking new allocations but keeps the
// quarantine; release (call ONLY after the captured graph is destroyed)
// returns the quarantined buffers to the driver.
void gpu_pool_begin_capture();
void gpu_pool_end_capture();
void gpu_pool_release_capture();

struct TensorShape {
    std::vector<int> dims;
    std::vector<size_t> strides;

    TensorShape() = default;
    TensorShape(const std::vector<int>& d) : dims(d) {
        compute_strides();
    }
    
    void compute_strides() {
        if (dims.empty()) {
            strides.clear();
            return;
        }
        strides.resize(dims.size());
        size_t s = 1;
        for (int i = (int)dims.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= dims[i];
        }
    }

    size_t numel() const {
        if (dims.empty()) return 1;
        size_t n = 1;
        for (int d : dims) n *= d;
        return n;
    }

    size_t size() const { return dims.size(); }
    int operator[](int i) const { return dims[i]; }
    int back() const { return dims.empty() ? 0 : dims.back(); }
    bool empty() const { return dims.empty(); }
    
    std::vector<int>::const_iterator begin() const { return dims.begin(); }
    std::vector<int>::const_iterator end() const { return dims.end(); }
    bool operator==(const TensorShape& other) const { return dims == other.dims; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }
};

struct TensorDeleter {
    Device device;
    TensorDeleter(Device dev) : device(dev) {}
    void operator()(float* ptr);
};

class Tensor {
public:
    std::shared_ptr<float> data_ptr;
    TensorShape shape;
    // Element count.  int64_t (not int) so the type itself never silently
    // truncates numel() (size_t).  Construction is still bounded by
    // checked_tensor_size (rejects numel > INT_MAX) so existing index
    // arithmetic stays valid and no behavior changes; the wider type is the
    // forward-compatible foundation for lifting that cap later.
    int64_t size;
    Device device;
    std::shared_ptr<Tensor> grad;

    Tensor();
    Tensor(std::vector<int> s, Device dev = Device::CPU, float fill_value = 0.0f);
    
    // Host pointer access. GPU storage provenance is tracked per Tensor:
    // cudaMalloc-backed storage is rejected, while CPU and CUDA managed
    // storage are synchronized before a host pointer is returned.
    //
    // For host-accessible CUDA managed storage this is also the dynamic
    // safety net for Pascal+Windows: data() always establishes the CUDA
    // synchronization boundary before returning. Kernel-launch hot paths
    // must use raw_data() and never dereference that pointer on the host.
    //
    float* data();
    const float* data() const;

    // Returns the raw pointer without any implicit sync.  Use ONLY
    // when passing into a CUDA kernel launch (which serializes with
    // prior launches via the default stream and does not need host-
    // side synchronization).
    float* raw_data() { return data_ptr.get(); }
    const float* raw_data() const { return data_ptr.get(); }

    // Explicit synchronization barrier. Idempotent and cheap when the GPU
    // has no pending work.
    void sync_host_access() const;

    // Legacy diagnostics predicate retained for source compatibility.
    // Host-accessible GPU storage now always synchronizes, so this is true.
    static bool eager_gpu_sync_enabled();

    Tensor to(Device dev) const;
    Tensor cpu() const;
    Tensor clone() const;
    void copy_from(const Tensor& other);
    
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor transpose() const;
    
    Tensor relu() const;
    // LEARN S1 (BitNet b1.58 2B4T): Squared ReLU activation.
    //   forward:  y = max(0, x)^2
    //   backward: dx = dy * 2 * max(0, x)
    // Used as the FFN/MoE activation because SwiGLU under low-precision
    // (ternary BitLinear / FP8) can spike and overflow the dynamic
    // range, causing loss divergence after extended training.  Squared
    // ReLU is numerically stable in quantized regimes and gives
    // comparable expressive power to SwiGLU at moderate parameter
    // counts (1-2B params, validated by BitNet b1.58 2B4T technical
    // report 2026).
    Tensor squared_relu() const;
    // Backward for squared_relu.  Returns dx = dy * 2 * max(0, pre).
    // pre_activation is the value BEFORE squared_relu was applied
    // (typically saved during forward).  dy is the upstream gradient.
    static Tensor squared_relu_backward(const Tensor& dy,
                                         const Tensor& pre_activation);
    Tensor sigmoid() const;
    Tensor softmax(int dim = -1) const;
    Tensor rmsnorm(float eps = 1e-6f) const;
    Tensor clamp(float min, float max) const;
    
    Tensor sum(int dim = -1, bool keepdim = false) const;
    float norm() const;
    
    Tensor rmsnorm_backward(const Tensor& grad, const Tensor& x_norm,
                            float eps = 1e-6f) const;
    std::pair<float, Tensor> cross_entropy(const std::vector<int>& target) const;
    // Mean weighted negative log-likelihood:
    //   L = (1 / rows) * sum_i row_weights[i] * CE(logits_i, target_i).
    // The returned gradient is the exact derivative of that scalar.  Keeping
    // the denominator equal to rows preserves the historical meaning of the
    // first-token/EOS multipliers used by Trainer.
    std::pair<float, Tensor> cross_entropy_weighted(
        const std::vector<int>& target,
        const std::vector<float>& row_weights) const;
    std::pair<float, Tensor> mse_loss(const Tensor& target) const;
    
    Tensor reshape(std::vector<int> new_shape) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor slice(int dim, int start, int end) const;
    int get_flat_index(const std::vector<int>& indices) const;
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    float get(const std::vector<int>& indices) const;
    
    void print(const std::string& name = "", int max_elements = 10) const;
    
    // Static Factories with Overloads for Vector support
    static Tensor from_blob(void* ptr, std::vector<int> s, Device dev, bool take_ownership=false);
    static Tensor zeros(const std::vector<int>& s, Device dev = Device::CPU) { return Tensor(s, dev, 0.0f); }
    static Tensor zeros(const TensorShape& s, Device dev = Device::CPU) { return Tensor(s.dims, dev, 0.0f); }
    // Aloca SEM zero-fill — exclusivamente para produtores que comprovadamente
    // sobrescrevem 100% do buffer (GEMM beta=0, cópias completas, kernels que
    // escrevem todo elemento).  O zero-fill incondicional custava um kernel de
    // memset por alocação (~centenas por step de treino).
    static Tensor uninitialized(const std::vector<int>& s, Device dev = Device::CPU);

    static Tensor ones(const std::vector<int>& s, Device dev = Device::CPU) { return Tensor(s, dev, 1.0f); }
    static Tensor ones(const TensorShape& s, Device dev = Device::CPU) { return Tensor(s.dims, dev, 1.0f); }
    static Tensor random(const std::vector<int>& s, Device dev = Device::CPU);
    static Tensor uniform(const std::vector<int>& s, float low, float high,
                          Device dev = Device::CPU);
    static Tensor kaiming_uniform(const std::vector<int>& s, Device dev = Device::CPU);
    // Seeded variant — uses a LOCAL mt19937 (does NOT touch tensor_rng global
    // state).  Required for determinism between instances when constructing
    // layers in sequence (e.g., TTTLayer's 3 BitLinears, or A/B probes that
    // need two models to produce identical initializations).  Seed=0 falls
    // back to the un-seeded variant for backwards compatibility.
    static Tensor kaiming_uniform(const std::vector<int>& s, Device dev,
                                  uint64_t seed);
    static Tensor xavier_uniform(const std::vector<int>& s, Device dev = Device::CPU);
    static Tensor from_scalar(float val, Device dev = Device::CPU);
    static Tensor eye(int n, Device dev = Device::CPU);
    
    static void clip_grad_norm_(std::vector<Tensor>& params, float max_norm);

    uintptr_t data_ptr_int() const { return reinterpret_cast<uintptr_t>(data()); }
    Device get_device() const { return device; }
    bool is_host_accessible() const noexcept {
        return host_accessible_storage_;
    }

    void zero_grad() {
        if (!grad) return;
#ifdef USE_CUDA
        if (grad->get_device() == Device::GPU) {
            const cudaError_t memset_status =
                cudaMemset(grad->raw_data(), 0,
                           static_cast<size_t>(grad->size) * sizeof(float));
            if (memset_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA gradient zero failed: ") +
                    cudaGetErrorString(memset_status));
            }
            const cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA gradient zero sync failed: ") +
                    cudaGetErrorString(sync_status));
            }
            return;
        }
#endif
        std::fill_n(grad->data(), grad->size, 0.0f);
    }
    void add_grad(const Tensor& g) {
        if (!grad) grad = std::make_shared<Tensor>(Tensor::zeros(shape.dims, device));
        // In-place accumulation (no per-call allocation) on CPU; GPU falls back
        // to the tensor add path which handles device memory.
        if (grad->get_device() == Device::CPU && g.get_device() == Device::CPU &&
            grad->size == g.size) {
            float* gp = grad->data();
            const float* sp = g.data();
            const int n = grad->size;
            for (int i = 0; i < n; ++i) gp[i] += sp[i];
        } else {
            Tensor new_grad = grad->add(g);
            grad->copy_from(new_grad);
        }
    }

private:
    // This is storage provenance, not a process-wide CUDA mode. Views and
    // ordinary Tensor copies preserve it automatically; constructors and
    // from_blob establish it from the allocation that is actually owned.
    bool host_accessible_storage_ = true;
};

} // namespace nsos
