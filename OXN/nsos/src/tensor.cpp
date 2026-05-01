#include "../include/tensor.h"
#include "../include/nsos_arena.h"
#include "../include/nsos/determinism.h"
#include "../include/tensor_iterator.h"
#include "../include/cuda/gpu_utils.h"
#include "../include/cuda/kernels.cuh"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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
    const size_t numel = shape.numel();
    if (numel > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Tensor element count exceeds NSOS v1 int storage limit");
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
        auto op_rng = manager.get_rng_for_operation(
            "tensor", "random", static_cast<uint64_t>(
                std::hash<std::thread::id>{}(std::this_thread::get_id())));
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
void copy_tensor_bytes(float* dst, Device dst_device, const float* src, Device src_device,
                       size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (dst_device == Device::GPU || src_device == Device::GPU) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy failed: ") +
                                     cudaGetErrorString(status));
        }
        return;
    }
    std::memcpy(dst, src, bytes);
}
#else
void copy_tensor_bytes(float* dst, Device, const float* src, Device, size_t bytes) {
    if (bytes > 0) {
        std::memcpy(dst, src, bytes);
    }
}
#endif

} // namespace

void TensorDeleter::operator()(float* ptr) {
    if (!ptr) return;
    if (device == Device::GPU) {
#ifdef USE_CUDA
        cudaFree(ptr);
#endif
    } else {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

Tensor::Tensor() : size(0), device(Device::CPU) {
    shape = TensorShape({});
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
        cudaMallocManaged(&raw_ptr, size * sizeof(float));
#endif
    } else {
#ifdef _WIN32
        raw_ptr = static_cast<float*>(_aligned_malloc(size * sizeof(float), 64));
#else
        posix_memalign((void**)&raw_ptr, 64, size * sizeof(float));
#endif
    }

    if (!raw_ptr) {
        throw std::runtime_error("Tensor allocation failed");
    }

    if (fill_value == 0.0f) {
        if (device == Device::GPU) {
#ifdef USE_CUDA
            cudaMemset(raw_ptr, 0, size * sizeof(float));
            cudaDeviceSynchronize();
#endif
        } else {
            std::memset(raw_ptr, 0, size * sizeof(float));
        }
    } else {
        if (device == Device::GPU) {
#ifdef USE_CUDA
            std::vector<float> host_values(static_cast<size_t>(size), fill_value);
            cudaMemcpy(raw_ptr,
                       host_values.data(),
                       static_cast<size_t>(size) * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
#endif
        } else {
            std::fill_n(raw_ptr, size, fill_value);
        }
    }

    data_ptr = std::shared_ptr<float>(raw_ptr, TensorDeleter(device));
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
            launch_add_kernel(result.data(), data(), other.data(), size);
            sync_cuda();
            return result;
        }
        if (!shape.empty() && other.shape.size() == 1 &&
            other.shape.back() == shape.back() && size == result.size) {
            launch_add_broadcast_kernel(result.data(), data(), other.data(), size,
                                        shape.back());
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
        launch_sub_kernel(result.data(), data(), other.data(), size);
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
            launch_mul_tensor_kernel(result.data(), data(), other.data(), size);
            sync_cuda();
            return result;
        }
        if (!shape.empty() && other.shape.size() == 1 &&
            other.shape.back() == shape.back() && size == result.size) {
            const int cols = shape.back();
            const int rows = size / std::max(cols, 1);
            launch_mul_vector_broadcast_kernel(result.data(), data(), other.data(),
                                               rows, cols);
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
    const float* src = data();
    float* dst = result.data();
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_mul_scalar_kernel(dst, src, scalar, size);
        sync_cuda();
        return result;
    }
#endif
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i] * scalar;
    }
    return result;
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

    Tensor result(out_dims, device);
    int batch = size / (m * k);
    int other_batch = other.size / (k * n);
    if (other_batch != 1 && other_batch != batch) {
        throw std::runtime_error("Unsupported batched matmul shape mismatch");
    }

    const float* a_ptr = data();
    const float* b_ptr = other.data();
    float* out_ptr = result.data();

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, other) && gpu_blas_supported()) {
        cublasHandle_t handle = cublas_handle();
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const float* a_batch = a_ptr + batch_idx * m * k;
            const float* b_batch = b_ptr + (other_batch == 1 ? 0 : batch_idx * k * n);
            float* out_batch = out_ptr + batch_idx * m * n;
            cublas_check(
                cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            b_batch,
                            n,
                            a_batch,
                            k,
                            &beta,
                            out_batch,
                            n),
                "cublasSgemm");
        }
        sync_cuda();
        return result;
    }
#endif

#pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float* a_batch = a_ptr + batch_idx * m * k;
        const float* b_batch = b_ptr + (other_batch == 1 ? 0 : batch_idx * k * n);
        float* out_batch = out_ptr + batch_idx * m * n;

        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    acc += a_batch[row * k + kk] * b_batch[kk * n + col];
                }
                out_batch[row * n + col] = acc;
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

    std::vector<int> idx = make_indices(rank);
    for (int flat = 0; flat < result.size; ++flat) {
        int remaining = flat;
        for (int d = rank - 1; d >= 0; --d) {
            idx[d] = remaining % out_dims[d];
            remaining /= out_dims[d];
        }

        std::vector<int> src_idx = idx;
        std::swap(src_idx[dim0], src_idx[dim1]);
        result.data()[flat] = get(src_idx);
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
    const float* src = data();
    float* dst = result.data();
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_relu_kernel(dst, src, size);
        sync_cuda();
        return result;
    }
#endif
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        dst[i] = std::max(src[i], 0.0f);
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape.dims, device);
    const float* src = data();
    float* dst = result.data();
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_sigmoid_kernel(dst, src, size);
        sync_cuda();
        return result;
    }
#endif
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
    const float* src = data();
    float* dst = result.data();

#ifdef USE_CUDA
    if (dim == rank - 1 && use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_softmax_kernel(dst, src, outer, axis);
        sync_cuda();
        return result;
    }
#endif

#pragma omp parallel for collapse(2)
    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int inner_idx = 0; inner_idx < inner; ++inner_idx) {
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
    }

    return result;
}

Tensor Tensor::rmsnorm(float eps) const {
    Tensor result(shape.dims, device);
    int inner = shape.back();
    int outer = size / inner;
    const float* src = data();
    float* dst = result.data();

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_rmsnorm_kernel(dst, src, outer, inner, 0, 0);
        sync_cuda();
        return result;
    }
#endif

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
    const float* src = data();
    float* dst = result.data();
#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        launch_clamp_kernel(dst, src, min_val, max_val, size);
        sync_cuda();
        return result;
    }
#endif
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
            launch_mean_kernel(result.data(), data(), 1, rows, cols);
            launch_scale_inplace_kernel(result.data(), static_cast<float>(rows),
                                        cols);
            sync_cuda();
            return result;
        }
        if (dim == 1) {
            launch_mean_kernel(result.data(), data(), rows, cols, 1);
            launch_scale_inplace_kernel(result.data(), static_cast<float>(cols),
                                        rows);
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
        launch_norm_kernel(d_sum_sq.get(), data(), size);
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
    Tensor result(shape.dims, device);
    if (size > 0) {
        copy_tensor_bytes(result.data(), result.device, data(), device, size * sizeof(float));
    }
    return result;
}

Tensor Tensor::to(Device dev) const {
    if (device == dev) {
        return *this;
    }

    Tensor result(shape.dims, dev);
    if (size > 0) {
        copy_tensor_bytes(result.data(), result.device, data(), device, size * sizeof(float));
    }
    return result;
}

Tensor Tensor::cpu() const {
    return to(Device::CPU);
}

void Tensor::copy_from(const Tensor& other) {
    if (size != other.size) {
        throw std::runtime_error("Tensor size mismatch in copy_from");
    }
    if (shape != other.shape) {
        throw std::runtime_error("Tensor shape mismatch in copy_from");
    }
    if (size > 0) {
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

    Tensor reshaped = *this;
    TensorShape new_tensor_shape(new_shape);
    if (static_cast<int>(new_tensor_shape.numel()) != size) {
        throw std::runtime_error("Reshape size mismatch");
    }
    reshaped.shape = new_tensor_shape;
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
    Tensor result(out_dims, device);

    if (dim == 0 && rank >= 1) {
        const int inner = size / shape[0];
        const float* src_ptr = data() + static_cast<size_t>(start * inner);
        const size_t bytes =
            static_cast<size_t>((end - start) * inner) * sizeof(float);
        copy_tensor_bytes(result.data(), result.device, src_ptr, device, bytes);
        return result;
    }

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
    const float* g = grad.data();
    const float* y = x_norm.data();
    float* dx_ptr = dx.data();

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this, grad) && x_norm.get_device() == Device::GPU &&
        gpu_custom_kernels_supported()) {
        launch_rmsnorm_backward_kernel(dx_ptr, g, y, outer, inner);
        sync_cuda();
        return dx;
    }
#endif

#pragma omp parallel for
    for (int i = 0; i < outer; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < inner; ++j) {
            dot += g[i * inner + j] * y[i * inner + j];
        }
        dot /= std::max(inner, 1);
        for (int j = 0; j < inner; ++j) {
            // Projeção ortogonal removendo a variância (Jacobiano Aproximado RMS)
            dx_ptr[i * inner + j] = g[i * inner + j] - y[i * inner + j] * dot;
        }
    }
    return dx;
}

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
    const float* logits = data();
    float* grad_ptr = grad.data();
    float loss = 0.0f;

#ifdef USE_CUDA
    if (use_gpu_fast_path(*this) && gpu_custom_kernels_supported()) {
        CudaBuffer<float> d_loss(1);
        CudaBuffer<int> d_target(static_cast<size_t>(rows));
        cudaMemset(d_loss.get(), 0, sizeof(float));
        cudaMemcpy(d_target.get(), target.data(), static_cast<size_t>(rows) * sizeof(int),
                   cudaMemcpyHostToDevice);
        launch_fused_cross_entropy(d_loss.get(), grad_ptr, logits, d_target.get(), rows, classes);
        sync_cuda();
        const float inv_rows = 1.0f / std::max(rows, 1);
        launch_scale_inplace_kernel(grad_ptr, inv_rows, grad.size);
        sync_cuda();
        loss = copy_scalar_from_device(d_loss.get()) * inv_rows;
        return {loss, grad};
    }
#endif

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

    float inv_size = 1.0f / std::max(size, 1);
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
    int limit = std::min(size, max_elements);
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
