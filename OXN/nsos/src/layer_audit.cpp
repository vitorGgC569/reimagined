#include "../include/layer_audit.h"
#include "../include/autograd.h"
#include "../include/nsos/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include "../include/cuda/kernels.cuh"
#include "../include/gpu_backend.h"
#endif

namespace nsos {

namespace {

std::filesystem::path unique_audit_temporary_path(
    const std::filesystem::path& destination) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t timestamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t ordinal =
        sequence.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    const uint64_t process_id = static_cast<uint64_t>(GetCurrentProcessId());
#else
    const uint64_t process_id = static_cast<uint64_t>(::getpid());
#endif
    return destination.parent_path() /
           (destination.filename().string() + ".tmp." +
            std::to_string(process_id) + "." +
            std::to_string(timestamp) + "." +
           std::to_string(ordinal));
}

class TemporaryAuditFileGuard {
public:
    explicit TemporaryAuditFileGuard(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~TemporaryAuditFileGuard() {
        if (active_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }
    TemporaryAuditFileGuard(const TemporaryAuditFileGuard&) = delete;
    TemporaryAuditFileGuard& operator=(const TemporaryAuditFileGuard&) =
        delete;
    void release() noexcept { active_ = false; }

private:
    std::filesystem::path path_;
    bool active_ = true;
};

void sync_audit_file_contents(const std::filesystem::path& path) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "Could not open audit temporary file for durable flush "
            "(Win32 error " +
            std::to_string(GetLastError()) + ")");
    }
    if (!FlushFileBuffers(handle)) {
        const DWORD error = GetLastError();
        CloseHandle(handle);
        throw std::runtime_error(
            "Could not durably flush audit temporary file (Win32 error " +
            std::to_string(error) + ")");
    }
    if (!CloseHandle(handle)) {
        throw std::runtime_error(
            "Could not close audit temporary file handle (Win32 error " +
            std::to_string(GetLastError()) + ")");
    }
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Could not open audit temporary file for durable flush: " +
            std::string(std::strerror(errno)));
    }
    if (::fsync(descriptor) != 0) {
        const int error = errno;
        ::close(descriptor);
        throw std::runtime_error(
            "Could not durably flush audit temporary file: " +
            std::string(std::strerror(error)));
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(
            "Could not close audit temporary file descriptor: " +
            std::string(std::strerror(errno)));
    }
#endif
}

void replace_audit_file(const std::filesystem::path& temporary,
                        const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error(
            "Could not atomically publish audit JSON '" +
            destination.string() + "' (Win32 error " +
            std::to_string(error) + ")");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        throw std::runtime_error(
            "Could not atomically publish audit JSON '" +
            destination.string() + "': " + error.message());
    }
    const std::filesystem::path directory =
        destination.has_parent_path()
            ? destination.parent_path()
            : std::filesystem::path(".");
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
    const int descriptor = ::open(directory.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Could not open audit output directory for durable flush: " +
            std::string(std::strerror(errno)));
    }
    if (::fsync(descriptor) != 0) {
        const int error_number = errno;
        ::close(descriptor);
        throw std::runtime_error(
            "Could not durably flush audit output directory: " +
            std::string(std::strerror(error_number)));
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(
            "Could not close audit output directory descriptor: " +
            std::string(std::strerror(errno)));
    }
#endif
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char ch : input) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

template <typename T>
void write_json_number(std::ostream& out, T value) {
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(value)) {
            // RFC 8259 has no NaN or Infinity literals. Counts and finite
            // flags retain the diagnostic fact while null keeps the artifact
            // parseable by strict JSON consumers.
            out << "null";
            return;
        }
    }
    out << value;
}

template <typename T>
void write_numeric_array(std::ostream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        write_json_number(out, values[i]);
    }
    out << "]";
}

void write_tensor_stats(std::ostream& out, const TensorAuditStats& stats) {
    out << "{";
    out << "\"shape\":";
    write_numeric_array(out, stats.shape);
    out << ",\"elements\":" << stats.elements;
    out << ",\"min\":";
    write_json_number(out, stats.min);
    out << ",\"max\":";
    write_json_number(out, stats.max);
    out << ",\"mean\":";
    write_json_number(out, stats.mean);
    out << ",\"stddev\":";
    write_json_number(out, stats.stddev);
    out << ",\"l2_norm\":";
    write_json_number(out, stats.l2_norm);
    out << ",\"max_abs\":";
    write_json_number(out, stats.max_abs);
    out << ",\"nan_count\":" << stats.nan_count;
    out << ",\"inf_count\":" << stats.inf_count;
    out << ",\"zero_count\":" << stats.zero_count;
    out << ",\"subnormal_count\":" << stats.subnormal_count;
    out << ",\"positive_count\":" << stats.positive_count;
    out << ",\"negative_count\":" << stats.negative_count;
    out << ",\"finite\":" << (stats.finite ? "true" : "false");
    out << "}";
}

void write_layer_summary(std::ostream& out, const LayerAuditSummary& summary) {
    out << "{";
    out << "\"phase\":\"" << json_escape(summary.phase) << "\"";
    out << ",\"records\":" << summary.records;
    out << ",\"forward_records\":" << summary.forward_records;
    out << ",\"backward_records\":" << summary.backward_records;
    out << ",\"router_records\":" << summary.router_records;
    out << ",\"token_contexts\":" << summary.token_contexts;
    out << ",\"training_steps\":" << summary.training_steps;
    out << ",\"parameter_records\":" << summary.parameter_records;
    out << ",\"changed_parameter_records\":"
        << summary.changed_parameter_records;
    out << ",\"hybrid_interaction_records\":"
        << summary.hybrid_interaction_records;
    out << ",\"parameter_nan\":" << summary.parameter_nan;
    out << ",\"parameter_inf\":" << summary.parameter_inf;
    out << ",\"total_nan\":" << summary.total_nan;
    out << ",\"total_inf\":" << summary.total_inf;
    out << ",\"max_latency_ms\":";
    write_json_number(out, summary.max_latency_ms);
    out << ",\"max_l2_norm\":";
    write_json_number(out, summary.max_l2_norm);
    out << ",\"layers_seen\":";
    write_numeric_array(out, summary.layers_seen);
    out << ",\"stored_records\":" << summary.stored_records;
    out << ",\"dropped_records\":" << summary.dropped_records;
    out << ",\"truncated_contexts\":" << summary.truncated_contexts;
    out << ",\"router_entropy_count\":" << summary.router_entropy_count;
    out << ",\"router_entropy_min\":";
    write_json_number(out, summary.router_entropy_min);
    out << ",\"router_entropy_max\":";
    write_json_number(out, summary.router_entropy_max);
    out << ",\"router_entropy_mean\":";
    write_json_number(out, summary.router_entropy_mean);
    out << ",\"router_num_experts_max\":" << summary.router_num_experts_max;
    out << ",\"hybrid_signal_cosine_mean\":";
    write_json_number(out, summary.hybrid_signal_cosine_mean);
    out << ",\"hybrid_contribution_cosine_mean\":";
    write_json_number(out, summary.hybrid_contribution_cosine_mean);
    out << ",\"hybrid_attention_to_mamba_mean\":";
    write_json_number(out, summary.hybrid_attention_to_mamba_mean);
    out << ",\"hybrid_mamba_ffn_contribution_cosine_mean\":";
    write_json_number(
        out, summary.hybrid_mamba_ffn_contribution_cosine_mean);
    out << ",\"hybrid_attention_ffn_contribution_cosine_mean\":";
    write_json_number(
        out, summary.hybrid_attention_ffn_contribution_cosine_mean);
    out << ",\"hybrid_ffn_to_mamba_mean\":";
    write_json_number(out, summary.hybrid_ffn_to_mamba_mean);
    out << ",\"hybrid_cancellation_max\":";
    write_json_number(out, summary.hybrid_cancellation_max);
    out << ",\"hybrid_forward_interaction_records\":"
        << summary.hybrid_forward_interaction_records;
    out << ",\"hybrid_forward_signal_cosine_mean\":";
    write_json_number(out, summary.hybrid_forward_signal_cosine_mean);
    out << ",\"hybrid_forward_contribution_cosine_mean\":";
    write_json_number(out, summary.hybrid_forward_contribution_cosine_mean);
    out << ",\"hybrid_forward_attention_to_mamba_mean\":";
    write_json_number(out, summary.hybrid_forward_attention_to_mamba_mean);
    out << ",\"hybrid_forward_cancellation_max\":";
    write_json_number(out, summary.hybrid_forward_cancellation_max);
    out << ",\"hybrid_backward_interaction_records\":"
        << summary.hybrid_backward_interaction_records;
    out << ",\"hybrid_backward_signal_cosine_mean\":";
    write_json_number(out, summary.hybrid_backward_signal_cosine_mean);
    out << ",\"hybrid_backward_contribution_cosine_mean\":";
    write_json_number(out, summary.hybrid_backward_contribution_cosine_mean);
    out << ",\"hybrid_backward_attention_to_mamba_mean\":";
    write_json_number(out, summary.hybrid_backward_attention_to_mamba_mean);
    out << ",\"hybrid_backward_cancellation_max\":";
    write_json_number(out, summary.hybrid_backward_cancellation_max);
    out << "}";
}

double router_entropy(const std::vector<float>& loads,
                      const std::vector<int>& topk_counts) {
    double total = 0.0;
    for (float value : loads) {
        if (std::isfinite(value) && value > 0.0f) {
            total += static_cast<double>(value);
        }
    }
    if (total <= 0.0) {
        for (int value : topk_counts) {
            if (value > 0) {
                total += static_cast<double>(value);
            }
        }
        if (total <= 0.0) {
            return 0.0;
        }
        double entropy = 0.0;
        for (int value : topk_counts) {
            if (value <= 0) {
                continue;
            }
            const double p = static_cast<double>(value) / total;
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double entropy = 0.0;
    for (float value : loads) {
        if (!std::isfinite(value) || value <= 0.0f) {
            continue;
        }
        const double p = static_cast<double>(value) / total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

void add_layer_seen(std::set<int>& seen, int layer_index) {
    if (layer_index >= 0) {
        seen.insert(layer_index);
    }
}

void add_layer_seen(std::vector<int>& seen, int layer_index) {
    if (layer_index < 0) {
        return;
    }
    if (std::find(seen.begin(), seen.end(), layer_index) == seen.end()) {
        seen.push_back(layer_index);
        std::sort(seen.begin(), seen.end());
    }
}

std::string tensor_sha256(const Tensor& host_tensor) {
    if (host_tensor.get_device() != Device::CPU) {
        throw std::invalid_argument(
            "tensor_sha256 requires a CPU tensor snapshot");
    }
    static constexpr char kDomain[] = "NSOS-TENSOR-F32-LE-v1";
    const float* values =
        host_tensor.size > 0 ? host_tensor.data() : nullptr;
    return integrity::sha256_hex_stream(
        [&](const integrity::Sha256Sink& sink) {
            sink(kDomain, sizeof(kDomain));
            auto write_u64 = [&](uint64_t value) {
                std::array<unsigned char, 8> bytes{};
                for (int shift = 0; shift < 64; shift += 8) {
                    bytes[static_cast<size_t>(shift / 8)] =
                        static_cast<unsigned char>(
                            (value >> shift) & 0xffu);
                }
                sink(bytes.data(), bytes.size());
            };
            write_u64(static_cast<uint64_t>(
                host_tensor.shape.size()));
            for (int dimension : host_tensor.shape.dims) {
                write_u64(static_cast<uint64_t>(
                    static_cast<int64_t>(dimension)));
            }

            std::array<unsigned char, 4096> bytes{};
            size_t buffered = 0;
            for (int index = 0; index < host_tensor.size; ++index) {
                uint32_t bits = 0;
                static_assert(sizeof(bits) == sizeof(values[index]));
                std::memcpy(&bits, values + index, sizeof(bits));
                for (int shift = 0; shift < 32; shift += 8) {
                    bytes[buffered++] =
                        static_cast<unsigned char>(
                            (bits >> shift) & 0xffu);
                }
                if (buffered == bytes.size()) {
                    sink(bytes.data(), buffered);
                    buffered = 0;
                }
            }
            if (buffered > 0) {
                sink(bytes.data(), buffered);
            }
        });
}

int parameter_layer_index(const std::string& name) {
    static constexpr char kPrefix[] = "layers.";
    if (name.rfind(kPrefix, 0) != 0) {
        return -1;
    }
    const size_t begin = sizeof(kPrefix) - 1;
    const size_t end = name.find('.', begin);
    if (end == std::string::npos || end == begin) {
        return -1;
    }
    try {
        const int layer = std::stoi(name.substr(begin, end - begin));
        return layer >= 0 ? layer : -1;
    } catch (const std::exception&) {
        return -1;
    }
}

std::string parameter_component(const std::string& name) {
    if (name.empty()) {
        return "unknown";
    }
    if (name.rfind("layers.", 0) == 0) {
        const size_t layer_end = name.find('.', 7);
        if (layer_end != std::string::npos) {
            const size_t component_end = name.find('.', layer_end + 1);
            return name.substr(
                layer_end + 1,
                component_end == std::string::npos
                    ? std::string::npos
                    : component_end - layer_end - 1);
        }
    }
    const size_t separator = name.find('.');
    return name.substr(0, separator);
}

std::string parameter_role(const std::string& name) {
    const size_t separator = name.rfind('.');
    return separator == std::string::npos
               ? name
               : name.substr(separator + 1);
}

double stable_ratio(double numerator, double denominator) {
    if (!std::isfinite(numerator) || numerator < 0.0 ||
        !std::isfinite(denominator) || denominator < 0.0) {
        return 0.0;
    }
    if (numerator == 0.0) {
        return 0.0;
    }
    constexpr double kFloor = 1e-300;
    const double ratio = numerator / std::max(denominator, kFloor);
    return std::isfinite(ratio)
               ? ratio
               : std::numeric_limits<double>::max();
}

double tensor_cosine(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.get_device() != Device::CPU ||
        rhs.get_device() != Device::CPU ||
        lhs.shape != rhs.shape || lhs.size != rhs.size ||
        lhs.size <= 0) {
        return 0.0;
    }
    const float* a = lhs.data();
    const float* b = rhs.data();
    double dot = 0.0;
    double a_sq = 0.0;
    double b_sq = 0.0;
    for (int index = 0; index < lhs.size; ++index) {
        if (!std::isfinite(a[index]) || !std::isfinite(b[index])) {
            return 0.0;
        }
        const double av = static_cast<double>(a[index]);
        const double bv = static_cast<double>(b[index]);
        dot += av * bv;
        a_sq += av * av;
        b_sq += bv * bv;
    }
    if (a_sq <= 0.0 || b_sq <= 0.0) {
        return 0.0;
    }
    const double value = dot / std::sqrt(a_sq * b_sq);
    return std::isfinite(value)
               ? std::clamp(value, -1.0, 1.0)
               : 0.0;
}

#ifdef USE_CUDA
TensorAuditStats tensor_stats_from_device_reduction(
    const NsosTensorAuditDeviceStats& raw,
    const TensorShape& shape,
    int64_t elements) {
    TensorAuditStats stats;
    stats.shape = shape.dims;
    stats.elements = elements;
    stats.nan_count = static_cast<size_t>(raw.nan_count);
    stats.inf_count = static_cast<size_t>(raw.inf_count);
    stats.zero_count = static_cast<size_t>(raw.zero_count);
    stats.subnormal_count =
        static_cast<size_t>(raw.subnormal_count);
    stats.positive_count =
        static_cast<size_t>(raw.positive_count);
    stats.negative_count =
        static_cast<size_t>(raw.negative_count);
    stats.finite =
        raw.nan_count == 0 && raw.inf_count == 0;
    if (raw.finite_count == 0) {
        return stats;
    }
    stats.min = raw.min_value;
    stats.max = raw.max_value;
    stats.mean =
        raw.sum / static_cast<double>(raw.finite_count);
    const double mean_square =
        raw.sum_sq / static_cast<double>(raw.finite_count);
    stats.stddev = std::sqrt(std::max(
        0.0, mean_square - stats.mean * stats.mean));
    stats.l2_norm = std::sqrt(std::max(0.0, raw.sum_sq));
    stats.max_abs = raw.max_abs;
    return stats;
}

double cosine_from_device_reduction(
    double dot,
    const TensorAuditStats& lhs,
    const TensorAuditStats& rhs) {
    if (!lhs.finite || !rhs.finite ||
        lhs.l2_norm <= 0.0 || rhs.l2_norm <= 0.0 ||
        !std::isfinite(dot)) {
        return 0.0;
    }
    const double value =
        dot / (lhs.l2_norm * rhs.l2_norm);
    return std::isfinite(value)
               ? std::clamp(value, -1.0, 1.0)
               : 0.0;
}

NsosTensorAuditDeviceStats reduce_tensor_audit_on_device(
    const Tensor& tensor) {
    static_assert(
        std::is_trivially_copyable_v<
            NsosTensorAuditDeviceStats>);
    Tensor device_result = Tensor::uninitialized(
        {static_cast<int>(
            (sizeof(NsosTensorAuditDeviceStats) +
             sizeof(float) - 1) /
            sizeof(float))},
        Device::GPU);
    launch_tensor_audit_stats_kernel(
        reinterpret_cast<NsosTensorAuditDeviceStats*>(
            device_result.raw_data()),
        tensor.raw_data(), tensor.size);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("tensor audit reduction launch failed on ") +
            NSOS_GPU_BACKEND_NAME + ": " +
            cudaGetErrorString(launch_status));
    }
    NsosTensorAuditDeviceStats host_result{};
    copy_tensor_bytes(
        reinterpret_cast<float*>(&host_result), Device::CPU,
        device_result.raw_data(), Device::GPU,
        sizeof(host_result));
    return host_result;
}

NsosHybridAuditDeviceMetrics reduce_hybrid_audit_on_device(
    const Tensor& mamba_signal,
    const Tensor& attention_signal,
    const Tensor& ffn_signal,
    const Tensor& mamba_contribution,
    const Tensor& attention_contribution,
    const Tensor& ffn_contribution) {
    static_assert(
        std::is_trivially_copyable_v<
            NsosHybridAuditDeviceMetrics>);
    Tensor device_result = Tensor::uninitialized(
        {static_cast<int>(
            (sizeof(NsosHybridAuditDeviceMetrics) +
             sizeof(float) - 1) /
            sizeof(float))},
        Device::GPU);
    launch_hybrid_audit_metrics_kernel(
        reinterpret_cast<NsosHybridAuditDeviceMetrics*>(
            device_result.raw_data()),
        mamba_signal.raw_data(),
        attention_signal.raw_data(),
        ffn_signal.raw_data(),
        mamba_contribution.raw_data(),
        attention_contribution.raw_data(),
        ffn_contribution.raw_data(),
        mamba_signal.size);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("hybrid audit reduction launch failed on ") +
            NSOS_GPU_BACKEND_NAME + ": " +
            cudaGetErrorString(launch_status));
    }
    NsosHybridAuditDeviceMetrics host_result{};
    copy_tensor_bytes(
        reinterpret_cast<float*>(&host_result), Device::CPU,
        device_result.raw_data(), Device::GPU,
        sizeof(host_result));
    return host_result;
}
#endif

void synchronize_parameter_audit_device(bool required) {
#ifdef USE_CUDA
    if (required) {
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("parameter audit device synchronization failed on ") +
                NSOS_GPU_BACKEND_NAME + ": " +
                cudaGetErrorString(status));
        }
        record_gpu_device_synchronization();
    }
#else
    if (required) {
        throw std::runtime_error(
            "parameter audit received a GPU tensor in a CPU-only build");
    }
#endif
}

} // namespace

void LayerAuditCollector::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    if (!enabled) {
        pending_parameters_.clear();
        pending_parameter_step_ = -1;
        pending_parameter_run_id_.clear();
        pending_parameter_phase_.clear();
    }
}

bool LayerAuditCollector::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void LayerAuditCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    next_sequence_ = 1;
    phase_ = "default";
    step_ = 0;
    records_.clear();
    token_contexts_.clear();
    training_steps_.clear();
    parameter_records_.clear();
    hybrid_interaction_records_.clear();
    pending_parameters_.clear();
    pending_parameter_step_ = -1;
    pending_parameter_run_id_.clear();
    pending_parameter_phase_.clear();
    phase_summaries_.clear();
}

void LayerAuditCollector::begin_run(const std::string& run_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    run_id_ = run_id.empty() ? "default" : run_id;
    phase_ = "default";
    step_ = 0;
    next_sequence_ = 1;
    records_.clear();
    token_contexts_.clear();
    training_steps_.clear();
    parameter_records_.clear();
    hybrid_interaction_records_.clear();
    pending_parameters_.clear();
    pending_parameter_step_ = -1;
    pending_parameter_run_id_.clear();
    pending_parameter_phase_.clear();
    phase_summaries_.clear();
}

void LayerAuditCollector::set_phase(const std::string& phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = phase.empty() ? "default" : phase;
}

void LayerAuditCollector::set_step(int step) {
    std::lock_guard<std::mutex> lock(mutex_);
    step_ = step;
}

void LayerAuditCollector::set_storage_policy(bool summary_only,
                                             int record_sample_rate,
                                             size_t max_records_per_phase,
                                             bool store_token_contexts) {
    std::lock_guard<std::mutex> lock(mutex_);
    summary_only_ = summary_only;
    record_sample_rate_ = std::max(record_sample_rate, 1);
    max_records_per_phase_ = max_records_per_phase;
    store_token_contexts_ = store_token_contexts;
}

void LayerAuditCollector::set_parameter_audit_policy(
    bool enabled,
    int step_sample_rate,
    size_t max_records,
    size_t max_snapshot_bytes) {
    if (step_sample_rate <= 0) {
        throw std::invalid_argument(
            "parameter audit step_sample_rate must be positive");
    }
    if (max_snapshot_bytes == 0) {
        throw std::invalid_argument(
            "parameter audit max_snapshot_bytes must be positive");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_parameters_.empty()) {
        throw std::logic_error(
            "cannot change parameter audit policy during a pending step");
    }
    parameter_audit_enabled_ = enabled;
    parameter_step_sample_rate_ = step_sample_rate;
    max_parameter_records_ = max_records;
    max_parameter_snapshot_bytes_ = max_snapshot_bytes;
}

bool LayerAuditCollector::summary_only() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return summary_only_;
}

int LayerAuditCollector::record_sample_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return record_sample_rate_;
}

size_t LayerAuditCollector::max_records_per_phase() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_records_per_phase_;
}

bool LayerAuditCollector::store_token_contexts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return store_token_contexts_;
}

bool LayerAuditCollector::parameter_audit_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parameter_audit_enabled_;
}

int LayerAuditCollector::parameter_step_sample_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parameter_step_sample_rate_;
}

size_t LayerAuditCollector::max_parameter_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_parameter_records_;
}

size_t LayerAuditCollector::max_parameter_snapshot_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_parameter_snapshot_bytes_;
}

uint64_t LayerAuditCollector::next_sequence_unlocked() {
    return next_sequence_++;
}

LayerAuditCollector::PhaseAuditAccumulator&
LayerAuditCollector::phase_accumulator_unlocked(const std::string& phase) {
    auto [it, inserted] = phase_summaries_.try_emplace(phase);
    if (inserted) {
        it->second.summary.phase = phase;
    }
    return it->second;
}

const LayerAuditCollector::PhaseAuditAccumulator*
LayerAuditCollector::find_phase_accumulator_unlocked(const std::string& phase) const {
    const auto it = phase_summaries_.find(phase);
    if (it == phase_summaries_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool LayerAuditCollector::should_collect_layer_stats_unlocked(
    const PhaseAuditAccumulator& accumulator,
    const std::string& pass,
    int layer_index) const {
    if (pass == "router" || record_sample_rate_ <= 1) {
        return true;
    }
    if (layer_index >= 0 &&
        std::find(accumulator.summary.layers_seen.begin(),
                  accumulator.summary.layers_seen.end(),
                  layer_index) == accumulator.summary.layers_seen.end()) {
        return true;
    }
    const size_t observed = accumulator.summary.forward_records +
                            accumulator.summary.backward_records;
    return (observed % static_cast<size_t>(record_sample_rate_)) == 0;
}

bool LayerAuditCollector::should_store_layer_record_unlocked(
    const PhaseAuditAccumulator& accumulator,
    const std::string& pass) const {
    if (summary_only_) {
        return false;
    }
    if (pass == "router") {
        return max_records_per_phase_ == 0 ||
               accumulator.summary.stored_records < max_records_per_phase_;
    }
    if (max_records_per_phase_ > 0 &&
        accumulator.summary.stored_records >= max_records_per_phase_) {
        return false;
    }
    return true;
}

void LayerAuditCollector::update_layer_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const LayerAuditRecord& record,
    bool has_tensor_stats) {
    auto& summary = accumulator.summary;
    ++summary.records;
    if (record.pass == "forward") {
        ++summary.forward_records;
    } else if (record.pass == "backward") {
        ++summary.backward_records;
    } else if (record.pass == "router") {
        ++summary.router_records;
        const double entropy = record.router.entropy;
        if (summary.router_entropy_count == 0) {
            summary.router_entropy_min = entropy;
            summary.router_entropy_max = entropy;
            summary.router_entropy_mean = entropy;
        } else {
            summary.router_entropy_min = std::min(summary.router_entropy_min, entropy);
            summary.router_entropy_max = std::max(summary.router_entropy_max, entropy);
            summary.router_entropy_mean +=
                (entropy - summary.router_entropy_mean) /
                static_cast<double>(summary.router_entropy_count + 1);
        }
        ++summary.router_entropy_count;
        summary.router_num_experts_max =
            std::max(summary.router_num_experts_max, record.router.num_experts);
    }
    add_layer_seen(summary.layers_seen, record.layer_index);
    summary.max_latency_ms = std::max(summary.max_latency_ms, record.latency_ms);
    if (has_tensor_stats) {
        summary.total_nan += record.input.nan_count + record.output.nan_count;
        summary.total_inf += record.input.inf_count + record.output.inf_count;
        summary.max_l2_norm =
            std::max(summary.max_l2_norm,
                     std::max(record.input.l2_norm, record.output.l2_norm));
    }
}

void LayerAuditCollector::mark_layer_record_storage_unlocked(
    PhaseAuditAccumulator& accumulator,
    bool stored) {
    if (stored) {
        ++accumulator.summary.stored_records;
    } else {
        ++accumulator.summary.dropped_records;
    }
}

void LayerAuditCollector::update_training_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const TrainingStepAuditRecord&) {
    ++accumulator.summary.training_steps;
}

void LayerAuditCollector::update_parameter_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const ParameterAuditRecord& record) {
    auto& summary = accumulator.summary;
    ++summary.parameter_records;
    if (record.changed_elements > 0) {
        ++summary.changed_parameter_records;
    }
    summary.parameter_nan +=
        record.weight_before.nan_count + record.gradient.nan_count +
        record.update.nan_count + record.weight_after.nan_count;
    summary.parameter_inf +=
        record.weight_before.inf_count + record.gradient.inf_count +
        record.update.inf_count + record.weight_after.inf_count;
    add_layer_seen(summary.layers_seen, record.layer_index);
    summary.max_l2_norm =
        std::max({summary.max_l2_norm,
                  record.weight_before.l2_norm,
                  record.gradient.l2_norm,
                  record.update.l2_norm,
                  record.weight_after.l2_norm});
}

void LayerAuditCollector::update_hybrid_summary_unlocked(
    PhaseAuditAccumulator& accumulator,
    const HybridInteractionAuditRecord& record) {
    auto& summary = accumulator.summary;
    ++summary.hybrid_interaction_records;
    const double count =
        static_cast<double>(summary.hybrid_interaction_records);
    summary.hybrid_signal_cosine_mean +=
        (record.signal_cosine -
         summary.hybrid_signal_cosine_mean) /
        count;
    summary.hybrid_contribution_cosine_mean +=
        (record.contribution_cosine -
         summary.hybrid_contribution_cosine_mean) /
        count;
    summary.hybrid_attention_to_mamba_mean +=
        (record.attention_to_mamba_contribution_ratio -
         summary.hybrid_attention_to_mamba_mean) /
        count;
    summary.hybrid_mamba_ffn_contribution_cosine_mean +=
        (record.mamba_ffn_contribution_cosine -
         summary.hybrid_mamba_ffn_contribution_cosine_mean) /
        count;
    summary.hybrid_attention_ffn_contribution_cosine_mean +=
        (record.attention_ffn_contribution_cosine -
         summary.hybrid_attention_ffn_contribution_cosine_mean) /
        count;
    summary.hybrid_ffn_to_mamba_mean +=
        (record.ffn_to_mamba_contribution_ratio -
         summary.hybrid_ffn_to_mamba_mean) /
        count;
    summary.hybrid_cancellation_max =
        std::max(summary.hybrid_cancellation_max,
                 record.cancellation_fraction);
    size_t* pass_records = nullptr;
    double* pass_signal_cosine = nullptr;
    double* pass_contribution_cosine = nullptr;
    double* pass_attention_to_mamba = nullptr;
    double* pass_cancellation_max = nullptr;
    if (record.pass == "forward") {
        pass_records = &summary.hybrid_forward_interaction_records;
        pass_signal_cosine =
            &summary.hybrid_forward_signal_cosine_mean;
        pass_contribution_cosine =
            &summary.hybrid_forward_contribution_cosine_mean;
        pass_attention_to_mamba =
            &summary.hybrid_forward_attention_to_mamba_mean;
        pass_cancellation_max =
            &summary.hybrid_forward_cancellation_max;
    } else if (record.pass == "backward") {
        pass_records = &summary.hybrid_backward_interaction_records;
        pass_signal_cosine =
            &summary.hybrid_backward_signal_cosine_mean;
        pass_contribution_cosine =
            &summary.hybrid_backward_contribution_cosine_mean;
        pass_attention_to_mamba =
            &summary.hybrid_backward_attention_to_mamba_mean;
        pass_cancellation_max =
            &summary.hybrid_backward_cancellation_max;
    } else {
        throw std::logic_error(
            "hybrid summary received an unsupported pass");
    }
    ++(*pass_records);
    const double pass_count = static_cast<double>(*pass_records);
    *pass_signal_cosine +=
        (record.signal_cosine - *pass_signal_cosine) / pass_count;
    *pass_contribution_cosine +=
        (record.contribution_cosine - *pass_contribution_cosine) /
        pass_count;
    *pass_attention_to_mamba +=
        (record.attention_to_mamba_contribution_ratio -
         *pass_attention_to_mamba) /
        pass_count;
    *pass_cancellation_max =
        std::max(*pass_cancellation_max,
                 record.cancellation_fraction);
    summary.total_nan +=
        record.mamba_signal.nan_count +
        record.attention_signal.nan_count +
        record.ffn_signal.nan_count +
        record.mamba_contribution.nan_count +
        record.attention_contribution.nan_count +
        record.ffn_contribution.nan_count +
        record.combined_contribution.nan_count;
    summary.total_inf +=
        record.mamba_signal.inf_count +
        record.attention_signal.inf_count +
        record.ffn_signal.inf_count +
        record.mamba_contribution.inf_count +
        record.attention_contribution.inf_count +
        record.ffn_contribution.inf_count +
        record.combined_contribution.inf_count;
    summary.max_l2_norm =
        std::max({summary.max_l2_norm,
                  record.mamba_signal.l2_norm,
                  record.attention_signal.l2_norm,
                  record.ffn_signal.l2_norm,
                  record.mamba_contribution.l2_norm,
                  record.attention_contribution.l2_norm,
                  record.ffn_contribution.l2_norm,
                  record.combined_contribution.l2_norm});
    add_layer_seen(summary.layers_seen, record.layer_index);
}

TensorAuditStats LayerAuditCollector::summarize_tensor(const Tensor& tensor) {
    TensorAuditStats stats;
    stats.shape = tensor.shape.dims;
    stats.elements = tensor.size;
    if (tensor.size <= 0) {
        return stats;
    }

    // Native device tensors are reduced in place; only a fixed-size statistics
    // record crosses D2H. This keeps audit mode usable on production-size
    // activations and prevents a hidden full-tensor transfer per layer.
#ifdef USE_CUDA
    if (tensor.get_device() == Device::GPU) {
        return tensor_stats_from_device_reduction(
            reduce_tensor_audit_on_device(tensor),
            tensor.shape, tensor.size);
    }
#else
    if (tensor.get_device() == Device::GPU) {
        throw std::runtime_error(
            "GPU tensor audit requires a GPU-enabled build");
    }
#endif
    const Tensor& host = tensor;
    const float* ptr = host.data();
    if (ptr == nullptr) {
        return stats;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    size_t finite_count = 0;

    for (int i = 0; i < host.size; ++i) {
        const float value = ptr[i];
        if (std::isnan(value)) {
            ++stats.nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++stats.inf_count;
            continue;
        }
        if (value == 0.0f) {
            ++stats.zero_count;
        } else {
            if (value > 0.0f) {
                ++stats.positive_count;
            } else {
                ++stats.negative_count;
            }
            if (std::fpclassify(value) == FP_SUBNORMAL) {
                ++stats.subnormal_count;
            }
        }
        const double d = static_cast<double>(value);
        sum += d;
        sum_sq += d * d;
        min_value = std::min(min_value, d);
        max_value = std::max(max_value, d);
        max_abs = std::max(max_abs, std::abs(d));
        ++finite_count;
    }

    stats.finite = stats.nan_count == 0 && stats.inf_count == 0;
    if (finite_count == 0) {
        return stats;
    }

    stats.min = min_value;
    stats.max = max_value;
    stats.mean = sum / static_cast<double>(finite_count);
    const double mean_sq = sum_sq / static_cast<double>(finite_count);
    const double variance = std::max(0.0, mean_sq - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    stats.l2_norm = std::sqrt(sum_sq);
    stats.max_abs = max_abs;
    return stats;
}

void LayerAuditCollector::record_token_context(const std::vector<int>& token_ids_sample,
                                               size_t batch_size,
                                               size_t prompt_tokens_total,
                                               size_t prompt_tokens_used,
                                               int context_limit,
                                               bool truncated) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }
    TokenContextAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.step = step_;
    record.batch_size = std::max<size_t>(batch_size, 1);
    record.prompt_tokens_total = prompt_tokens_total;
    record.prompt_tokens_used = prompt_tokens_used;
    record.context_limit = context_limit;
    record.truncated = truncated;
    const size_t sample_limit = std::min<size_t>(token_ids_sample.size(), 128);
    record.token_ids_sample.assign(token_ids_sample.begin(),
                                   token_ids_sample.begin() + sample_limit);
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    ++accumulator.summary.token_contexts;
    if (truncated) {
        ++accumulator.summary.truncated_contexts;
    }
    const bool store_record =
        store_token_contexts_ && !summary_only_ &&
        (max_records_per_phase_ == 0 ||
         accumulator.summary.stored_records < max_records_per_phase_);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        token_contexts_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_forward(int layer_index,
                                         const std::string& block_type,
                                         const std::string& tensor_role,
                                         const Tensor& input,
                                         const Tensor& output,
                                         double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }
    auto& accumulator = phase_accumulator_unlocked(phase_);
    const bool collect_stats =
        should_collect_layer_stats_unlocked(accumulator, "forward", layer_index);
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "forward";
    record.block_type = block_type;
    record.tensor_role = tensor_role;
    record.step = step_;
    record.layer_index = layer_index;
    if (collect_stats) {
        record.input = summarize_tensor(input);
        record.output = summarize_tensor(output);
    }
    record.latency_ms = latency_ms;
    update_layer_summary_unlocked(accumulator, record, collect_stats);
    const bool store_record =
        collect_stats && should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_backward(int layer_index,
                                          const std::string& block_type,
                                          const Tensor& grad_output,
                                          const Tensor& grad_input,
                                          double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }
    auto& accumulator = phase_accumulator_unlocked(phase_);
    const bool collect_stats =
        should_collect_layer_stats_unlocked(accumulator, "backward", layer_index);
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "backward";
    record.block_type = block_type;
    record.tensor_role = "gradient";
    record.step = step_;
    record.layer_index = layer_index;
    if (collect_stats) {
        record.input = summarize_tensor(grad_output);
        record.output = summarize_tensor(grad_input);
    }
    record.latency_ms = latency_ms;
    record.grad_l2_norm = record.output.l2_norm;
    update_layer_summary_unlocked(accumulator, record, collect_stats);
    const bool store_record =
        collect_stats && should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_router(int layer_index,
                                        const std::string& block_type,
                                        int rows,
                                        int num_experts,
                                        int top_k,
                                        const std::vector<int>& topk_counts,
                                        const std::vector<float>& expert_loads) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }
    LayerAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = "router";
    record.block_type = block_type;
    record.tensor_role = "moe_router";
    record.step = step_;
    record.layer_index = layer_index;
    record.has_router = true;
    record.router.rows = rows;
    record.router.num_experts = num_experts;
    record.router.top_k = top_k;
    record.router.topk_counts = topk_counts;
    record.router.expert_loads = expert_loads;
    record.router.entropy = router_entropy(expert_loads, topk_counts);
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    update_layer_summary_unlocked(accumulator, record, true);
    const bool store_record = should_store_layer_record_unlocked(accumulator, record.pass);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::record_training_step(int step,
                                               double loss,
                                               double grad_l2_norm,
                                               size_t parameter_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }
    TrainingStepAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.step = step;
    record.loss = loss;
    record.grad_l2_norm = grad_l2_norm;
    record.parameter_count = parameter_count;
    auto& accumulator = phase_accumulator_unlocked(record.phase);
    update_training_summary_unlocked(accumulator, record);
    const bool store_record =
        !summary_only_ &&
        (max_records_per_phase_ == 0 ||
         accumulator.summary.stored_records < max_records_per_phase_);
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        training_steps_.push_back(std::move(record));
    }
    step_ = step;
}

void LayerAuditCollector::record_hybrid_interaction(
    int layer_index,
    const std::string& pass,
    const Tensor& mamba_signal,
    const Tensor& attention_signal,
    const Tensor& ffn_signal,
    const Tensor& mamba_contribution,
    const Tensor& attention_contribution,
    const Tensor& ffn_contribution) {
    if (pass != "forward" && pass != "backward") {
        throw std::invalid_argument(
            "hybrid interaction pass must be forward or backward");
    }
    if (mamba_signal.shape != attention_signal.shape ||
        mamba_signal.shape != ffn_signal.shape ||
        mamba_contribution.shape != attention_contribution.shape ||
        mamba_contribution.shape != ffn_contribution.shape ||
        mamba_signal.shape != mamba_contribution.shape) {
        throw std::invalid_argument(
            "hybrid interaction tensors must have identical shapes");
    }
    const Device interaction_device = mamba_signal.get_device();
    if (attention_signal.get_device() != interaction_device ||
        ffn_signal.get_device() != interaction_device ||
        mamba_contribution.get_device() != interaction_device ||
        attention_contribution.get_device() != interaction_device ||
        ffn_contribution.get_device() != interaction_device) {
        throw std::invalid_argument(
            "hybrid interaction tensors must share one device");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return;
    }

    HybridInteractionAuditRecord record;
    record.sequence = next_sequence_unlocked();
    record.run_id = run_id_;
    record.phase = phase_;
    record.pass = pass;
    record.step = step_;
    record.layer_index = layer_index;
    if (interaction_device == Device::GPU) {
#ifdef USE_CUDA
        const NsosHybridAuditDeviceMetrics raw =
            reduce_hybrid_audit_on_device(
                mamba_signal, attention_signal, ffn_signal,
                mamba_contribution, attention_contribution,
                ffn_contribution);
        TensorAuditStats* stats[7] = {
            &record.mamba_signal,
            &record.attention_signal,
            &record.ffn_signal,
            &record.mamba_contribution,
            &record.attention_contribution,
            &record.ffn_contribution,
            &record.combined_contribution};
        for (size_t index = 0; index < 7; ++index) {
            *stats[index] =
                tensor_stats_from_device_reduction(
                    raw.tensor_stats[index],
                    mamba_signal.shape,
                    mamba_signal.size);
        }
        record.signal_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[0], record.mamba_signal,
                record.attention_signal);
        record.mamba_ffn_signal_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[1], record.mamba_signal,
                record.ffn_signal);
        record.attention_ffn_signal_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[2], record.attention_signal,
                record.ffn_signal);
        record.contribution_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[3], record.mamba_contribution,
                record.attention_contribution);
        record.mamba_ffn_contribution_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[4], record.mamba_contribution,
                record.ffn_contribution);
        record.attention_ffn_contribution_cosine =
            cosine_from_device_reduction(
                raw.pair_dots[5],
                record.attention_contribution,
                record.ffn_contribution);
#else
        throw std::runtime_error(
            "GPU hybrid audit requires a GPU-enabled build");
#endif
    } else {
        const Tensor mamba_signal_host = mamba_signal.clone();
        const Tensor attention_signal_host =
            attention_signal.clone();
        const Tensor ffn_signal_host = ffn_signal.clone();
        const Tensor mamba_contribution_host =
            mamba_contribution.clone();
        const Tensor attention_contribution_host =
            attention_contribution.clone();
        const Tensor ffn_contribution_host =
            ffn_contribution.clone();
        Tensor combined =
            Tensor::zeros(
                mamba_contribution_host.shape.dims,
                Device::CPU);
        const float* mamba_values =
            mamba_contribution_host.data();
        const float* attention_values =
            attention_contribution_host.data();
        const float* ffn_values =
            ffn_contribution_host.data();
        float* combined_values = combined.data();
        for (int index = 0; index < combined.size; ++index) {
            combined_values[index] =
                mamba_values[index] +
                attention_values[index] +
                ffn_values[index];
        }
        record.mamba_signal =
            summarize_tensor(mamba_signal_host);
        record.attention_signal =
            summarize_tensor(attention_signal_host);
        record.ffn_signal =
            summarize_tensor(ffn_signal_host);
        record.mamba_contribution =
            summarize_tensor(mamba_contribution_host);
        record.attention_contribution =
            summarize_tensor(attention_contribution_host);
        record.ffn_contribution =
            summarize_tensor(ffn_contribution_host);
        record.combined_contribution =
            summarize_tensor(combined);
        record.signal_cosine =
            tensor_cosine(
                mamba_signal_host, attention_signal_host);
        record.contribution_cosine =
            tensor_cosine(
                mamba_contribution_host,
                attention_contribution_host);
        record.mamba_ffn_signal_cosine =
            tensor_cosine(
                mamba_signal_host, ffn_signal_host);
        record.attention_ffn_signal_cosine =
            tensor_cosine(
                attention_signal_host, ffn_signal_host);
        record.mamba_ffn_contribution_cosine =
            tensor_cosine(
                mamba_contribution_host,
                ffn_contribution_host);
        record.attention_ffn_contribution_cosine =
            tensor_cosine(
                attention_contribution_host,
                ffn_contribution_host);
    }
    record.attention_to_mamba_signal_ratio =
        stable_ratio(record.attention_signal.l2_norm,
                     record.mamba_signal.l2_norm);
    record.attention_to_mamba_contribution_ratio =
        stable_ratio(record.attention_contribution.l2_norm,
                     record.mamba_contribution.l2_norm);
    record.ffn_to_mamba_signal_ratio =
        stable_ratio(record.ffn_signal.l2_norm,
                     record.mamba_signal.l2_norm);
    record.ffn_to_mamba_contribution_ratio =
        stable_ratio(record.ffn_contribution.l2_norm,
                     record.mamba_contribution.l2_norm);
    const double separate_norm =
        record.mamba_contribution.l2_norm +
        record.attention_contribution.l2_norm +
        record.ffn_contribution.l2_norm;
    record.cancellation_fraction =
        separate_norm > 0.0
            ? std::clamp(
                  1.0 -
                      record.combined_contribution.l2_norm /
                          separate_norm,
                  0.0, 1.0)
            : 0.0;

    auto& accumulator =
        phase_accumulator_unlocked(record.phase);
    update_hybrid_summary_unlocked(accumulator, record);
    const bool store_record =
        should_store_layer_record_unlocked(accumulator,
                                           "hybrid_interaction");
    mark_layer_record_storage_unlocked(accumulator, store_record);
    if (store_record) {
        hybrid_interaction_records_.push_back(std::move(record));
    }
}

void LayerAuditCollector::begin_parameter_step(
    int step,
    const std::vector<Parameter*>& parameters) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_parameters_.empty() || pending_parameter_step_ >= 0) {
        throw std::logic_error(
            "parameter audit step started before the previous step completed");
    }
    if (!enabled_ || !parameter_audit_enabled_ || summary_only_ ||
        step < 0 ||
        (step % parameter_step_sample_rate_) != 0) {
        return;
    }

    size_t active_count = 0;
    size_t snapshot_bytes = 0;
    bool has_gpu_parameter = false;
    std::set<Parameter*> unique_parameters;
    std::set<std::string> unique_names;
    for (Parameter* parameter : parameters) {
        if (!parameter || parameter->data.size <= 0) {
            continue;
        }
        if (!unique_parameters.insert(parameter).second) {
            throw std::runtime_error(
                "parameter audit received a duplicate parameter pointer");
        }
        const std::string& name =
            parameter->name.empty() ? parameter->base_name : parameter->name;
        if (name.empty() || !unique_names.insert(name).second) {
            throw std::runtime_error(
                "parameter audit requires unique non-empty parameter names");
        }
        const size_t elements =
            static_cast<size_t>(parameter->data.size);
        // Peak transactional cohort: before + gradient remain live while
        // complete() materializes after + update. Hashing is streamed/short-
        // lived, so four full snapshots are the persistent peak.
        constexpr size_t kSnapshotsPerElement = 4;
        if (elements >
            (std::numeric_limits<size_t>::max() - snapshot_bytes) /
                (kSnapshotsPerElement * sizeof(float))) {
            throw std::overflow_error(
                "parameter audit snapshot size overflow");
        }
        snapshot_bytes +=
            elements * kSnapshotsPerElement * sizeof(float);
        has_gpu_parameter =
            has_gpu_parameter ||
            parameter->data.get_device() == Device::GPU ||
            (parameter->grad.size > 0 &&
             parameter->grad.get_device() == Device::GPU);
        ++active_count;
    }
    if (snapshot_bytes > max_parameter_snapshot_bytes_) {
        throw std::runtime_error(
            "parameter audit snapshot requires " +
            std::to_string(snapshot_bytes) +
            " bytes, exceeding the configured safety limit of " +
            std::to_string(max_parameter_snapshot_bytes_));
    }
    if (max_parameter_records_ > 0 &&
        (parameter_records_.size() >= max_parameter_records_ ||
         active_count >
             max_parameter_records_ - parameter_records_.size())) {
        // Preserve step atomicity: never persist a partial parameter cohort.
        return;
    }

    synchronize_parameter_audit_device(has_gpu_parameter);
    std::vector<PendingParameterAudit> staged_parameters;
    staged_parameters.reserve(active_count);
    for (Parameter* parameter : parameters) {
        if (!parameter || parameter->data.size <= 0) {
            continue;
        }
        PendingParameterAudit pending;
        pending.parameter = parameter;
        pending.name =
            parameter->name.empty() ? parameter->base_name : parameter->name;
        pending.base_name = parameter->base_name;
        pending.version_before = parameter->version;
        pending.weight_before =
            parameter->data.get_device() == Device::CPU
                ? parameter->data.clone()
                : parameter->data.cpu();
        if (!parameter->has_gradient()) {
            pending.gradient =
                Tensor::zeros(parameter->data.shape.dims, Device::CPU);
        } else {
            Tensor aligned_gradient = parameter->grad;
            if (aligned_gradient.size == parameter->data.size &&
                aligned_gradient.shape != parameter->data.shape) {
                aligned_gradient =
                    aligned_gradient.reshape(parameter->data.shape.dims);
            }
            if (aligned_gradient.shape != parameter->data.shape) {
                throw std::runtime_error(
                    "parameter audit gradient shape mismatch for '" +
                    pending.name + "'");
            }
            pending.gradient =
                aligned_gradient.get_device() == Device::CPU
                    ? aligned_gradient.clone()
                    : aligned_gradient.cpu();
        }
        staged_parameters.push_back(std::move(pending));
    }
    pending_parameters_ = std::move(staged_parameters);
    pending_parameter_step_ = step;
    pending_parameter_run_id_ = run_id_;
    pending_parameter_phase_ = phase_;
}

void LayerAuditCollector::complete_parameter_step(
    int step,
    bool optimizer_applied) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_parameter_step_ < 0) {
        return;
    }
    if (step != pending_parameter_step_) {
        throw std::logic_error(
            "parameter audit completion step does not match its snapshot");
    }

    bool has_gpu_parameter = false;
    for (const PendingParameterAudit& pending : pending_parameters_) {
        has_gpu_parameter =
            has_gpu_parameter ||
            (pending.parameter &&
             pending.parameter->data.get_device() == Device::GPU);
    }
    synchronize_parameter_audit_device(has_gpu_parameter);

    std::vector<ParameterAuditRecord> staged_records;
    staged_records.reserve(pending_parameters_.size());
    for (const PendingParameterAudit& pending : pending_parameters_) {
        if (!pending.parameter) {
            throw std::logic_error(
                "parameter audit lost a parameter before completion");
        }
        Tensor after =
            pending.parameter->data.get_device() == Device::CPU
                ? pending.parameter->data.clone()
                : pending.parameter->data.cpu();
        if (after.shape != pending.weight_before.shape) {
            throw std::runtime_error(
                "parameter shape changed during audited optimizer step for '" +
                pending.name + "'");
        }
        Tensor update =
            Tensor::zeros(after.shape.dims, Device::CPU);
        float* update_values = update.data();
        const float* before_values = pending.weight_before.data();
        const float* after_values = after.data();
        const float* gradient_values = pending.gradient.data();
        double update_gradient_dot = 0.0;
        double update_sq = 0.0;
        double gradient_sq = 0.0;
        size_t changed_elements = 0;
        size_t ternary_before = 0;
        size_t ternary_after = 0;
        for (int index = 0; index < after.size; ++index) {
            const float before_value = before_values[index];
            const float after_value = after_values[index];
            const float gradient_value = gradient_values[index];
            const float delta = after_value - before_value;
            update_values[index] = delta;

            uint32_t before_bits = 0;
            uint32_t after_bits = 0;
            std::memcpy(&before_bits, &before_value, sizeof(before_bits));
            std::memcpy(&after_bits, &after_value, sizeof(after_bits));
            changed_elements += before_bits != after_bits ? 1u : 0u;
            ternary_before +=
                (before_value == -1.0f || before_value == 0.0f ||
                 before_value == 1.0f)
                    ? 1u
                    : 0u;
            ternary_after +=
                (after_value == -1.0f || after_value == 0.0f ||
                 after_value == 1.0f)
                    ? 1u
                    : 0u;
            if (std::isfinite(delta) && std::isfinite(gradient_value)) {
                const double delta_d = static_cast<double>(delta);
                const double gradient_d =
                    static_cast<double>(gradient_value);
                update_gradient_dot += delta_d * gradient_d;
                update_sq += delta_d * delta_d;
                gradient_sq += gradient_d * gradient_d;
            }
        }

        ParameterAuditRecord record;
        record.run_id = pending_parameter_run_id_;
        record.phase = pending_parameter_phase_;
        record.step = step;
        record.name = pending.name;
        record.base_name = pending.base_name;
        record.layer_index = parameter_layer_index(record.name);
        record.component = parameter_component(record.name);
        record.role = parameter_role(record.name);
        record.optimizer_applied = optimizer_applied;
        record.version_before = pending.version_before;
        record.version_after = pending.parameter->version;
        record.weight_before = summarize_tensor(pending.weight_before);
        record.gradient = summarize_tensor(pending.gradient);
        record.update = summarize_tensor(update);
        record.weight_after = summarize_tensor(after);
        record.grad_to_weight_ratio =
            stable_ratio(record.gradient.l2_norm,
                         record.weight_before.l2_norm);
        record.update_to_weight_ratio =
            stable_ratio(record.update.l2_norm,
                         record.weight_before.l2_norm);
        record.update_to_grad_ratio =
            stable_ratio(record.update.l2_norm,
                         record.gradient.l2_norm);
        if (update_sq > 0.0 && gradient_sq > 0.0) {
            record.gradient_update_cosine =
                std::clamp(
                    update_gradient_dot /
                        std::sqrt(update_sq * gradient_sq),
                    -1.0, 1.0);
        }
        record.changed_elements = changed_elements;
        record.ternary_elements_before = ternary_before;
        record.ternary_elements_after = ternary_after;
        record.weight_sha256_before =
            tensor_sha256(pending.weight_before);
        record.gradient_sha256 = tensor_sha256(pending.gradient);
        record.update_sha256 = tensor_sha256(update);
        record.weight_sha256_after = tensor_sha256(after);
        staged_records.push_back(std::move(record));
    }

    for (ParameterAuditRecord& record : staged_records) {
        record.sequence = next_sequence_unlocked();
        auto& accumulator =
            phase_accumulator_unlocked(record.phase);
        update_parameter_summary_unlocked(accumulator, record);
        parameter_records_.push_back(std::move(record));
    }

    pending_parameters_.clear();
    pending_parameter_step_ = -1;
    pending_parameter_run_id_.clear();
    pending_parameter_phase_.clear();
}

void LayerAuditCollector::abort_parameter_step() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_parameters_.clear();
        pending_parameter_step_ = -1;
        pending_parameter_run_id_.clear();
        pending_parameter_phase_.clear();
    } catch (...) {
        // noexcept cleanup used during optimizer exception unwinding.
    }
}

std::vector<LayerAuditRecord> LayerAuditCollector::records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

std::vector<TokenContextAuditRecord> LayerAuditCollector::token_contexts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return token_contexts_;
}

std::vector<TrainingStepAuditRecord> LayerAuditCollector::training_steps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return training_steps_;
}

std::vector<ParameterAuditRecord>
LayerAuditCollector::parameter_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parameter_records_;
}

std::vector<HybridInteractionAuditRecord>
LayerAuditCollector::hybrid_interaction_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hybrid_interaction_records_;
}

LayerAuditSummary LayerAuditCollector::summarize_phase(const std::string& phase) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto* accumulator = find_phase_accumulator_unlocked(phase)) {
        return accumulator->summary;
    }

    LayerAuditSummary summary;
    summary.phase = phase;
    std::set<int> layers_seen;

    for (const auto& record : records_) {
        if (record.phase != phase) {
            continue;
        }
        ++summary.records;
        if (record.pass == "forward") {
            ++summary.forward_records;
        } else if (record.pass == "backward") {
            ++summary.backward_records;
        } else if (record.pass == "router") {
            ++summary.router_records;
        }
        summary.total_nan += record.input.nan_count + record.output.nan_count;
        summary.total_inf += record.input.inf_count + record.output.inf_count;
        summary.max_latency_ms = std::max(summary.max_latency_ms, record.latency_ms);
        summary.max_l2_norm =
            std::max(summary.max_l2_norm,
                     std::max(record.input.l2_norm, record.output.l2_norm));
        add_layer_seen(layers_seen, record.layer_index);
    }

    for (const auto& context : token_contexts_) {
        if (context.phase == phase) {
            ++summary.token_contexts;
        }
    }
    for (const auto& step : training_steps_) {
        if (step.phase == phase) {
            ++summary.training_steps;
        }
    }

    summary.layers_seen.assign(layers_seen.begin(), layers_seen.end());
    return summary;
}

bool LayerAuditCollector::compare_phase_health(const std::string& lhs_phase,
                                               const std::string& rhs_phase,
                                               std::string* reason) const {
    const LayerAuditSummary lhs = summarize_phase(lhs_phase);
    const LayerAuditSummary rhs = summarize_phase(rhs_phase);
    auto fail = [&](const std::string& message) {
        if (reason) {
            *reason = message;
        }
        return false;
    };

    if (!lhs.healthy()) {
        return fail("left phase has NaN/Inf: " + lhs_phase);
    }
    if (!rhs.healthy()) {
        return fail("right phase has NaN/Inf: " + rhs_phase);
    }
    if (lhs.forward_records == 0 || rhs.forward_records == 0) {
        return fail("both phases must contain forward records");
    }
    if (lhs.layers_seen.empty() || rhs.layers_seen.empty()) {
        return fail("both phases must see at least one layer");
    }
    if (lhs.layers_seen != rhs.layers_seen) {
        return fail("phase layer coverage differs");
    }
    if (reason) {
        reason->clear();
    }
    return true;
}

void LayerAuditCollector::write_json(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_parameter_step_ >= 0 || !pending_parameters_.empty()) {
        throw std::logic_error(
            "cannot serialize audit while a parameter step is pending");
    }
    namespace fs = std::filesystem;
    const fs::path output_path(path);
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    const fs::path temporary_path =
        unique_audit_temporary_path(output_path);
    TemporaryAuditFileGuard temporary_guard(temporary_path);
    std::ofstream out(temporary_path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Could not write audit JSON: " + path);
    }
    out << std::setprecision(10);
    out << "{\n";
    out << "  \"schema_version\":2,\n";
    out << "  \"run_id\":\"" << json_escape(run_id_) << "\",\n";
    out << "  \"storage_policy\":{";
    out << "\"summary_only\":" << (summary_only_ ? "true" : "false");
    out << ",\"record_sample_rate\":" << record_sample_rate_;
    out << ",\"max_records_per_phase\":" << max_records_per_phase_;
    out << ",\"store_token_contexts\":" << (store_token_contexts_ ? "true" : "false");
    out << "},\n";
    out << "  \"parameter_audit_policy\":{";
    out << "\"enabled\":"
        << (parameter_audit_enabled_ ? "true" : "false");
    out << ",\"step_sample_rate\":" << parameter_step_sample_rate_;
    out << ",\"max_records\":" << max_parameter_records_;
    out << ",\"max_snapshot_bytes\":"
        << max_parameter_snapshot_bytes_;
    out << ",\"hash_format\":\"sha256/nsos-tensor-f32-le-v1\"";
    out << "},\n";
    out << "  \"phase_summaries\":[\n";
    size_t summary_index = 0;
    for (const auto& [_, accumulator] : phase_summaries_) {
        out << "    ";
        write_layer_summary(out, accumulator.summary);
        if (++summary_index < phase_summaries_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"records\":[\n";
    for (size_t i = 0; i < records_.size(); ++i) {
        const auto& record = records_[i];
        out << "    {";
        out << "\"sequence\":" << record.sequence;
        out << ",\"run_id\":\"" << json_escape(record.run_id) << "\"";
        out << ",\"phase\":\"" << json_escape(record.phase) << "\"";
        out << ",\"pass\":\"" << json_escape(record.pass) << "\"";
        out << ",\"step\":" << record.step;
        out << ",\"layer_index\":" << record.layer_index;
        out << ",\"block_type\":\"" << json_escape(record.block_type) << "\"";
        out << ",\"tensor_role\":\"" << json_escape(record.tensor_role) << "\"";
        out << ",\"latency_ms\":";
        write_json_number(out, record.latency_ms);
        out << ",\"grad_l2_norm\":";
        write_json_number(out, record.grad_l2_norm);
        out << ",\"input\":";
        write_tensor_stats(out, record.input);
        out << ",\"output\":";
        write_tensor_stats(out, record.output);
        if (record.has_router) {
            out << ",\"router\":{";
            out << "\"rows\":" << record.router.rows;
            out << ",\"num_experts\":" << record.router.num_experts;
            out << ",\"top_k\":" << record.router.top_k;
            out << ",\"topk_counts\":";
            write_numeric_array(out, record.router.topk_counts);
            out << ",\"expert_loads\":";
            write_numeric_array(out, record.router.expert_loads);
            out << ",\"entropy\":";
            write_json_number(out, record.router.entropy);
            out << "}";
        }
        out << "}";
        if (i + 1 < records_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"token_contexts\":[\n";
    for (size_t i = 0; i < token_contexts_.size(); ++i) {
        const auto& context = token_contexts_[i];
        out << "    {";
        out << "\"sequence\":" << context.sequence;
        out << ",\"run_id\":\"" << json_escape(context.run_id) << "\"";
        out << ",\"phase\":\"" << json_escape(context.phase) << "\"";
        out << ",\"step\":" << context.step;
        out << ",\"batch_size\":" << context.batch_size;
        out << ",\"prompt_tokens_total\":" << context.prompt_tokens_total;
        out << ",\"prompt_tokens_used\":" << context.prompt_tokens_used;
        out << ",\"context_limit\":" << context.context_limit;
        out << ",\"truncated\":" << (context.truncated ? "true" : "false");
        out << ",\"token_ids_sample\":";
        write_numeric_array(out, context.token_ids_sample);
        out << "}";
        if (i + 1 < token_contexts_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"training_steps\":[\n";
    for (size_t i = 0; i < training_steps_.size(); ++i) {
        const auto& step = training_steps_[i];
        out << "    {";
        out << "\"sequence\":" << step.sequence;
        out << ",\"run_id\":\"" << json_escape(step.run_id) << "\"";
        out << ",\"phase\":\"" << json_escape(step.phase) << "\"";
        out << ",\"step\":" << step.step;
        out << ",\"loss\":";
        write_json_number(out, step.loss);
        out << ",\"grad_l2_norm\":";
        write_json_number(out, step.grad_l2_norm);
        out << ",\"parameter_count\":" << step.parameter_count;
        out << "}";
        if (i + 1 < training_steps_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"hybrid_interactions\":[\n";
    for (size_t i = 0; i < hybrid_interaction_records_.size(); ++i) {
        const auto& record = hybrid_interaction_records_[i];
        out << "    {";
        out << "\"sequence\":" << record.sequence;
        out << ",\"run_id\":\"" << json_escape(record.run_id) << "\"";
        out << ",\"phase\":\"" << json_escape(record.phase) << "\"";
        out << ",\"pass\":\"" << json_escape(record.pass) << "\"";
        out << ",\"step\":" << record.step;
        out << ",\"layer_index\":" << record.layer_index;
        out << ",\"mamba_signal\":";
        write_tensor_stats(out, record.mamba_signal);
        out << ",\"attention_signal\":";
        write_tensor_stats(out, record.attention_signal);
        out << ",\"ffn_signal\":";
        write_tensor_stats(out, record.ffn_signal);
        out << ",\"mamba_contribution\":";
        write_tensor_stats(out, record.mamba_contribution);
        out << ",\"attention_contribution\":";
        write_tensor_stats(out, record.attention_contribution);
        out << ",\"ffn_contribution\":";
        write_tensor_stats(out, record.ffn_contribution);
        out << ",\"combined_contribution\":";
        write_tensor_stats(out, record.combined_contribution);
        out << ",\"signal_cosine\":";
        write_json_number(out, record.signal_cosine);
        out << ",\"contribution_cosine\":";
        write_json_number(out, record.contribution_cosine);
        out << ",\"mamba_ffn_signal_cosine\":";
        write_json_number(out, record.mamba_ffn_signal_cosine);
        out << ",\"attention_ffn_signal_cosine\":";
        write_json_number(out, record.attention_ffn_signal_cosine);
        out << ",\"mamba_ffn_contribution_cosine\":";
        write_json_number(out, record.mamba_ffn_contribution_cosine);
        out << ",\"attention_ffn_contribution_cosine\":";
        write_json_number(
            out, record.attention_ffn_contribution_cosine);
        out << ",\"attention_to_mamba_signal_ratio\":";
        write_json_number(out, record.attention_to_mamba_signal_ratio);
        out << ",\"attention_to_mamba_contribution_ratio\":";
        write_json_number(
            out, record.attention_to_mamba_contribution_ratio);
        out << ",\"ffn_to_mamba_signal_ratio\":";
        write_json_number(out, record.ffn_to_mamba_signal_ratio);
        out << ",\"ffn_to_mamba_contribution_ratio\":";
        write_json_number(
            out, record.ffn_to_mamba_contribution_ratio);
        out << ",\"cancellation_fraction\":";
        write_json_number(out, record.cancellation_fraction);
        out << "}";
        if (i + 1 < hybrid_interaction_records_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"parameter_records\":[\n";
    for (size_t i = 0; i < parameter_records_.size(); ++i) {
        const auto& record = parameter_records_[i];
        out << "    {";
        out << "\"sequence\":" << record.sequence;
        out << ",\"run_id\":\"" << json_escape(record.run_id) << "\"";
        out << ",\"phase\":\"" << json_escape(record.phase) << "\"";
        out << ",\"step\":" << record.step;
        out << ",\"name\":\"" << json_escape(record.name) << "\"";
        out << ",\"base_name\":\""
            << json_escape(record.base_name) << "\"";
        out << ",\"layer_index\":" << record.layer_index;
        out << ",\"component\":\""
            << json_escape(record.component) << "\"";
        out << ",\"role\":\"" << json_escape(record.role) << "\"";
        out << ",\"trainable\":"
            << (record.trainable ? "true" : "false");
        out << ",\"optimizer_applied\":"
            << (record.optimizer_applied ? "true" : "false");
        out << ",\"version_before\":" << record.version_before;
        out << ",\"version_after\":" << record.version_after;
        out << ",\"weight_before\":";
        write_tensor_stats(out, record.weight_before);
        out << ",\"gradient\":";
        write_tensor_stats(out, record.gradient);
        out << ",\"update\":";
        write_tensor_stats(out, record.update);
        out << ",\"weight_after\":";
        write_tensor_stats(out, record.weight_after);
        out << ",\"grad_to_weight_ratio\":";
        write_json_number(out, record.grad_to_weight_ratio);
        out << ",\"update_to_weight_ratio\":";
        write_json_number(out, record.update_to_weight_ratio);
        out << ",\"update_to_grad_ratio\":";
        write_json_number(out, record.update_to_grad_ratio);
        out << ",\"gradient_update_cosine\":";
        write_json_number(out, record.gradient_update_cosine);
        out << ",\"changed_elements\":" << record.changed_elements;
        out << ",\"ternary_elements_before\":"
            << record.ternary_elements_before;
        out << ",\"ternary_elements_after\":"
            << record.ternary_elements_after;
        out << ",\"weight_sha256_before\":\""
            << record.weight_sha256_before << "\"";
        out << ",\"gradient_sha256\":\""
            << record.gradient_sha256 << "\"";
        out << ",\"update_sha256\":\""
            << record.update_sha256 << "\"";
        out << ",\"weight_sha256_after\":\""
            << record.weight_sha256_after << "\"";
        out << "}";
        if (i + 1 < parameter_records_.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    if (!out) {
        throw std::runtime_error("Could not flush audit JSON: " + path);
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Could not close audit JSON: " + path);
    }
    sync_audit_file_contents(temporary_path);
    replace_audit_file(temporary_path, output_path);
    temporary_guard.release();
}

} // namespace nsos
