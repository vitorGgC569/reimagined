#include "ttt_recurrence.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef USE_CUDA
#include "gpu_backend.h"
#include "cuda/kernels.cuh"
#endif

namespace nsos::ttt {
namespace {
size_t validate(const RecurrenceConfig& c, int rows) {
    const long long cells = static_cast<long long>(c.hidden) * c.dim;
    if (c.hidden <= 0 || c.dim <= 0 || c.chunk != 32 || rows < 0 ||
        cells > std::numeric_limits<int>::max() / c.chunk ||
        !std::isfinite(c.decay) || c.decay < 0 || c.decay >= 1 ||
        !std::isfinite(c.step) || c.step < 0 ||
        !std::isfinite(c.max_norm) || c.max_norm < 0)
        throw std::invalid_argument("Invalid full TTT recurrence geometry or coefficients");
    return static_cast<size_t>(cells);
}

float error_scale(const float* error, int dim, float max_norm, float* norm = nullptr) {
    double sum = 0;
    for (int d = 0; d < dim; ++d) sum += static_cast<double>(error[d]) * error[d];
    const float n = static_cast<float>(std::sqrt(sum));
    if (norm) *norm = n;
    return max_norm > 0 && n > max_norm ? max_norm / (n + 1e-6f) : 1.0f;
}

void advance(const float* key, const float* error, float scale,
             float* a, float* m, const RecurrenceConfig& c) {
    for (int h = 0; h < c.hidden; ++h) for (int d = 0; d < c.dim; ++d) {
        const size_t i = static_cast<size_t>(h) * c.dim + d;
        const float g = (key[h] * error[d]) * scale;
        m[i] = m[i] * c.decay + (c.hamiltonian ? g * (1.0f - c.decay) : g);
        a[i] = c.hamiltonian ? a[i] - (g + m[i] * 0.25f) * c.step
                             : a[i] * c.decay - g * c.step;
    }
}
} // namespace

void forward_sequence(const Tensor& input, const Tensor& keys, const Tensor& base,
    Tensor& adaptation, Tensor& momentum, Tensor& boundaries, Tensor& momentum_boundaries,
    Tensor& errors, Tensor& output, int offset, int rows, int boundary_offset,
    const RecurrenceConfig& c) {
    const size_t cells = validate(c, rows);
    if (rows == 0) return;
    const size_t ro = static_cast<size_t>(offset) * c.dim;
    const size_t ko = static_cast<size_t>(offset) * c.hidden;
    const size_t bo = static_cast<size_t>(boundary_offset) * cells;
#ifdef USE_CUDA
    if (input.get_device() == Device::GPU) {
        Tensor scale = Tensor::uninitialized({1}, Device::GPU);
        if (!launch_ttt_device_forward(input.raw_data() + ro, keys.raw_data() + ko,
            base.raw_data() + ro, adaptation.raw_data(), momentum.raw_data(),
            boundaries.raw_data() + bo, errors.raw_data() + ro, output.raw_data() + ro,
            scale.raw_data(), rows, c.hidden, c.dim, c.decay, c.step, c.max_norm,
            c.hamiltonian, c.chunk, momentum_boundaries.raw_data() + bo))
            throw std::runtime_error("TTT full forward launch failed");
        return;
    }
#endif
    auto* a = adaptation.data(); auto* m = momentum.data();
    for (int r = 0; r < rows; ++r) {
        if (r % c.chunk == 0) {
            const size_t dest = bo + static_cast<size_t>(r / c.chunk) * cells;
            std::memcpy(boundaries.data() + dest, a, cells * sizeof(float));
            std::memcpy(momentum_boundaries.data() + dest, m, cells * sizeof(float));
        }
        const float* k = keys.data() + ko + static_cast<size_t>(r) * c.hidden;
        const size_t row = ro + static_cast<size_t>(r) * c.dim;
        for (int d = 0; d < c.dim; ++d) {
            float correction = 0;
            for (int h = 0; h < c.hidden; ++h)
                correction += k[h] * a[static_cast<size_t>(h) * c.dim + d];
            output.data()[row + d] = base.data()[row + d] + correction;
            errors.data()[row + d] = output.data()[row + d] - input.data()[row + d];
        }
        const float* e = errors.data() + row;
        advance(k, e, error_scale(e, c.dim, c.max_norm), a, m, c);
    }
}

void backward_sequence(const Tensor& keys, const Tensor& errors, const Tensor& boundaries,
    const Tensor& momentum_boundaries, const Tensor& grad, Tensor& grad_keys,
    Tensor& grad_base, Tensor& grad_direct, int offset, int rows, int boundary_offset,
    const RecurrenceConfig& c) {
    const size_t cells = validate(c, rows);
    if (rows == 0) return;
    const size_t ro = static_cast<size_t>(offset) * c.dim;
    const size_t ko = static_cast<size_t>(offset) * c.hidden;
    const size_t bo = static_cast<size_t>(boundary_offset) * cells;
    const int chunk_rows = std::min(rows, c.chunk);
#ifdef USE_CUDA
    if (keys.get_device() == Device::GPU) {
        Tensor a = Tensor::uninitialized({c.hidden, c.dim}, Device::GPU);
        Tensor m = Tensor::uninitialized({c.hidden, c.dim}, Device::GPU);
        Tensor local = Tensor::uninitialized({chunk_rows, c.hidden, c.dim}, Device::GPU);
        Tensor adj_a = Tensor::uninitialized({c.hidden, c.dim}, Device::GPU);
        Tensor adj_m = Tensor::uninitialized({c.hidden, c.dim}, Device::GPU);
        Tensor q = Tensor::uninitialized({c.dim}, Device::GPU);
        Tensor coefficients = Tensor::uninitialized({4}, Device::GPU); // aligned 2 doubles
        if (!launch_ttt_full_backward(keys.raw_data() + ko, errors.raw_data() + ro,
            boundaries.raw_data() + bo, momentum_boundaries.raw_data() + bo, grad.raw_data() + ro,
            grad_keys.raw_data() + ko, grad_base.raw_data() + ro, grad_direct.raw_data() + ro,
            a.raw_data(), m.raw_data(), local.raw_data(), adj_a.raw_data(), adj_m.raw_data(),
            q.raw_data(), reinterpret_cast<double*>(coefficients.raw_data()), rows, c.hidden, c.dim,
            c.chunk, c.decay, c.step, c.max_norm, c.hamiltonian))
            throw std::runtime_error("TTT full backward launch failed");
        return;
    }
#endif
    std::vector<float> a(cells), m(cells), local(static_cast<size_t>(chunk_rows) * cells);
    std::vector<double> adj_a(cells, 0), adj_m(cells, 0), gbar(cells), q(c.dim), total(c.dim);
    for (int chunk = (rows - 1) / c.chunk; chunk >= 0; --chunk) {
        const int start = chunk * c.chunk;
        const int stop = start + std::min(c.chunk, rows - start);
        const size_t boundary = bo + static_cast<size_t>(chunk) * cells;
        std::memcpy(a.data(), boundaries.data() + boundary, cells * sizeof(float));
        std::memcpy(m.data(), momentum_boundaries.data() + boundary, cells * sizeof(float));
        for (int r = start; r < stop; ++r) {
            std::memcpy(local.data() + static_cast<size_t>(r - start) * cells, a.data(), cells * sizeof(float));
            const float* e = errors.data() + ro + static_cast<size_t>(r) * c.dim;
            advance(keys.data() + ko + static_cast<size_t>(r) * c.hidden, e,
                    error_scale(e, c.dim, c.max_norm), a.data(), m.data(), c);
        }
        for (int r = stop - 1; r >= start; --r) {
            const float* k = keys.data() + ko + static_cast<size_t>(r) * c.hidden;
            const size_t row = ro + static_cast<size_t>(r) * c.dim;
            const float* e = errors.data() + row;
            const float* state = local.data() + static_cast<size_t>(r - start) * cells;
            std::fill(q.begin(), q.end(), 0);
            for (int h = 0; h < c.hidden; ++h) for (int d = 0; d < c.dim; ++d) {
                const size_t i = static_cast<size_t>(h) * c.dim + d;
                const double next_m_bar = adj_m[i] - (c.hamiltonian ? 0.25 * c.step * adj_a[i] : 0);
                gbar[i] = -c.step * adj_a[i] + (c.hamiltonian ? (1.0f - c.decay) * next_m_bar : next_m_bar);
                adj_m[i] = c.decay * next_m_bar;
                q[d] += gbar[i] * k[h];
            }
            float n = 0;
            const float scale = error_scale(e, c.dim, c.max_norm, &n);
            double dot = 0;
            for (int d = 0; d < c.dim; ++d) dot += q[d] * e[d];
            const double beta = c.max_norm > 0 && n > c.max_norm
                ? static_cast<double>(scale) * dot / (static_cast<double>(n) * (n + static_cast<double>(1e-6f))) : 0;
            for (int d = 0; d < c.dim; ++d) {
                const double de = scale * q[d] - beta * e[d];
                total[d] = grad.data()[row + d] + de;
                grad_base.data()[row + d] = static_cast<float>(total[d]);
                grad_direct.data()[row + d] = static_cast<float>(-de);
            }
            for (int h = 0; h < c.hidden; ++h) {
                double dk = 0;
                for (int d = 0; d < c.dim; ++d) {
                    const size_t i = static_cast<size_t>(h) * c.dim + d;
                    dk += total[d] * state[i] + scale * gbar[i] * e[d];
                    adj_a[i] = (c.hamiltonian ? adj_a[i] : c.decay * adj_a[i]) + k[h] * total[d];
                }
                grad_keys.data()[ko + static_cast<size_t>(r) * c.hidden + h] = static_cast<float>(dk);
            }
        }
    }
}
} // namespace nsos::ttt
