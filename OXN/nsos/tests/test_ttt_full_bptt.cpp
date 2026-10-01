#include "ttt_layer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace nsos;
namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void set_full(bool enabled) {
#ifdef _WIN32
    _putenv_s("NSOS_TTT_FULL_BPTT", enabled ? "1" : "0");
#else
    setenv("NSOS_TTT_FULL_BPTT", enabled ? "1" : "0", 1);
#endif
}
void close(double a, double b, const std::string& name, double atol = 3e-4, double rtol = 2e-3) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > atol + rtol * std::max(std::abs(a), std::abs(b)))
        throw std::runtime_error(name + " analytic=" + std::to_string(a) + " reference=" + std::to_string(b));
}
void configure(TTTLayer& layer, bool exact, bool ham, bool clip) {
    for (BitLinear* linear : {layer.w_k.get(), layer.w_v.get(), layer.w_out.get()}) {
        linear->set_reference_path(true);
        linear->set_exact_linear_mode(exact);
        for (int i = 0; i < linear->weight.data.size; ++i)
            linear->weight.data.data()[i] = 0.035f * static_cast<float>((i * 7 + 3) % 13 - 6);
        for (int i = 0; i < linear->bias.data.size; ++i)
            linear->bias.data.data()[i] = 0.02f * static_cast<float>((i + 2) % 5 - 2);
    }
    layer.set_use_hamiltonian(ham);
    layer.set_friction(0.73f);
    layer.set_max_grad_norm(clip ? 0.02f : 0.0f);
    layer.set_temperature(0.8f);
}

// Independent double primal: no Tensor operations, recurrence helpers, saved
// histories, or implementation VJP. Includes RMS/magnitude compatibility mode.
std::vector<double> linear(const BitLinear& op, const std::vector<double>& input) {
    const int in = op.input_features(), out = op.output_features();
    double inv = 1;
    if (!op.exact_linear_mode()) {
        double sum = 0;
        for (double value : input) sum += value * value;
        inv = 1 / std::sqrt(sum / in + 1e-6);
    }
    std::vector<double> result(out);
    for (int o = 0; o < out; ++o) {
        double sum = 0;
        for (int i = 0; i < in; ++i) sum += input[i] * inv * op.weight.data.data()[o * in + i];
        result[o] = sum * (op.exact_linear_mode() ? 1 : op.magnitude.data.data()[o]) + op.bias.data.data()[o];
    }
    return result;
}
double objective(const TTTLayer& layer, const std::vector<double>& x, const Tensor& dy,
                 const TTTSessionSnapshot& initial, int seq, bool ham, bool clip) {
    const int d = layer.dim, h = layer.hidden;
    std::vector<double> a(h * d), m(h * d);
    for (int i = 0; i < h * d; ++i) {
        a[i] = initial.adaptation.data()[i]; m[i] = initial.momentum.data()[i];
    }
    const double decay = 0.73f, step = static_cast<double>(layer.learning_rate) * 0.8f;
    double value = 0;
    for (int r = 0; r < seq; ++r) {
        std::vector<double> row(x.begin() + r * d, x.begin() + (r + 1) * d);
        const auto k = linear(*layer.w_k, row), v = linear(*layer.w_v, row);
        auto y = linear(*layer.w_out, v);
        std::vector<double> e(d);
        double norm = 0;
        for (int j = 0; j < d; ++j) {
            for (int i = 0; i < h; ++i) y[j] += k[i] * a[i * d + j];
            value += y[j] * dy.data()[r * d + j];
            e[j] = y[j] - row[j]; norm += e[j] * e[j];
        }
        norm = std::sqrt(norm);
        const double scale = clip && norm > static_cast<double>(0.02f) ? static_cast<double>(0.02f) / (norm + 1e-6f) : 1;
        for (int i = 0; i < h; ++i) for (int j = 0; j < d; ++j) {
            const int n = i * d + j;
            const double g = k[i] * e[j] * scale;
            m[n] = decay * m[n] + (ham ? (1 - decay) * g : g);
            a[n] = ham ? a[n] - step * (g + 0.25 * m[n]) : decay * a[n] - step * g;
        }
    }
    return value;
}

void finite_differences(bool exact, bool ham, bool clip, int seq) {
    TTTLayer layer(4, 3, 0.07f, 91);
    configure(layer, exact, ham, clip);
    for (int i = 0; i < layer.grad_accum_.size; ++i) {
        layer.grad_accum_.data()[i] = 0.01f * (i % 3 - 1);
        layer.momentum_.data()[i] = 0.02f * (i % 5 - 2);
    }
    const auto initial = layer.snapshot_state();
    Tensor x({seq, 4}), dy({seq, 4});
    std::vector<double> xd(x.size);
    for (int i = 0; i < x.size; ++i) {
        x.data()[i] = -0.43f + 0.033f * (i % 29);
        xd[i] = x.data()[i]; dy.data()[i] = 0.13f - 0.017f * (i % 17);
    }
    const Tensor y = layer.forward(x);
    double primal = 0;
    for (int i = 0; i < y.size; ++i) primal += static_cast<double>(y.data()[i]) * dy.data()[i];
    close(primal, objective(layer, xd, dy, initial, seq, ham, clip), "independent primal", 2e-5, 1e-5);
    const size_t expected = 2 * static_cast<size_t>(1 + (seq - 1) / 32) * 3 * 4 * sizeof(float);
    require(layer.saved_state_history_bytes() == expected, "TTT retained dense rather than boundary history");
    // Policy/hyperparameters changed AFTER forward must not change its VJP.
    layer.set_friction(0.1f); layer.set_max_grad_norm(0); layer.set_use_hamiltonian(!ham);
    set_full(false);
    const Tensor dx = layer.backward(dy);
    set_full(true);
    require(layer.saved_state_history_bytes() == 0, "TTT did not release consumed history");
    for (int i = 0; i < x.size; ++i) {
        const double original = xd[i], eps = 1e-5;
        xd[i] = original + eps; const double plus = objective(layer, xd, dy, initial, seq, ham, clip);
        xd[i] = original - eps; const double minus = objective(layer, xd, dy, initial, seq, ham, clip);
        xd[i] = original;
        close(dx.data()[i], (plus - minus) / (2 * eps), "input " + std::to_string(i));
    }
    for (auto* parameter : layer.parameters()) if (parameter->trainable) {
        require(parameter->grad.size == parameter->data.size, "missing parameter gradient " + parameter->name);
        for (int i = 0; i < parameter->data.size; ++i) {
            const float original = parameter->data.data()[i];
            parameter->data.data()[i] = original + 1e-4f;
            const double high = parameter->data.data()[i], plus = objective(layer, xd, dy, initial, seq, ham, clip);
            parameter->data.data()[i] = original - 1e-4f;
            const double low = parameter->data.data()[i], minus = objective(layer, xd, dy, initial, seq, ham, clip);
            parameter->data.data()[i] = original;
            close(parameter->grad.data()[i], (plus - minus) / (high - low), parameter->name + ":" + std::to_string(i));
        }
    }
    if (exact && !ham && !clip && seq == 65) {
        TTTLayer old(4, 3, 0.07f, 91); configure(old, exact, ham, clip);
        old.restore_state(initial); set_full(false);
        (void)old.forward(x); const Tensor truncated = old.backward(dy); set_full(true);
        float gap = 0;
        for (int i = 0; i < dx.size; ++i) gap = std::max(gap, std::abs(dx.data()[i] - truncated.data()[i]));
        require(gap > 1e-4f, "full BPTT oracle does not distinguish the old truncated derivative");
    }
    bool consumed = false;
    try { (void)layer.backward(dy); } catch (const std::runtime_error&) { consumed = true; }
    require(consumed, "TTT accepted a second backward without a matching forward");
}

void batch_isolation() {
    TTTLayer layer(4, 3, 0.03f, 81), single(4, 3, 0.03f, 81);
    configure(layer, true, true, true); configure(single, true, true, true);
    Tensor x({3, 33, 4}), dy({3, 33, 4});
    for (int i = 0; i < x.size; ++i) { x.data()[i] = 0.02f * (i % 31 - 15); dy.data()[i] = 0.1f; }
    layer.set_batch_valid_lengths({33, 29, 0});
    std::fill_n(layer.grad_accum_.data(), layer.grad_accum_.size, 0.35f);
    std::fill_n(layer.momentum_.data(), layer.momentum_.size, -0.25f);
    const auto state = layer.snapshot_state();
    const Tensor y = layer.forward(x), dx = layer.backward(dy);
    for (int i = 0; i < layer.grad_accum_.size; ++i)
        close(layer.grad_accum_.data()[i], state.adaptation.data()[i], "batch mutated serving state", 0, 0);
    for (int b = 0; b < 2; ++b) {
        single.reset();
        const int length = b == 0 ? 33 : 29;
        Tensor row({length, 4});
        std::copy_n(x.data() + b * 33 * 4, row.size, row.data());
        const Tensor expected = single.forward(row);
        Tensor row_grad({length, 4}); std::fill_n(row_grad.data(), row_grad.size, 0.1f);
        const Tensor expected_grad = single.backward(row_grad);
        for (int i = 0; i < row.size; ++i) {
            close(y.data()[b * 33 * 4 + i], expected.data()[i], "batch isolation", 0, 0);
            close(dx.data()[b * 33 * 4 + i], expected_grad.data()[i], "batch gradient isolation");
        }
    }
    const auto bp = layer.parameters(), sp = single.parameters();
    for (size_t p = 0; p < bp.size(); ++p) if (bp[p]->trainable)
        for (int i = 0; i < bp[p]->grad.size; ++i)
            close(bp[p]->grad.data()[i], sp[p]->grad.data()[i], "batch summed parameter gradient");
    for (int b = 1; b < 3; ++b) for (int t = b == 1 ? 29 : 0; t < 33; ++t) for (int d = 0; d < 4; ++d) {
        const int i = (b * 33 + t) * 4 + d;
        require(y.data()[i] == 0 && dx.data()[i] == 0, "padding affected TTT output/gradient");
    }
    Tensor swapped({3, 33, 4});
    std::copy_n(x.data() + 33 * 4, 33 * 4, swapped.data());
    std::copy_n(x.data(), 33 * 4, swapped.data() + 33 * 4);
    std::copy_n(x.data() + 66 * 4, 33 * 4, swapped.data() + 66 * 4);
    layer.set_batch_valid_lengths({29, 33, 0});
    const Tensor permuted = layer.forward(swapped);
    for (int i = 0; i < 33 * 4; ++i) {
        close(permuted.data()[i], y.data()[33 * 4 + i], "batch permutation", 0, 0);
        close(permuted.data()[33 * 4 + i], y.data()[i], "batch permutation", 0, 0);
    }
    auto bad = state; bad.adaptation = Tensor({1, 4});
    bool rejected = false;
    try { layer.restore_state(bad); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid session shape accepted");
    rejected = false;
    try { layer.set_friction(std::numeric_limits<float>::quiet_NaN()); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "NaN friction accepted");
}
} // namespace
int main() {
    try {
        set_full(true);
        for (bool exact : {false, true}) for (bool ham : {false, true})
            for (bool clip : {false, true}) for (int seq : {3, 33, 65}) finite_differences(exact, ham, clip, seq);
        batch_isolation();
        set_full(false);
        std::cout << "TTT full sequence BPTT: independent double primal, dInput/all trainable parameters, chunk tails, isolation/padding passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
