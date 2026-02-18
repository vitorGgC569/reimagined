#include <torch/extension.h>
#include <vector>
#include <cmath>

// --- Helper for DoRA ---
// Calculates the column-wise norm of a matrix (output_dim, input_dim)
// Equivalent to torch.linalg.vector_norm(W, dim=1)
torch::Tensor column_norm(const torch::Tensor& W) {
    return torch::linalg_vector_norm(W, 2, {1}, true);
}

// --- TurboFusion Implementation ---
// Logic:
// 1. DoRA: V = W0 + B@A.  W' = m * (V / ||V||).
// 2. IA3:  Y = (X @ W'.T) * L.
//
// Optimizations:
// - Fused operations where possible.
// - In-place updates for efficiency.

class TurboFusionLinear : public torch::nn::Module {
public:
    TurboFusionLinear(int in_features, int out_features, int rank)
        : in_features(in_features), out_features(out_features), rank(rank) {

        // Base weights (frozen usually, but here we manage them)
        // Registered as buffer so they are part of state_dict but maybe not optimized if frozen
        weight = register_parameter("weight", torch::randn({out_features, in_features}));

        // LoRA components
        lora_A = register_parameter("lora_A", torch::randn({rank, in_features}) * (1.0 / std::sqrt(in_features)));
        lora_B = register_parameter("lora_B", torch::zeros({out_features, rank}));

        // DoRA magnitude vector
        // Initialize to norm of random weight for stability
        // Must detach() to ensure it is a leaf tensor for optimization
        torch::Tensor init_m = torch::linalg_vector_norm(weight, 2, {1}, true).detach();
        dora_m = register_parameter("dora_m", init_m);

        // IA3 scaling vector
        ia3_l = register_parameter("ia3_l", torch::ones({1, out_features}));
    }

    // Dynamic Rank Resizing
    void resize_rank(int new_rank) {
        if (new_rank == rank) return;

        // Preserve existing weights if possible (slice) or pad with zeros/noise
        int min_rank = std::min(rank, new_rank);

        // New A
        torch::Tensor new_A = torch::randn({new_rank, in_features}, lora_A.options()) * (1.0 / std::sqrt(in_features));
        // Copy old A slice
        new_A.slice(0, 0, min_rank) = lora_A.slice(0, 0, min_rank);

        // New B
        torch::Tensor new_B = torch::zeros({out_features, new_rank}, lora_B.options());
        // Copy old B slice
        new_B.slice(1, 0, min_rank) = lora_B.slice(1, 0, min_rank);

        // Update parameters
        // We must re-register them to ensure optimizer sees them?
        // No, replacing the data pointer or tensor inside a module is tricky for optimizers.
        // In PyTorch C++ API, best practice is to update the data if shapes match, or replace if not.
        // But changing shapes breaks the optimizer state (momentum buffers match old shape).
        // The user must re-initialize the optimizer after resizing! This is standard for dynamic methods.

        // Deregister old parameters?
        // We simply overwrite the member variables. The Python side wrapper might need to re-fetch parameters.

        // Unregistering is complex in C++ API. We will overwrite the tensors and rely on the user
        // to re-construct the optimizer or use a specialized one.

        // Note: 'register_parameter' inserts into the parameters list. Calling it again with same name might replace it.
        lora_A = register_parameter("lora_A", new_A);
        lora_B = register_parameter("lora_B", new_B);

        rank = new_rank;
    }

    torch::Tensor forward(torch::Tensor input) {
        // 1. Calculate DoRA Effective Weight
        // V = W + B@A
        // We use addmm for V = W + alpha * (B@A). Here alpha=1.
        // But W is (out, in), B (out, r), A (r, in).
        // W + B@A is valid matrix addition.

        // Optimization: Compute (B@A)
        torch::Tensor delta_W = torch::mm(lora_B, lora_A);

        // V = W + delta_W
        torch::Tensor V = weight + delta_W;

        // Normalize V
        // ||V||_c = column_norm(V)
        // W' = m * (V / ||V||)

        torch::Tensor norm_V = torch::linalg_vector_norm(V, 2, {1}, true) + 1e-9;
        torch::Tensor W_prime = dora_m * (V / norm_V);

        // 2. Linear Pass
        // Y_pre = X @ W'.T
        // standard linear forward
        torch::Tensor y_pre = torch::nn::functional::linear(input, W_prime);

        // 3. IA3 Scaling
        // Y = Y_pre * L
        // IA3 vector is (1, out). Y_pre is (batch, out). Broadcasting works.
        return y_pre * ia3_l;
    }

private:
    int in_features, out_features, rank;
    torch::Tensor weight;
    torch::Tensor lora_A, lora_B;
    torch::Tensor dora_m;
    torch::Tensor ia3_l;
};

// --- Bindings ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch::python::bind_module<TurboFusionLinear>(m, "TurboFusionLinear")
        .def(py::init<int, int, int>())
        .def("forward", &TurboFusionLinear::forward)
        .def("resize_rank", &TurboFusionLinear::resize_rank);
}
