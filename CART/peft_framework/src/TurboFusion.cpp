#include "../include/TurboFusion.h"
#include "../include/TensorOps.h"
#include <iostream>

TurboFusionLayer::TurboFusionLayer(int input_dims, int output_dims, int rank) {
    // Initialize the DoRA layer (Core Engine)
    m_dora = std::make_unique<DoRALayer>(input_dims, output_dims, rank);

    // Initialize the IA3 layer (Scaling Engine)
    // IA3 is usually applied to the output of a linear layer.
    // Here we apply it to the output of DoRA.
    m_ia3 = std::make_unique<IA3Layer>(input_dims, output_dims);
}

void TurboFusionLayer::setBaseWeights(const Tensor& weights) {
    // Both components might need base weights information
    // DoRA definitely needs it. IA3 usually just needs dimensions which it has.
    // However, IA3 implementation takes base weights if we were wrapping a linear layer directly.
    // Since we are wrapping DoRA output, IA3's "base weights" concept is slightly different here.
    // Our IA3 implementation expects setBaseWeights to compute the base pass XW0.
    // BUT here, DoRA computes the full XW' (base + adapter).
    // So we need to treat IA3 here as just the *scaling* part.

    // Let's look at IA3 implementation:
    // forward() calculates Y_pre = X * W0^T, then scales.
    // We want Y_final = (DoRA_Output) * L.
    // We can't use IA3::forward directly because it re-computes X*W0.
    // We need to modify IA3 or manually implement the scaling here using IA3's L vector.

    // For "TurboFusion", it's cleaner to implement the scaling logic here
    // reusing the L vector from the IA3 class if possible, or just manage L here.
    // To keep it modular, let's assume we use IA3 class just for its parameter L and update logic?
    // Or we use IA3 as intended:
    // If we pass Identity matrix as Base Weights to IA3, then IA3::forward(DoRA_Output) = DoRA_Output * I * L = DoRA_Output * L.
    // Yes! That's a clever way to re-use the class without modifying it.

    m_dora->setBaseWeights(weights);

    // Create Identity weights for IA3
    // IA3 takes input (batch, output_dim) now because input to IA3 is output of DoRA.
    // Wait, IA3 constructor takes (in, out).
    // Here input to IA3 is DoRA output (batch, out).
    // So IA3 input_dim = output_dim of layer.
    // And IA3 output_dim = output_dim of layer.
    // So we need an Identity matrix of size (out, out).

    // Re-creating IA3 with correct dims for this chaining
    // The previous constructor call `m_ia3 = std::make_unique<IA3Layer>(input_dims, output_dims);` was wrong for chaining.
    // It should be (output_dims, output_dims).

    int out_dim = weights.getRows(); // output_dims
    m_ia3 = std::make_unique<IA3Layer>(out_dim, out_dim);

    Tensor identity(out_dim, out_dim);
    for(int i=0; i<out_dim; ++i) {
        for(int j=0; j<out_dim; ++j) {
            identity.at(i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }
    m_ia3->setBaseWeights(identity);
}

Tensor TurboFusionLayer::forward(const Tensor& input) {
    // 1. DoRA Forward
    Tensor y_dora = m_dora->forward(input);

    // 2. IA3 Forward (Scaling)
    // We pass y_dora as "input" to IA3.
    // Since IA3 has Identity base weights, it effectively just does y_dora * L.
    return m_ia3->forward(y_dora);
}

void TurboFusionLayer::backward(const Tensor& upstream_grad) {
    // Backprop through IA3 first
    m_ia3->backward(upstream_grad);

    // We need the gradient w.r.t the INPUT of IA3.
    // Our IA3::backward computes grad_L. It doesn't currently return dL/dInput (dL/dX).
    // We need to check IA3 implementation.
    // "We don't compute grad_X ... For now, focusing on the parameter gradient."

    // CRITICAL: For TurboFusion chaining to work, IA3 MUST return dL/dInput to pass it to DoRA.
    // I must update IA3 first to compute and return input gradients, or calculate it here.
    // dL/dInput_IA3 = dL/dOutput * L.
    // Since IA3 is element-wise scaling: dL/dx_i = dL/dy_i * L_i.

    // Let's implement this calculation here since IA3 class doesn't expose it.
    // We need access to L. IA3 has get_L().

    const Tensor* L = m_ia3->get_L();
    Tensor grad_for_dora(upstream_grad.getRows(), upstream_grad.getCols());

    // dL/d(DoRA_Out) = upstream_grad * L
    for(int i=0; i<upstream_grad.getRows(); ++i) {
        for(int j=0; j<upstream_grad.getCols(); ++j) {
            grad_for_dora.at(i, j) = upstream_grad.at(i, j) * L->at(0, j);
        }
    }

    // Backprop through DoRA
    m_dora->backward(grad_for_dora);
}

void TurboFusionLayer::update(float learning_rate) {
    m_dora->update(learning_rate);
    m_ia3->update(learning_rate);
}
