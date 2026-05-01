#ifndef TURBO_FUSION_H
#define TURBO_FUSION_H

#include "Tensor.h"
#include "DoRA.h"
#include "IA3.h"
#include <memory>
#include <vector>

// TurboFusion: A hybrid architecture combining DoRA (Weight Decomposition)
// and IA3 (Learned Vector Scaling).
//
// Logic:
// 1. Apply DoRA: Y_dora = X * (m * V/||V||)^T
// 2. Apply IA3 Scaling: Y_final = Y_dora * L
//
// This allows the model to learn both subspace adaptations (LoRA/DoRA)
// and activation scaling (IA3), effectively acting as a superset of both.

class TurboFusionLayer {
public:
    TurboFusionLayer(int input_dims, int output_dims, int rank);
    void setBaseWeights(const Tensor& weights);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& upstream_grad);
    void update(float learning_rate);

    // Access to underlying components
    const DoRALayer& getDoRA() const { return *m_dora; }
    const IA3Layer& getIA3() const { return *m_ia3; }

private:
    std::unique_ptr<DoRALayer> m_dora;
    std::unique_ptr<IA3Layer> m_ia3;

    // Intermediate storage for backward pass
    std::unique_ptr<Tensor> m_dora_output;
};

#endif // TURBO_FUSION_H
