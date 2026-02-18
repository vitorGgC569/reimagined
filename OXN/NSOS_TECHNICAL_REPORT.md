# NSOS Technical Report: The Omni-Efficient Architecture (v1.0)

## 1. Introduction
This report documents the v1.0 "Extremely Complete" implementation of the NSOS architecture. It synthesizes **Mamba-2 SSD**, **BitNet b1.58**, **Kolmogorov-Arnold Networks (KAN)**, and **Latent MCTS** into a fully differentiable, stable, and trainable system.

## 2. Core Backbone: Jamba Hybrid
The backbone combines Linear Recurrence with Attention:
*   **Mamba-2 SSD**: Uses Structured State Space Duality. We implemented **Real Backpropagation Through Time (BPTT)** by storing the full state history $h_t$ during the forward pass. The backward pass iterates time in reverse, correctly propagating gradients through the recurrence $h_t = h_{t-1} \cdot A + u_t \cdot B_t$ and the output $y_t = h_t \cdot C_t$. This enables end-to-end training of the SSD projections.
*   **Attention**: Standard Self-Attention blocks are interspersed (1:7 ratio). We implemented full backward differentiation for $Q, K, V, O$ projections using the Chain Rule.

## 3. BitNet b1.58 & BitFastKAN
*   **BitLinear**: Implements **Straight-Through Estimator (STE)**. Forward pass uses quantized weights $\{-1, 0, 1\}$, while backward pass updates the high-precision latent weights.
*   **BitFastKAN**: Implements a learnable feature extractor using **Radial Basis Function (RBF)** splines.
    *   Forward: $y = x \cdot W_{base} + \text{Spline}(x) \cdot W_{rbf}$.
    *   Backward: We derive gradients for $W_{base}$ (linear path) and $W_{rbf}$ (spline coefficients). Crucially, we also compute $\partial \text{Basis} / \partial x$, allowing gradients to flow back to the input, enabling stacking of KAN layers.

## 4. Reasoning: Latent MCTS
We integrate Monte Carlo Tree Search into the training loop via **Policy Distillation**:
*   The model predicts a policy $\pi_\theta(s)$.
*   MCTS performs a search to find a better policy $\pi_{MCTS}$.
*   Loss Function: $L = L_{CE} + \alpha \cdot KL(\pi_{MCTS} || \pi_\theta)$.
This effectively uses MCTS as a "teacher" for the neural network "student", imparting reasoning capabilities (System 2) into the fast weights (System 1).

## 5. Stability & Engineering
*   **Memory Safety**: We resolved critical heap corruption issues in the Python/C++ boundary (Pybind11) by strictly enforcing ownership policies (`std::shared_ptr`, `return_value_policy::reference`).
*   **Pipeline**: The training system includes:
    *   **Cosine Learning Rate Schedule** with Warmup.
    *   **Gradient Clipping** to prevent exploding gradients in deep recurrences.
    *   **Validation & Checkpointing** for reliable experimentation.

## 6. Conclusion
The NSOS v1.0 is no longer just a collection of architectural components but a **functional, stable, and trainable neural system**. It converges on sequence modeling tasks and possesses the infrastructure for advanced reasoning research.
