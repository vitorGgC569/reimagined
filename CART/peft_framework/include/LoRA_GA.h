#ifndef LORA_GA_H
#define LORA_GA_H

#include "LoRA.h"

/**
 * @class LoRA_GA
 * @brief Implements LoRA with Gradient Approximation.
 *
 * This variant initializes the LoRA matrices (A and B) to approximate the
 * gradients of the full model at the beginning of training. This aligns the
 * optimization trajectory and can lead to faster convergence.
 */
class LoRA_GA : public LoRALayer {
public:
    /**
     * @brief Constructs a LoRA_GA layer.
     * @param input_dims The number of input dimensions.
     * @param output_dims The number of output dimensions.
     * @param rank The rank of the adaptation matrices.
     */
    LoRA_GA(int input_dims, int output_dims, int rank);

    /**
     * @brief Initializes matrices A and B using a low-rank approximation
     *        of the initial full-model gradients.
     * @param full_gradient A Tensor representing the full gradient.
     */
    void initializeWithGradients(const Tensor& full_gradient);
};

#endif  // LORA_GA_H
