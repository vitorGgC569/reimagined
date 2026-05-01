#ifndef BOFT_H
#define BOFT_H

#include "LoRA.h"

/**
 * @class BOFT
 * @brief Implements Butterfly Orthogonal Fine-Tuning.
 *
 * BOFT imposes an orthogonality constraint on the adaptation matrices, often
 * parameterized using efficient structures like butterfly factorizations. This
 * preserves the geometry of the latent space, mitigating catastrophic
 * forgetting and improving stability in continual learning scenarios.
 */
class BOFT : public LoRALayer {
public:
    /**
     * @brief Constructs a BOFT layer.
     * @param input_dims The number of input dimensions.
     * @param output_dims The number of output dimensions.
     * @param rank The rank of the adaptation matrices.
     */
    BOFT(int input_dims, int output_dims, int rank);

    /**
     * @brief Overrides the forward pass to signify the use of orthogonal
     * matrices.
     * @param input The input Tensor.
     * @return The output Tensor.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Simulates an update step that preserves matrix orthogonality.
     */
    void orthogonalUpdateStep();

private:
    /**
     * @brief Placeholder for applying the butterfly structure to the
     * adaptation matrices.
     */
    void applyButterflyStructure();
};

#endif  // BOFT_H
