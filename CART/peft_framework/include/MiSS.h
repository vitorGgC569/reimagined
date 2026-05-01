#ifndef MISS_H
#define MISS_H

#include "LoRA.h"
#include <map>

/**
 * @class MiSS
 * @brief Implements Multiple-stage importance-aware Sparse Sharing.
 *
 * This technique dynamically allocates the rank of the LoRA matrices based on
 * importance scores, typically derived from the singular value spectrum of the
 * gradients. This allows for a more surgical allocation of model capacity
 * during fine-tuning.
 */
class MiSS : public LoRALayer {
public:
    /**
     * @brief Constructs a MiSS layer.
     * @param input_dims The number of input dimensions.
     * @param output_dims The number of output dimensions.
     * @param initial_rank The initial rank of the adaptation matrices.
     */
    MiSS(int input_dims, int output_dims, int initial_rank);

    /**
     * @brief Adjusts the rank of the LoRA matrices based on gradient importance.
     * @param gradient The gradient tensor of the layer's weights.
     * @param threshold A threshold to decide singular value importance.
     */
    void updateRank(const Tensor& gradient, float threshold = 0.1f);

    /**
     * @brief Creates a curriculum for rank allocation across multiple layers
     * (placeholder).
     * @param num_layers The total number of layers in the model.
     * @return A map from layer index to allocated rank.
     */
    static std::map<int, int> createRankSchedule(int num_layers);

private:
    /**
     * @brief Resizes the adaptation matrices A and B to a new rank.
     * @param new_rank The new rank.
     */
    void resizeMatrices(int new_rank);
};

#endif  // MISS_H
