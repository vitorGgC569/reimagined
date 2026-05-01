#include "TransMamba.h"
#include <iostream>

void TransMamba::weightSubcloning(SSMLayer& ssm_layer, const LoRALayer& lora_layer) {
    std::cout << "Performing Weight Subcloning..." << std::endl;

    const Tensor* lora_A = lora_layer.get_A();
    Tensor* ssm_B = ssm_layer.get_B();

    // --- Weight Mapping ---
    // A simplified mapping where the LoRA 'A' matrix (rank x input_dim)
    // is used to initialize the SSM 'B' matrix (state_dim x input_dim).
    // This requires the SSM's state_dim to be equal to the LoRA's rank.

    if (lora_A->getRows() != ssm_B->getRows() || lora_A->getCols() != ssm_B->getCols()) {
        std::cerr << "Warning: Dimensions of LoRA A and SSM B do not match for subcloning. Skipping." << std::endl;
        return;
    }

    *ssm_B = *lora_A;

    std::cout << "Weight Subcloning complete." << std::endl;
}
