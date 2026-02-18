#ifndef TRANSMAMBA_H
#define TRANSMAMBA_H

#include "LoRA.h"
#include "SSM.h"

/**
 * @class TransMamba
 * @brief Orchestrates the cross-architecture transfer from a Transformer-like
 *        layer to an SSM layer.
 */
class TransMamba {
public:
    /**
     * @brief Performs Weight Subcloning to initialize an SSM layer.
     *
     * This method uses the weights from a pre-trained LoRA layer (acting as a
     * proxy for a Transformer's attention block) to provide a "warm start"
     * for a new SSM layer.
     *
     * @param ssm_layer The SSM layer to be initialized.
     * @param lora_layer The pre-trained LoRA layer to source weights from.
     */
    static void weightSubcloning(SSMLayer& ssm_layer, const LoRALayer& lora_layer);
};

#endif // TRANSMAMBA_H
