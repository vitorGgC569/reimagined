#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <atomic>
#include <limits>
#include <stdexcept>

namespace nsos {

// Enum for Fast Access (O(1))
enum class CtxKey : int {
    NONE = 0,
    INPUT_IDS,
    // Attention Keys
    MLA_Q_NOPE, MLA_Q_ROPE, MLA_K_NOPE, MLA_K_ROPE, MLA_V, MLA_INPUT, MLA_WEIGHTS,
    // MoE Keys
    MOE_IDX, MOE_W, MOE_SG, MOE_SU,
    // Mamba Keys
    MAMBA_X, MAMBA_DT, MAMBA_B, MAMBA_C, MAMBA_Z, MAMBA_HISTORY,
    // SwiGLU Keys
    SWIGLU_GATE, SWIGLU_UP,
    // Layer/Block Keys (Reserved Logic needed for multiple layers)
    MAX_KEYS
};

class AbortException : public std::exception {
public:
    const char* what() const noexcept override {
        return "Inferência abortada via AbortController (Signal Atômico marcado como false/true)";
    }
};

// Layer-aware Context
// We treat keys as (LayerIdx << 16) | KeyEnum
class Context {
public:
    // Sparse storage: Map int -> Tensor is faster than string, 
    // but Vector is fastest if dense. Since it's sparse per layer, 
    // we use a vector of vectors or a flat map with int key.
    // Optimization: Flat vector with direct indexing if we limit max layers.
    
    // Let's use a simple robust approach: Layer-Stride Vector.
    // Index = LayerID * MAX_KEYS + KeyID.
    
    std::vector<Tensor> storage;
    int max_layers;
    std::atomic<bool>* abort_signal = nullptr;

    Context(int layers = 128) : max_layers(layers) {
        storage.resize(max_layers * (int)CtxKey::MAX_KEYS);
    }

    void save(int layer_idx, CtxKey key, const Tensor& t) {
        if (key == CtxKey::NONE) return;
        if (layer_idx < 0 || layer_idx >= max_layers) {
            throw std::out_of_range("Context layer index out of range");
        }
        const size_t stride = static_cast<size_t>(CtxKey::MAX_KEYS);
        const size_t idx = static_cast<size_t>(layer_idx) * stride +
                           static_cast<size_t>(key);
        if (idx >= storage.size()) {
            throw std::out_of_range("Context storage index overflow");
        }
        storage[idx] = t; // Tensor is shared_ptr, cheap copy
    }

    Tensor get(int layer_idx, CtxKey key) {
        if (layer_idx < 0 || layer_idx >= max_layers) {
            return Tensor();
        }
        const size_t stride = static_cast<size_t>(CtxKey::MAX_KEYS);
        const size_t idx = static_cast<size_t>(layer_idx) * stride +
                           static_cast<size_t>(key);
        if (idx >= storage.size() || storage[idx].size == 0) 
            return Tensor(); // Return empty if not found
        return storage[idx];
    }
    
    // Legacy String Support (Wrapper for backward compat if needed, but deprecated)
    // We enforce integer keys for performance.
    
    // Helper helpers
    [[deprecated("Use integer CtxKey-based Context::save instead of string keys")]]
    void save(const std::string& s, const Tensor& t) {
        (void)s;
        (void)t;
        throw std::runtime_error("String-based Context::save is no longer supported");
    }
    [[deprecated("Use integer CtxKey-based Context::get instead of string keys")]]
    Tensor get(const std::string& s) {
        (void)s;
        throw std::runtime_error("String-based Context::get is no longer supported");
    }
};

} // namespace nsos
