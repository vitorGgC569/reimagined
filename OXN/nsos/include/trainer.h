#pragma once
#include "autograd.h"
#include "jamba.h"
#include <vector>
#include <functional>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

class Trainer {
public:
    JambaModel* model;
    float learning_rate;
    
    #ifdef USE_CUDA
    cudaStream_t compute_stream;
    cudaStream_t copy_stream;
    #endif

    Trainer(JambaModel* m, float lr = 0.001f);
    ~Trainer();
    
    float train_step(const std::vector<int>& tokens, const std::vector<int>& targets);
    void train_loop(const std::vector<int>& tokens, int epochs, int batch_size, int seq_len, std::function<void(int, float)> callback = nullptr);
};

} // namespace nsos
