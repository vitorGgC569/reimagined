#include "../OXN/nsos/include/jamba.h"
#include "../OXN/nsos/include/tokenizer.h"
#include "../OXN/nsos/include/tensor.h"
#include "../OXN/nsos/include/nsos_sdk.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

// Mock Teacher in C++
class TeacherModelCPP {
public:
    int vocab;
    int dim;
    Tensor W;

    TeacherModelCPP(int v, int d) : vocab(v), dim(d) {
        W = Tensor::random({dim, vocab}, Device::CPU).mul(0.1f);
    }

    Tensor forward(const Tensor& x) {
        // x: [B, S, D]
        // W: [D, V]
        // out: x @ W -> [B, S, V]
        return x.matmul(W);
    }
};

// Loss Function
std::pair<float, Tensor> distillation_loss_grad(const Tensor& s_logits, const Tensor& t_logits, float T=2.0f) {
    // Softmax
    Tensor p_s = s_logits.mul(1.0f/T).softmax(-1);
    Tensor p_t = t_logits.mul(1.0f/T).softmax(-1);

    // Grad: (p_s - p_t) / T
    Tensor grad = p_s.sub(p_t).mul(1.0f/T);

    // Loss: sum(p_t * (log p_t - log p_s))
    // Approx calc for logging
    float loss = 0;
    // ... skipping scalar loss calc for perf in C++ loop

    return {0.0f, grad};
}

int main() {
    std::cout << "=== 🏭 Industrial Training (Pure C++) ===" << std::endl;

    int vocab_size = 1000;
    int dim = 64;
    int layers = 2;
    float lr = 0.01f;

    // 1. Init
    JambaModel student(layers, dim, vocab_size);
    TeacherModelCPP teacher(vocab_size, dim);
    Tokenizer tokenizer; // Mock or load

    // 2. Data
    std::vector<std::string> dataset = {
        "science is cool",
        "physics rules",
        "math is hard"
    };

    std::cout << "[Start] Loop..." << std::endl;

    for(int step=0; step<50; ++step) {
        std::string text = dataset[step % dataset.size()];

        // Encode
        // Basic mock encoding if tokenizer empty
        std::vector<int> ids;
        for(char c : text) ids.push_back((int)c % vocab_size);
        if(ids.empty()) continue;

        // Forward
        Tensor emb = student.embedding->forward(ids); // Access via pointer

        std::vector<int> shape3d = {1, (int)ids.size(), dim};
        emb = emb.reshape(shape3d);

        Context ctx;
        student.set_hamiltonian_mode(step % 5 == 0);

        Tensor hidden = student.forward(emb, &ctx);

        // Teacher
        // We need logits.
        Tensor w_student = student.embedding->weight.data;
        // student.embedding.weight is Parameter. data is Tensor.
        // weight is [V, D].
        // hidden: [1, S, D].
        // logits = hidden @ W.T
        Tensor logits_s = hidden.matmul(w_student.transpose());
        Tensor logits_t = teacher.forward(hidden);

        // Loss
        auto [loss_val, grad_logits] = distillation_loss_grad(logits_s, logits_t);

        // Backward
        // dL/dH = dL/dLogits @ W
        Tensor grad_h = grad_logits.matmul(w_student);

        student.backward(grad_h, ctx);

        // Update
        auto params = student.parameters();
        for(auto* p : params) {
            // p->data -= lr * p->grad
            // Manual loop or Tensor op
            int sz = p->data.size;
            float* d = p->data.data();
            float* g = p->grad.data();
            for(int i=0; i<sz; ++i) {
                d[i] -= lr * g[i];
                g[i] = 0.0f;
            }
        }

        if (step % 10 == 0) std::cout << "Step " << step << " Complete." << std::endl;
    }

    std::cout << "=== ✅ C++ Training Complete ===" << std::endl;
    return 0;
}
