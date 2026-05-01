#include "optimizers.h"
#include <cmath>
#include <algorithm>

namespace nsos {

AdamWOptimizer::AdamWOptimizer(float lr, float b1, float b2, float e, float wd)
    : learning_rate(lr), beta1(b1), beta2(b2), eps(e), weight_decay(wd), t(0) {}

void AdamWOptimizer::step(std::vector<Parameter*> params) {
    t++;
    float bc1 = 1.0f - std::pow(beta1, (float)t);
    float bc2 = 1.0f - std::pow(beta2, (float)t);
    
    for(auto* p : params) {
        if(p->grad.size == 0) continue;
        
        float* d = p->data.data();
        float* g = p->grad.data();
        
        if(m_states.find(p->name) == m_states.end()) {
            m_states[p->name] = Tensor::zeros(p->data.shape.dims, p->data.device);
            v_states[p->name] = Tensor::zeros(p->data.shape.dims, p->data.device);
        }
        
        float* m = m_states[p->name].data();
        float* v = v_states[p->name].data();
        
        #pragma omp parallel for
        for(int i=0; i<p->data.size; ++i) {
            d[i] -= learning_rate * weight_decay * d[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
            float m_h = m[i] / bc1;
            float v_h = v[i] / bc2;
            d[i] -= learning_rate * m_h / (std::sqrt(v_h) + eps);
        }
    }
}

} // namespace nsos
