#ifndef LUT_CACHE_H
#define LUT_CACHE_H

#include <vector>
#include <cmath>
#include <algorithm>

// Fast Lookup Table for expensive functions (Exp, Sigmoid, Tanh)
class LUTCache {
private:
    std::vector<float> table;
    float min_val;
    float max_val;
    float step;
    float inv_step;
    int size;

public:
    LUTCache(float min_v, float max_v, int steps, float (*func)(float))
        : min_val(min_v), max_val(max_v), size(steps) {
        step = (max_val - min_val) / (steps - 1);
        inv_step = 1.0f / step;
        table.resize(steps);
        for(int i=0; i<steps; ++i) {
            table[i] = func(min_val + i * step);
        }
    }

    inline float eval(float x) const {
        if (x <= min_val) return table[0];
        if (x >= max_val) return table[size - 1];

        float pos = (x - min_val) * inv_step;
        int idx = (int)pos;
        float frac = pos - idx;

        // Linear interpolation
        return table[idx] * (1.0f - frac) + table[idx+1] * frac;
    }

    static LUTCache& get_exp() {
        static LUTCache cache(-10.0f, 10.0f, 4096, [](float x){ return std::exp(x); });
        return cache;
    }

    static LUTCache& get_sigmoid() {
        static LUTCache cache(-10.0f, 10.0f, 4096, [](float x){ return 1.0f / (1.0f + std::exp(-x)); });
        return cache;
    }
};

#endif
