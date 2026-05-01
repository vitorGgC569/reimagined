#pragma once

#include <vector>
#include <cmath>
#include <complex>

namespace pantheon {
namespace frontier {

    class QuantumKernel {
    public:
        // Quantum Fidelity Loss
        // Matches the quantum state |psi_S> with |psi_T>
        // Fidelity F = |<psi_S|psi_T>|^2
        // Loss = 1 - F
        // Inputs are real vectors representing complex amplitudes (Real, Imag, Real, Imag...)

        static float compute_fidelity_loss(const std::vector<float>& s_state, const std::vector<float>& t_state) {
            float real_dot = 0.0f;
            float imag_dot = 0.0f; // Contribution to inner product

            // Assume normalized states encoded as [r0, i0, r1, i1, ...]
            size_t n = s_state.size();
            if (n % 2 != 0) return 1.0f; // Error

            for (size_t k = 0; k < n; k += 2) {
                float sr = s_state[k];
                float si = s_state[k+1];
                float tr = t_state[k];
                float ti = t_state[k+1];

                // <s|t> = sum (sr - i*si)(tr + i*ti)
                //       = sum (sr*tr + si*ti) + i(sr*ti - si*tr)

                real_dot += sr * tr + si * ti;
                imag_dot += sr * ti - si * tr;
            }

            float fidelity = real_dot * real_dot + imag_dot * imag_dot;
            return 1.0f - fidelity;
        }
    };

}
}
