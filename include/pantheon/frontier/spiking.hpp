#pragma once

#include <vector>
#include <cmath>

namespace pantheon {
namespace frontier {

    class SpikingDistance {
    public:
        // Van Rossum Distance for Spike Trains
        // Convolves spike train with exponential kernel and computes Euclidean distance.
        // trains: binary vectors (1=spike, 0=silence)
        // tau: time constant

        static float compute_spike_loss(const std::vector<int>& s_spikes,
                                      const std::vector<int>& t_spikes,
                                      float tau = 5.0f) {

            auto filter = [&](const std::vector<int>& spikes) {
                std::vector<float> continuous(spikes.size(), 0.0f);
                float val = 0.0f;
                for (size_t i = 0; i < spikes.size(); ++i) {
                    // Exponential decay
                    val = val * std::exp(-1.0f / tau);
                    if (spikes[i]) val += 1.0f; // Spike adds charge
                    continuous[i] = val;
                }
                return continuous;
            };

            auto s_filt = filter(s_spikes);
            auto t_filt = filter(t_spikes);

            float loss = 0.0f;
            for (size_t i = 0; i < s_filt.size(); ++i) {
                float diff = s_filt[i] - t_filt[i];
                loss += diff * diff;
            }

            return loss; // L2 on filtered traces
        }
    };

}
}
