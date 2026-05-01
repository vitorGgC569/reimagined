#ifndef SPIKE_ENCODER_H
#define SPIKE_ENCODER_H

#include <vector>
#include <cmath>

namespace uhk {
namespace bci {

// Leaky Integrate-and-Fire (LIF) Encoder
class SpikeEncoder {
public:
    SpikeEncoder(float threshold, float decay) : threshold_(threshold), decay_(decay), membrane_potential_(0.0f) {}

    bool step(float input_current) {
        membrane_potential_ = membrane_potential_ * decay_ + input_current;

        if (membrane_potential_ >= threshold_) {
            membrane_potential_ = 0.0f; // Reset
            return true; // Spike
        }
        return false; // No spike
    }

    std::vector<uint8_t> encode_signal(const std::vector<float>& signal) {
        std::vector<uint8_t> spikes;
        for (float val : signal) {
            spikes.push_back(step(val) ? 1 : 0);
        }
        return spikes;
    }

private:
    float threshold_;
    float decay_;
    float membrane_potential_;
};

} // namespace bci
} // namespace uhk

#endif // SPIKE_ENCODER_H
