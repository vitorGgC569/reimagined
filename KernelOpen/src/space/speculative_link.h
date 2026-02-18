#ifndef SPECULATIVE_LINK_H
#define SPECULATIVE_LINK_H

#include <vector>
#include <deque>
#include <chrono>

namespace uhk {
namespace space {

struct Packet {
    uint64_t timestamp;
    std::vector<uint8_t> data;
};

// Simulated Time Warp Rollback Protocol for High-Latency Links
class SpeculativeLink {
public:
    SpeculativeLink(double latency_ms) : latency_ms_(latency_ms), local_time_(0) {}

    void send(const Packet& p) {
        // Speculatively assume it arrives instantly in the "future" buffer
        future_buffer_.push_back(p);
    }

    void receive_confirmation(uint64_t timestamp) {
        // If we received a confirmation from the past that contradicts our speculation, rollback
        // (Simplified simulation)
        if (!future_buffer_.empty() && future_buffer_.front().timestamp != timestamp) {
            rollback();
        } else {
             commit();
        }
    }

private:
    double latency_ms_;
    uint64_t local_time_;
    std::deque<Packet> future_buffer_;

    void rollback() {
        // Clear speculative state
        future_buffer_.clear();
    }

    void commit() {
        if (!future_buffer_.empty()) {
            future_buffer_.pop_front();
        }
    }
};

} // namespace space
} // namespace uhk

#endif // SPECULATIVE_LINK_H
