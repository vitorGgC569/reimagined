#include "aion/RingBuffer.hpp"
#include <deque>
#include <iostream>

namespace Aion {

    // Simulation Backend (Windows / Fallback)
    struct SimEntry {
        int fd;
        void* buffer;
        size_t size;
        uint64_t offset;
        uint64_t user_data;
    };

    struct SimCompletion {
        uint64_t user_data;
        int res;
    };

    class RingBufferSim : public RingBufferBackend {
    private:
        std::deque<SimEntry> sq;
        std::deque<SimCompletion> cq;

    public:
        void submit_read(int fd, void* buffer, size_t size, uint64_t offset, uint64_t user_data) override {
            sq.push_back({fd, buffer, size, offset, user_data});
        }

        int process_sq() override {
            int processed = 0;
            while (!sq.empty()) {
                SimEntry op = sq.front();
                sq.pop_front();
                // Simulation: Just pretend we did it
                // In a real fallback, we would call pread here
                cq.push_back({op.user_data, (int)op.size});
                processed++;
            }
            return processed;
        }

        bool peek_cq(uint64_t& user_data, int& res) override {
            if (cq.empty()) return false;
            SimCompletion c = cq.front();
            cq.pop_front();
            user_data = c.user_data;
            res = c.res;
            return true;
        }
    };
}
