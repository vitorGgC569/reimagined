#include "aion/RingBuffer.hpp"
#include <deque>
#include <iostream>
#include "platform_file_io.hpp"

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
                const auto bytes = platform_io::read_at_offset(op.fd, op.buffer, op.size, op.offset);
                cq.push_back({op.user_data, bytes > 0 ? static_cast<int>(bytes) : 0});
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
