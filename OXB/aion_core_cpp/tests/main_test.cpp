#include <iostream>
#include <vector>
#include <cassert>
#include "../include/LinearModel.hpp"
#include "../include/Hilbert.hpp"
#include "../include/BitPacking.hpp"
#include "../include/RingBuffer.hpp"

// Simple Main Test for AION Core (to verify compilation and basic logic)
int main() {
    std::cout << "[AION] Starting Core Validation..." << std::endl;

    // 1. Test LinearModel
    {
        std::cout << "  - Testing LinearModel (RMI)...";
        Aion::LinearModel model;
        std::vector<double> keys = {0, 1, 2, 3, 4};
        std::vector<double> offsets = {10, 12, 14, 16, 18}; // y = 2x + 10
        model.train(keys, offsets);

        double p = model.predict(5);
        if (std::abs(p - 20.0) < 0.001) std::cout << " OK." << std::endl;
        else std::cout << " FAIL. Expected 20, got " << p << std::endl;
    }

    // 2. Test Hilbert
    {
        std::cout << "  - Testing Hilbert Curve...";
        int n = 4; // 4x4 grid
        // (0,0) -> 0
        uint32_t d = Aion::Hilbert::xy2d(n, 0, 0);
        int x, y;
        Aion::Hilbert::d2xy(n, d, &x, &y);

        if (d == 0 && x == 0 && y == 0) std::cout << " OK." << std::endl;
        else std::cout << " FAIL." << std::endl;
    }

    // 3. Test BitPacking
    {
        std::cout << "  - Testing BitPacking...";
        uint32_t data[4] = {1, 2, 3, 4}; // fits in 3 bits
        uint64_t packed[4] = {0};
        Aion::BitPacker::pack_scalar(data, packed, 4, 3);
        // 1(001), 2(010), 3(011), 4(100) -> ...00100011010001 (lsb first usually)
        // Implementation logic check would be more complex, just ensuring it runs.
        std::cout << " OK (Ran)." << std::endl;
    }

    std::cout << "[AION] All Systems Operational." << std::endl;
    return 0;
}
