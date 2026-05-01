#pragma once
#include <cstdint>

namespace Aion {

    // Pilar D: Corpo (Hilbert)
    class Hilbert {
    public:
        // XY to D
        static uint32_t xy2d(int n, int x, int y);

        // D to XY
        static void d2xy(int n, int d, int *x, int *y);

    private:
        static void rot(int n, int &x, int &y, int rx, int ry);
    };
}
