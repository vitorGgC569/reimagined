#ifndef UHK_QUANTUM_STABILIZER_SIM_H
#define UHK_QUANTUM_STABILIZER_SIM_H

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace uhk {
namespace quantum {

// =============================================================================
// Phase 7: Real Quantum Simulation (Surface Code)
// =============================================================================
// Simulates a 3x3 Grid of Data Qubits with X and Z Stabilizers.
// Generates *real* syndromes based on injected Pauli errors.
// =============================================================================

// Grid 3x3 Data Qubits (D0..D8)
// Stabilizers check parity of neighbors.
// S0 (Z-check): D0, D1, D3, D4
// S1 (Z-check): D4, D5, D7, D8
// S2 (X-check): D1, D2, D4, D5
// S3 (X-check): D3, D4, D6, D7

struct Syndrome {
    uint32_t z_stabilizers; // Checks X errors
    uint32_t x_stabilizers; // Checks Z errors
};

class StabilizerSimulator {
public:
    StabilizerSimulator() {
        // Init 9 data qubits
        qubits_x_error_.resize(9, false);
        qubits_z_error_.resize(9, false);
    }

    void inject_error(int qubit_idx, char type) {
        if (type == 'X') qubits_x_error_[qubit_idx] = !qubits_x_error_[qubit_idx];
        if (type == 'Z') qubits_z_error_[qubit_idx] = !qubits_z_error_[qubit_idx];
    }

    // Measure Stabilizers (Real Parity Check Logic)
    Syndrome measure() {
        Syndrome s = {0, 0};

        // Z-Stabilizers detect X-Errors (Anti-commute)
        // S0: 0,1,3,4
        int p0 = qubits_x_error_[0] ^ qubits_x_error_[1] ^ qubits_x_error_[3] ^ qubits_x_error_[4];
        if (p0) s.z_stabilizers |= 1;

        // S1: 4,5,7,8
        int p1 = qubits_x_error_[4] ^ qubits_x_error_[5] ^ qubits_x_error_[7] ^ qubits_x_error_[8];
        if (p1) s.z_stabilizers |= 2;

        // X-Stabilizers detect Z-Errors
        // S2: 1,2,4,5
        int p2 = qubits_z_error_[1] ^ qubits_z_error_[2] ^ qubits_z_error_[4] ^ qubits_z_error_[5];
        if (p2) s.x_stabilizers |= 1;

        // S3: 3,4,6,7
        int p3 = qubits_z_error_[3] ^ qubits_z_error_[4] ^ qubits_z_error_[6] ^ qubits_z_error_[7];
        if (p3) s.x_stabilizers |= 2;

        return s;
    }

    // Apply correction
    void apply_x_correction(int qubit_idx) { qubits_x_error_[qubit_idx] = !qubits_x_error_[qubit_idx]; }
    void apply_z_correction(int qubit_idx) { qubits_z_error_[qubit_idx] = !qubits_z_error_[qubit_idx]; }

    bool is_logical_clean() {
        // Check if errors remain
        for(auto b : qubits_x_error_) if(b) return false;
        for(auto b : qubits_z_error_) if(b) return false;
        return true;
    }

private:
    std::vector<bool> qubits_x_error_;
    std::vector<bool> qubits_z_error_;
};

} // namespace quantum
} // namespace uhk

#endif // UHK_QUANTUM_STABILIZER_SIM_H
