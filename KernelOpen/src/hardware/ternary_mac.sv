// =============================================================================
// Universal Heterogeneous Kernel (UHK) - Hardware IP
// Module: Native Ternary MAC (Multiply-Accumulate)
// Target: ASIC / FPGA (SystemVerilog)
// Description:
//   Implements the "Native Ternary Unit" proposed in Phase 3.
//   Performs Dot Product of 16-element vectors.
//   Weights are 2-bit Ternary {-1, 0, 1}.
//   Activations are INT8.
//   Saves area by replacing 16x8 multipliers with Add/Sub/Mux logic.
// =============================================================================

module ternary_mac_16 (
    input logic clk,
    input logic rst_n,
    input logic [1:0] weights [15:0],   // 16 pesos ternários (2-bit encoding)
    input logic [7:0] activations [15:0], // 16 ativações INT8
    output logic [19:0] accumulator       // Resultado acumulado (20 bits para evitar overflow de 16*255)
);

    // Encoding:
    // 00: 0
    // 01: +1
    // 10: -1
    // 11: 0 (ou X)

    logic [19:0] partial_sums [15:0];
    logic [19:0] stage1_sums [7:0];
    logic [19:0] stage2_sums [3:0];
    logic [19:0] stage3_sums [1:0];
    logic [19:0] final_sum;

    always_comb begin
        // 1. Ternary Logic Layer (No Multipliers!)
        for (int i = 0; i < 16; i++) begin
            case (weights[i])
                2'b00: partial_sums[i] = 20'sd0;
                2'b01: partial_sums[i] = {{12{activations[i][7]}}, activations[i]}; // Sign-extend +A
                2'b10: partial_sums[i] = -{{12{activations[i][7]}}, activations[i]}; // Negate -A
                2'b11: partial_sums[i] = 20'sd0; // Treat as zero
                default: partial_sums[i] = 20'sd0;
            endcase
        end

        // 2. Adder Tree (Logarithmic Reduction)
        // Stage 1 (16 -> 8)
        for (int i = 0; i < 8; i++) begin
            stage1_sums[i] = partial_sums[2*i] + partial_sums[2*i+1];
        end

        // Stage 2 (8 -> 4)
        for (int i = 0; i < 4; i++) begin
            stage2_sums[i] = stage1_sums[2*i] + stage1_sums[2*i+1];
        end

        // Stage 3 (4 -> 2)
        for (int i = 0; i < 2; i++) begin
            stage3_sums[i] = stage2_sums[2*i] + stage2_sums[2*i+1];
        end

        // Stage 4 (2 -> 1)
        final_sum = stage3_sums[0] + stage3_sums[1];
    end

    // 3. Output Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 20'sd0;
        end else begin
            accumulator <= final_sum;
        end
    end

endmodule
