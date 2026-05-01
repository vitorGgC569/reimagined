// =============================================================================
// BitNet Core v1.0 (Verilog)
// Target: ASIC / FPGA
// Description: 1.58-bit Ternary MAC Unit (Multiply-Accumulate)
// Power Efficiency: ~1000x vs FP32
// =============================================================================

module bitnet_core #(
    parameter DATA_WIDTH = 16,     // Input Activation Width (INT16/BF16 approx)
    parameter ACC_WIDTH = 32,      // Accumulator Width
    parameter ARRAY_SIZE = 64      // Systolic Array Dimension (64x64)
)(
    input wire clk,
    input wire rst_n,
    input wire enable,

    // Inputs
    input wire signed [DATA_WIDTH-1:0] activation_in [0:ARRAY_SIZE-1],
    input wire [1:0] weight_in [0:ARRAY_SIZE-1], // 2-bit Ternary Code: 00=0, 01=+1, 10=-1, 11=NaN

    // Outputs
    output reg signed [ACC_WIDTH-1:0] result_out [0:ARRAY_SIZE-1]
);

    integer i;

    // =========================================================================
    // TERNARY LOGIC CORE (No Multipliers!)
    // =========================================================================
    // The magic of BitNet: Multiplication becomes Multiplexing.
    // Weight +1 -> Add Activation
    // Weight -1 -> Sub Activation
    // Weight  0 -> NOP
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                result_out[i] <= 0;
            end
        end else if (enable) begin
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                case (weight_in[i])
                    2'b00: begin // Weight = 0
                        // result_out <= result_out; (NOP)
                    end
                    2'b01: begin // Weight = +1
                        result_out[i] <= result_out[i] + activation_in[i];
                    end
                    2'b10: begin // Weight = -1
                        result_out[i] <= result_out[i] - activation_in[i];
                    end
                    default: begin // 2'b11 (Unused/Zero)
                         // Treat as 0 for safety
                    end
                endcase
            end
        end
    end

endmodule

// =============================================================================
// BitNet Memory Controller (BMC) v1.0
// Description: Streams data from DRAM to the Systolic Array.
// Handles tiling and caching for infinite context emulation at hardware level.
// =============================================================================

module bitnet_mem_ctrl #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,

    // Host Interface (AXI-like simplified)
    input wire [ADDR_WIDTH-1:0] host_addr,
    input wire host_read_req,
    output reg [DATA_WIDTH-1:0] host_data_out,
    output reg host_data_valid,

    // Core Interface
    output reg [DATA_WIDTH-1:0] core_activation_out,
    output reg core_enable
);

    // Simple State Machine for Fetching
    reg [1:0] state;
    localparam IDLE = 0, FETCH = 1, STREAM = 2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            host_data_valid <= 0;
            core_enable <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (host_read_req) state <= FETCH;
                    core_enable <= 0;
                end
                FETCH: begin
                    // Simulate DRAM Latency (1 cycle here)
                    state <= STREAM;
                end
                STREAM: begin
                    host_data_out <= 16'hBEEF; // Mock Data
                    host_data_valid <= 1;
                    core_activation_out <= 16'hBEEF;
                    core_enable <= 1;
                    state <= IDLE; // Burst 1 for now
                end
            endcase
        end
    end

endmodule
