`timescale 1ns / 1ps

module bitnet_tb;

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter ACC_WIDTH = 32;
    parameter ARRAY_SIZE = 4; // Smaller size for testbench clarity

    // Signals
    reg clk;
    reg rst_n;
    reg enable;
    reg signed [DATA_WIDTH-1:0] activation_in [0:ARRAY_SIZE-1];
    reg [1:0] weight_in [0:ARRAY_SIZE-1];
    wire signed [ACC_WIDTH-1:0] result_out [0:ARRAY_SIZE-1];

    // DUT Instantiation
    bitnet_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ARRAY_SIZE(ARRAY_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .activation_in(activation_in),
        .weight_in(weight_in),
        .result_out(result_out)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    integer i;

    // Test Sequence
    initial begin
        // Initialize
        rst_n = 0;
        enable = 0;
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            activation_in[i] = 0;
            weight_in[i] = 2'b00;
        end

        #20;
        rst_n = 1;
        #10;
        enable = 1;

        // Test Case 1: Simple Accumulation
        // W = [+1, -1, 0, +1]
        // A = [10, 10, 50, -5]
        // Exp = [10, -10, 0, -5]
        $display("Test Case 1: Simple Accumulation");
        activation_in[0] = 16'd10; weight_in[0] = 2'b01; // +1
        activation_in[1] = 16'd10; weight_in[1] = 2'b10; // -1
        activation_in[2] = 16'd50; weight_in[2] = 2'b00; // 0
        activation_in[3] = -16'd5; weight_in[3] = 2'b01; // +1

        #10; // Wait one clock cycle

        // Verify Output
        if (result_out[0] !== 32'd10) $display("Error Idx 0: Exp 10, Got %d", result_out[0]);
        else $display("Idx 0 OK");

        if (result_out[1] !== -32'd10) $display("Error Idx 1: Exp -10, Got %d", result_out[1]);
        else $display("Idx 1 OK");

        if (result_out[2] !== 32'd0) $display("Error Idx 2: Exp 0, Got %d", result_out[2]);
        else $display("Idx 2 OK");

        if (result_out[3] !== -32'd5) $display("Error Idx 3: Exp -5, Got %d", result_out[3]);
        else $display("Idx 3 OK");


        // Test Case 2: Accumulation over time
        // Keep inputs same, next cycle should double the values
        $display("Test Case 2: Accumulation (Cycle 2)");
        #10;
        if (result_out[0] !== 32'd20) $display("Error Idx 0: Exp 20, Got %d", result_out[0]);
        else $display("Idx 0 OK (Accumulated)");


        // Test Case 3: Reset
        $display("Test Case 3: Reset");
        rst_n = 0;
        #10;
        if (result_out[0] !== 0) $display("Error: Reset failed");
        else $display("Reset OK");

        $finish;
    end

endmodule
