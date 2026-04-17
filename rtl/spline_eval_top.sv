// =============================================================================
// spline_eval_top.sv
//
// Cocotb harness for spline_eval. Provides BRAM models and reg_delay1 stub,
// exposes clean top-level ports for Python-side stimulus/capture.
// No initial blocks, no $finish — cocotb owns the simulation.
//
// BRAM contents are loaded via $readmemh from coeff_init.mem / width_init.mem
// (produced by spline_coeff_pack.py). For multi-test runs, cocotb can also
// force the BRAM arrays directly via dut.coeff_bram[i].value = ...
// =============================================================================

`timescale 1ns/1ps

// reg_delay1 standalone stub (matches the one in tb_spline_eval.sv)
module reg_delay1 #(parameter DW=1, parameter LEN=1)(
    input              clk,
    input              gate,
    input  [DW-1:0]    din,
    output [DW-1:0]    dout,
    input              reset
);
    reg [DW-1:0] sr [0:LEN-1];
    integer k;
    always @(posedge clk) begin
        if (reset) begin
            for (k=0; k<LEN; k=k+1) sr[k] <= '0;
        end else if (gate) begin
            sr[0] <= din;
            for (k=1; k<LEN; k=k+1) sr[k] <= sr[k-1];
        end
    end
    assign dout = sr[LEN-1];
endmodule


module spline_eval_top #(
    parameter SEG_ADDRW    = 12,
    parameter OUT_DELAY    = 28,
    parameter BRAM_DEPTH   = 4096,
    parameter MIN_H_CYCLES = 4
)(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  cmdstb,
    input  wire [SEG_ADDRW-1:0]  coeff_base,
    input  wire [SEG_ADDRW-1:0]  n_segs,
    output wire [31:0]           envdata_out,
    output wire                  valid_out,
    output wire                  busy_out
);

// ── BRAM models (READ_LATENCY=2) ─────────────────────────────────────────────
reg [127:0] coeff_bram [0:BRAM_DEPTH-1];
reg [31:0]  width_bram [0:BRAM_DEPTH-1];

wire [SEG_ADDRW-1:0] coeff_addr;
wire [SEG_ADDRW-1:0] width_addr;

reg [127:0] coeff_data_r0 = 0, coeff_data_out = 0;
reg [31:0]  width_data_r0 = 0, width_data_out = 0;

// Cocotb pokes coeff_bram[] / width_bram[] directly before driving cmdstb.
// If you want to preload from .mem files, add a $readmemh here — but the
// cocotb test doesn't need it.

// Reset the output pipeline registers so a new gate doesn't see the
// previous gate's last-segment data for the first 2 cycles after cmdstb
// (the spline_eval FSM asserts `active` before BRAM latency drains).
always @(posedge clk) begin
    if (reset) begin
        coeff_data_r0  <= 128'b0;
        coeff_data_out <= 128'b0;
        width_data_r0  <= 32'b0;
        width_data_out <= 32'b0;
    end else begin
        coeff_data_r0  <= coeff_bram[coeff_addr];
        coeff_data_out <= coeff_data_r0;
        width_data_r0  <= width_bram[width_addr];
        width_data_out <= width_data_r0;
    end
end

// ── DUT ──────────────────────────────────────────────────────────────────────
spline_eval #(
    .SEG_ADDRW    (SEG_ADDRW),
    .OUT_DELAY    (OUT_DELAY),
    .MIN_H_CYCLES (MIN_H_CYCLES)
) dut (
    .clk        (clk),
    .cmdstb     (cmdstb),
    .reset      (reset),
    .coeff_base (coeff_base),
    .n_segs     (n_segs),
    .coeff_addr (coeff_addr),
    .coeff_data (coeff_data_out),
    .width_addr (width_addr),
    .width_data (width_data_out),
    .envdata_out(envdata_out),
    .valid_out  (valid_out),
    .busy_out   (busy_out)
);

endmodule
