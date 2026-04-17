// =============================================================================
// spline_eval.sv
//
// Cubic spline envelope evaluator for QubiC gateware (gateware/dsp/).
// Replaces the envelope BRAM read path in elementconn with on-the-fly
// Horner polynomial evaluation.
//
// ─────────────────────────────────────────────────────────────────────────────
// INTEGRATION POINT IN elementconn
// ─────────────────────────────────────────────────────────────────────────────
// The existing path from cmdstb to ammod:
//
//   envaddr_cnt ──► reg_delay1 LEN=30 ──► external BRAM (READ_LATENCY=2)
//                                               │  envdata_r (1 cycle)
//                                               │  envdata_r2 (1 cycle)
//                                               │  envdata_r3 (1 cycle)
//                                               ▼
//                                           ammod envxy32x16
//   Total: 30 + 2 + 3 = 35 cycles from cmdstb
//
// spline_eval produces envdata_out with identical 35-cycle latency
// (7-cycle Horner pipeline + reg_delay1 OUT_DELAY=28).
// Replace the BRAM + pipeline with a single spline_eval instantiation.
//
// ─────────────────────────────────────────────────────────────────────────────
// env_word BIT MAPPING (pulse_iface ENV_WORD_WIDTH=24, unchanged)
// ─────────────────────────────────────────────────────────────────────────────
//   Original:  [23:12] = envstart   [11:0] = envlength
//   Spline:    [23:12] = coeff_base [11:0] = n_segs
//
//   No changes needed to pulse_reg.sv, pulse_iface.sv, or proc.sv.
//
// ─────────────────────────────────────────────────────────────────────────────
// FIXED-POINT FORMAT: Q1.15 signed 16-bit
// ─────────────────────────────────────────────────────────────────────────────
//   Range ±2.0 (all QubiC amplitudes < 1.0).  Precision = 1 DAC LSB (16-bit).
//   1.0 → 0x7FFF,  -1.0 → 0x8000
//
// ─────────────────────────────────────────────────────────────────────────────
// u NORMALIZATION
// ─────────────────────────────────────────────────────────────────────────────
//   u_norm = u_cnt / h_i  is computed as:
//       u_norm_q115 = (u_cnt * recip_h_q115) >> 15
//   where recip_h_q115 = round(32768 / h_i) is stored in the width BRAM.
//   This gives u_norm in [0, 1) for every segment, independent of width.
//
//   Coefficients are pre-scaled in Python (spline_coeff_pack.py):
//       a_hw = a,  b_hw = b*h_s,  c_hw = c*h_s^2,  d_hw = d*h_s^3
//   where h_s = segment width in seconds. All values are in (-1, 1) for
//   typical QubiC pulse amplitudes.
//
// ─────────────────────────────────────────────────────────────────────────────
// BRAM LAYOUTS
// ─────────────────────────────────────────────────────────────────────────────
// Coefficient BRAM — 128 bits per row, one row per segment:
//   [127:112] d_I  [111:96] c_I  [95:80] b_I  [79:64] a_I
//   [ 63: 48] d_Q  [ 47:32] c_Q  [31:16] b_Q  [15: 0] a_Q
//   All Q1.15 signed fixed-point.
//
// Width BRAM — 32 bits per row:
//   [31:16] = recip_h_q115  (= round(32768 / h_cycles), Q1.15 reciprocal)
//   [15: 0] = h_cycles      (segment width in clock cycles, integer)
//
// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE LATENCY: 7 cycles from cmdstb to Horner output
//   Cycle 1: segment controller registers cmdstb, issues BRAM addresses
//   Cycle 2: BRAM read stage 1 (READ_LATENCY=2)
//   Cycle 3: BRAM read stage 2 — coeff_data and width_data valid
//   Cycle 4: Stage 1 — latch coefficients + compute u_norm + p1 = d*u
//   Cycle 5: Stage 2 — p2 = (p1+c)*u
//   Cycle 6: Stage 3 — p3 = (p2+b)*u
//   Cycle 7: Output register — out = p3 + a
//   + OUT_DELAY=28 cycles via reg_delay1  → total = 35 cycles (matches original)
//
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// MIN_H_CYCLES — minimum allowed segment width (floor on knot spacing)
// ─────────────────────────────────────────────────────────────────────────────
// The time axis on our DAC is 14-bit; knots packed closer than this cannot be
// resolved by the hardware. MIN_H_CYCLES is the hardware equivalent of the
// `min_dt` parameter in the Python AutoKnots implementation:
//
//     min_dt [seconds]  =  MIN_H_CYCLES / FS
//
// The floor is also required for correctness of the prefetch logic below,
// which computes `u_cnt == h_cycles - 16'd3` and silently wraps if
// h_cycles < 3. Default 4 keeps prefetch safe with one cycle of margin.
//
// The packer (spline_coeff_pack.py) is responsible for enforcing this when
// generating the width BRAM; the simulation assertion below catches any
// violation at runtime if a stale BRAM image is loaded.
// ─────────────────────────────────────────────────────────────────────────────

module spline_eval #(
    parameter SEG_ADDRW    = 12,   // 12-bit address = 4096 max segments
    parameter OUT_DELAY    = 28,   // cycles added to match 35-cycle original path
    parameter MIN_H_CYCLES = 4     // minimum segment width in clock cycles
)(
    input  wire                  clk,
    input  wire                  cmdstb,      // one-cycle pulse at pulse start
    input  wire                  reset,

    // env_word fields from ifelement (via pulse_reg → pulse_iface)
    input  wire [SEG_ADDRW-1:0]  coeff_base,  // elem.envstart[11:0]
    input  wire [SEG_ADDRW-1:0]  n_segs,      // elem.envlength[11:0]

    // Coefficient BRAM — simple dual port, READ_LATENCY=2
    output reg  [SEG_ADDRW-1:0]  coeff_addr,
    input  wire [127:0]           coeff_data,

    // Width BRAM — simple dual port, READ_LATENCY=2
    // [31:16] = recip_h in Q1.15,  [15:0] = h_cycles
    output reg  [SEG_ADDRW-1:0]  width_addr,
    input  wire [31:0]            width_data,

    // Output: {I[15:0], Q[15:0]} — matches envxy32x16 format in ammod
    // Valid 35 cycles after cmdstb
    output wire [31:0]            envdata_out,
    output wire                   valid_out,
    output wire                   busy_out    // replaces elementconn's busy
);

// ─────────────────────────────────────────────────────────────────────────────
// Segment controller
// ─────────────────────────────────────────────────────────────────────────────

reg [SEG_ADDRW-1:0] seg_idx  = '0;
reg [15:0]          u_cnt    = '0;
reg [15:0]          h_cycles = '0;   // segment width in clock cycles
reg [15:0]          recip_h  = '0;   // 32768/h_cycles in Q1.15
reg                 active   = '0;

wire last_cycle = active && (u_cnt == h_cycles - 16'd1);
wire last_seg   = (seg_idx == n_segs - 1'b1);

always @(posedge clk) begin
    if (reset) begin
        seg_idx  <= '0;
        u_cnt    <= '0;
        active   <= '0;
        h_cycles <= '0;
        recip_h  <= '0;
    end
    else if (cmdstb) begin
        seg_idx <= '0;
        u_cnt   <= '0;
        active  <= 1'b1;
        // h_cycles and recip_h will be latched from BRAM in cycle 3
    end
    else if (active) begin
        if (last_cycle) begin
            u_cnt <= '0;
            if (last_seg)
                active <= 1'b0;
            else
                seg_idx <= seg_idx + 1'b1;
        end
        else begin
            u_cnt <= u_cnt + 1'b1;
        end
        // Latch segment parameters from width BRAM (READ_LATENCY=2,
        // so width_data reflects the address from 2 cycles ago — valid)
        h_cycles <= width_data[15:0];
        recip_h  <= width_data[31:16];
    end
end

// ─────────────────────────────────────────────────────────────────────────────
// Simulation-only assertion: enforce the MIN_H_CYCLES spacing floor.
// Catches a packer/BRAM mismatch before it corrupts the prefetch FSM.
// Synthesis tools strip `synthesis translate_off` blocks automatically.
// ─────────────────────────────────────────────────────────────────────────────
// synthesis translate_off
always @(posedge clk) begin
    if (active && width_data[15:0] != 16'd0 && width_data[15:0] < MIN_H_CYCLES) begin
        $display("ERROR spline_eval: h_cycles=%0d < MIN_H_CYCLES=%0d at addr %0d. Repack with min_h_cycles >= %0d.",
                 width_data[15:0], MIN_H_CYCLES, width_addr, MIN_H_CYCLES);
        $fatal(1);
    end
end
// synthesis translate_on

// BRAM address generation with sticky prefetch.
//
// BRAM READ_LATENCY=2: address issued at cycle T -> data valid at cycle T+2.
// We need segment i+1's data at the first cycle of segment i+1.
// Issue address 3 cycles early (accounting for 1 registered addr + 2 BRAM cycles).
//
// Once prefetch fires (at u_cnt == h_cycles-3), hold the next-segment address
// until the segment actually advances. Without holding, the address reverts to
// the current segment and the BRAM refetches wrong data.
reg prefetched = 0;   // stays high from prefetch trigger until seg advance

always @(posedge clk) begin
    if (reset || cmdstb) begin
        prefetched <= 0;
    end
    else if (active) begin
        if (last_cycle)
            prefetched <= 0;   // reset for next segment
        else if (u_cnt == h_cycles - 16'd3)
            prefetched <= 1;   // fire once and hold
    end
end

wire [SEG_ADDRW-1:0] next_seg = seg_idx + 1'b1;
wire use_next = prefetched || (active && (u_cnt == h_cycles - 16'd3));

always @(posedge clk) begin
    coeff_addr <= use_next ? (coeff_base + next_seg) : (coeff_base + seg_idx);
    width_addr <= use_next ? (coeff_base + next_seg) : (coeff_base + seg_idx);
end

assign busy_out = active;

// ─────────────────────────────────────────────────────────────────────────────
// Q1.15 multiply helper
// Full product of two 16-bit signed values = 32 bits.
// Q1.15 × Q1.15 = Q2.30; keep bits [29:15] for Q1.15 result.
// Synthesises to one DSP48E2 multiply per call.
// ─────────────────────────────────────────────────────────────────────────────

function automatic signed [15:0] q115_mul(
    input signed [15:0] a,
    input signed [15:0] b
);
    reg signed [31:0] prod;
    begin
        prod = a * b;
        q115_mul = prod[30:15];
    end
endfunction

// ─────────────────────────────────────────────────────────────────────────────
// Horner pipeline — 4 registered stages
//
// S(u) = a + u*(b + u*(c + u*d))
//
// u_norm is computed ONCE per output sample at stage 1 and frozen through
// all multiply stages. Using different u values at each stage would compute
// the wrong polynomial entirely.
//
// Stage 1 (cycle 4 from cmdstb): latch coeff + compute u_norm + p1 = d*u
// Stage 2 (cycle 5):             p2 = (p1 + c) * u
// Stage 3 (cycle 6):             p3 = (p2 + b) * u
// Stage 4 (cycle 7):             out = p3 + a
// ─────────────────────────────────────────────────────────────────────────────

// Stage 1 inputs — latched coefficients and u_norm
reg signed [15:0] s1_d_I=0, s1_c_I=0, s1_b_I=0, s1_a_I=0;
reg signed [15:0] s1_d_Q=0, s1_c_Q=0, s1_b_Q=0, s1_a_Q=0;
reg signed [15:0] s1_p1_I=0, s1_p1_Q=0;
reg signed [15:0] s1_u=0;      // u_norm, frozen here and propagated
reg               s1_gate=0;

// u_norm = u_cnt * recip_h >> 15  (Q1.15 result, always in [0,1))
// Use combinatorial multiply, result registered in stage 1
wire [31:0] u_norm_wide = u_cnt * recip_h;
wire signed [15:0] u_norm_q115 = $signed(u_norm_wide[15:0]);  // keep [0,1)

always @(posedge clk) begin
    if (reset) begin
        s1_d_I <= '0; s1_c_I <= '0; s1_b_I <= '0; s1_a_I <= '0;
        s1_d_Q <= '0; s1_c_Q <= '0; s1_b_Q <= '0; s1_a_Q <= '0;
        s1_p1_I <= '0; s1_p1_Q <= '0;
        s1_u <= '0;
        s1_gate <= 1'b0;
    end else begin
        // Latch coefficients from BRAM (coeff_data is valid here — READ_LATENCY=2 done)
        s1_d_I  <= $signed(coeff_data[127:112]);
        s1_c_I  <= $signed(coeff_data[111: 96]);
        s1_b_I  <= $signed(coeff_data[ 95: 80]);
        s1_a_I  <= $signed(coeff_data[ 79: 64]);
        s1_d_Q  <= $signed(coeff_data[ 63: 48]);
        s1_c_Q  <= $signed(coeff_data[ 47: 32]);
        s1_b_Q  <= $signed(coeff_data[ 31: 16]);
        s1_a_Q  <= $signed(coeff_data[ 15:  0]);
        // First Horner multiply: p1 = d * u
        s1_p1_I <= q115_mul($signed(coeff_data[127:112]), u_norm_q115);
        s1_p1_Q <= q115_mul($signed(coeff_data[ 63: 48]), u_norm_q115);
        // Freeze u for all downstream stages
        s1_u    <= u_norm_q115;
        s1_gate <= active;
    end
end

// Stage 2: p2 = (p1 + c) * u
reg signed [15:0] s2_p2_I=0, s2_p2_Q=0;
reg signed [15:0] s2_b_I=0,  s2_b_Q=0;
reg signed [15:0] s2_a_I=0,  s2_a_Q=0;
reg signed [15:0] s2_u=0;
reg               s2_gate=0;

always @(posedge clk) begin
    if (reset) begin
        s2_p2_I <= '0; s2_p2_Q <= '0;
        s2_b_I <= '0; s2_b_Q <= '0; s2_a_I <= '0; s2_a_Q <= '0;
        s2_u <= '0;
        s2_gate <= 1'b0;
    end else begin
        s2_p2_I <= q115_mul(s1_p1_I + s1_c_I, s1_u);
        s2_p2_Q <= q115_mul(s1_p1_Q + s1_c_Q, s1_u);
        s2_b_I  <= s1_b_I;
        s2_b_Q  <= s1_b_Q;
        s2_a_I  <= s1_a_I;
        s2_a_Q  <= s1_a_Q;
        s2_u    <= s1_u;
        s2_gate <= s1_gate;
    end
end

// Stage 3: p3 = (p2 + b) * u
reg signed [15:0] s3_p3_I=0, s3_p3_Q=0;
reg signed [15:0] s3_a_I=0,  s3_a_Q=0;
reg               s3_gate=0;

always @(posedge clk) begin
    if (reset) begin
        s3_p3_I <= '0; s3_p3_Q <= '0;
        s3_a_I <= '0; s3_a_Q <= '0;
        s3_gate <= 1'b0;
    end else begin
        s3_p3_I <= q115_mul(s2_p2_I + s2_b_I, s2_u);
        s3_p3_Q <= q115_mul(s2_p2_Q + s2_b_Q, s2_u);
        s3_a_I  <= s2_a_I;
        s3_a_Q  <= s2_a_Q;
        s3_gate <= s2_gate;
    end
end

// Stage 4: out = p3 + a, with saturation to Q1.15 range.
//
// Spline overshoot near the flat top of a clipped pulse (amp close to 1.0)
// can push p3+a slightly above 1.0, which in 16-bit two's-complement wraps
// to a large negative. Saturate to ±(2^15 - 1) instead so the DAC sees a
// legal clipped value rather than a sign flip.
reg [31:0] horner_out   = '0;
reg        horner_valid = '0;

wire signed [16:0] sum_I = $signed({s3_p3_I[15], s3_p3_I}) + $signed({s3_a_I[15], s3_a_I});
wire signed [16:0] sum_Q = $signed({s3_p3_Q[15], s3_p3_Q}) + $signed({s3_a_Q[15], s3_a_Q});

wire signed [15:0] sat_I = (sum_I > 17'sd32767)  ?  16'sd32767 :
                           (sum_I < -17'sd32768) ? -16'sd32768 :
                           sum_I[15:0];
wire signed [15:0] sat_Q = (sum_Q > 17'sd32767)  ?  16'sd32767 :
                           (sum_Q < -17'sd32768) ? -16'sd32768 :
                           sum_Q[15:0];

always @(posedge clk) begin
    if (reset) begin
        horner_out   <= '0;
        horner_valid <= 1'b0;
    end else begin
        // Pack as {I[15:0], Q[15:0]} — matches envxy32x16 in ammod:
        //   assign {envx[i],envy[i]} = envxy32x16[32*i+31:32*i+0]
        horner_out   <= {sat_I, sat_Q};
        horner_valid <= s3_gate;
    end
end

// ─────────────────────────────────────────────────────────────────────────────
// Output delay (reg_delay1 from QubiC gateware primitives)
// Pads Horner pipeline (7 cycles) to match original 35-cycle path latency.
// ─────────────────────────────────────────────────────────────────────────────

reg_delay1 #(.DW(32), .LEN(OUT_DELAY))
    env_delay (.clk(clk), .gate(1'b1), .din(horner_out),
               .dout(envdata_out), .reset(reset));

reg_delay1 #(.DW(1), .LEN(OUT_DELAY))
    valid_delay (.clk(clk), .gate(1'b1), .din(horner_valid),
                 .dout(valid_out), .reset(reset));

endmodule
