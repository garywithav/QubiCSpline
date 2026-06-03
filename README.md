# QubiCSpline

Adaptive cubic spline pulse compression for the QubiC RFSoC envelope path.

---

## What this project does and why

QubiC is the superconducting-qubit control system built at LBNL, running on the Xilinx ZCU216 RFSoC evaluation board. For every gate the system plays, the FPGA pushes a complex envelope waveform into the DAC at 1 billion samples per second (1 GSPS). Traditionally those envelopes are stored as raw samples in on-chip block RAM — one signed 16-bit I (in-phase) value and one Q (quadrature) value per nanosecond. A 2 µs readout pulse costs 8 kB; multiplied across all qubits and gate types, envelope memory dominates the gateware's storage budget. Worse, every calibration update requires re-uploading the entire sample buffer over PCIe.

QubiCSpline replaces raw samples with a cubic spline that the FPGA evaluates on the fly. Instead of storing n samples per pulse, we store k << n knots plus four polynomial coefficients per segment. A small SystemVerilog module reconstructs one output sample per clock cycle from those coefficients. For a typical readout pulse this reduces storage by 48×. For neutral-atom transport waveforms the reduction is approximately 2700×.

---

## Mathematical background

### Cubic splines

A cubic spline is a piecewise polynomial of degree 3. The time axis of a pulse is divided into segments by breakpoints called knots. On each segment between adjacent knots, the waveform is described by a cubic polynomial:

$$S_i(t) = a_i + b_i(t - t_i) + c_i(t - t_i)^2 + d_i(t - t_i)^3$$

The coefficients are chosen so that the curve and its first two derivatives are continuous across every knot boundary (C² continuity). This smoothness is physically important — a discontinuity in the waveform or its slope would inject unwanted frequency content into the qubit. Cubic polynomials are the minimum degree that gives C² continuity, making them the natural choice: smooth enough that the DAC cannot distinguish them from an ideal waveform, cheap enough that one segment is just four 16-bit numbers, and local in the sense that each segment only depends on its own coefficients.

### Horner evaluation

To evaluate the polynomial efficiently in hardware, we use Horner's method, which rewrites the cubic as:

$$S(u) = a + u(b + u(c + ud))$$

Working from the inside out: multiply d by u, add c, multiply by u, add b, multiply by u, add a. This requires only three multiplications and three additions, compared to six multiplications in the naive form. Since each multiplication maps to one dedicated DSP slice on the FPGA, Horner's method halves the hardware cost. The full design uses exactly 7 DSP slices: 3 for the I channel, 3 for the Q channel, and 1 for the u normalization described below.

The variable u is a normalized intra-segment time that runs from 0 at the start of the segment to just under 1 at the end, regardless of how many clock cycles the segment spans. Using a normalized variable keeps the coefficients in a well-behaved range and avoids the need for physical time units in the hardware.

### The u normalization and why division is avoided

The hardware maintains an integer counter u_cnt that increments every clock cycle from 0 to h_i - 1, where h_i is the number of clock cycles in the current segment. To get the normalized u = u_cnt / h_i, a division would be required. Division in hardware is expensive — it either requires many clock cycles or a large iterative circuit.

Since h_i is fixed and known at compile time for every segment, the Python compiler precomputes recip_h = round(32768 / h_i) and stores it in the width BRAM alongside h_i. At runtime the hardware computes u = (u_cnt × recip_h) >> 15, which is a single DSP multiply. The polynomial coefficients are also pre-scaled in Python by powers of the segment width in seconds so the hardware can use u ∈ [0, 1) directly and produce the correct output without knowing anything about physical time.

### AutoKnots: adaptive knot placement

Given an envelope function and a relative error tolerance δ, the AutoKnots algorithm (Vitenti et al., arXiv:2412.13423) returns the smallest knot set such that the cubic spline satisfies two conditions on every segment:

- Pointwise condition: the spline matches the original function at the segment midpoint to within δ.
- Integral condition: the integral of the spline over the segment matches the integral of the original function to within δ.

The integral condition catches features that the midpoint test would miss, such as a narrow spike straddling a wide segment.

The algorithm starts with a small uniform grid of knots, fits a spline, identifies segments failing either condition, bisects those segments by inserting a new knot at the midpoint, refits, and repeats until all segments pass. The result is a knot distribution that is dense where the pulse is complex and sparse where it is flat, which is why compression ratios scale with the information content of the pulse rather than its duration.

### Three hardware-aware extensions to AutoKnots

The published algorithm operates in continuous floating-point and does not know about the target hardware. Three extensions make it produce splines that round-trip correctly through Q1.15 fixed-point packing and FPGA evaluation.

**min_dt — minimum segment width.** The FPGA's prefetch logic reads the next segment's BRAM data 3 cycles before the current segment ends. A segment shorter than 4 clock cycles causes this comparison to wrap and the control logic to fail. If bisection would produce a segment narrower than min_dt = 4 cycles, that segment is frozen and the best achievable fit at that resolution is returned with a warning.

**max_dt — maximum segment width.** The segment width h_cycles is stored as a 16-bit integer in the width BRAM, capping segments at 65,535 cycles (65.5 µs). Segments wider than max_dt are force-subdivided every iteration regardless of the error condition. For a 1 ms atom-transport waveform this turns 13 algorithm-optimal segments into 23 hardware-legal ones.

**grid_quantum — knot snapping to the sample grid.** The FPGA counter is integer cycles, so the actual segment boundary executed by the hardware is round(t_knot × fs). If the algorithm converged against a spline with a knot at 13.5 ns but the hardware executes a knot at 13 ns, the two splines are different and the error guarantee no longer applies. In testing this produced errors of 44 DAC LSBs — eleven times over the 4 LSB project budget. The fix is to snap every proposed knot to the nearest integer nanosecond before the error check runs, so convergence applies to the on-grid spline that the hardware actually executes.

### Fixed-point arithmetic: Q1.15

The FPGA evaluates the polynomial in Q1.15 fixed-point: a signed 16-bit integer representing values in [-1, +1) with resolution 2⁻¹⁵ ≈ 3 × 10⁻⁵. One Q1.15 unit equals one DAC LSB, matching the hardware's native resolution. The project error budget is 4 DAC LSBs, which is well below the dominant gate-error source (qubit decoherence at approximately 10⁻³ relative error).

---

## Architecture: offline Python and online FPGA

The project divides cleanly into two halves that communicate only through BRAM contents.

The offline Python side runs on the control workstation. It takes a pulse description, runs AutoKnots to find optimal knot placements, converts the resulting spline to Q1.15 fixed-point integers, and writes the coefficient and width BRAMs as .mem files (for simulation) and .coe files (for Vivado initialization).

The online FPGA side runs in the bitstream at 250 MHz. On receiving a cmdstb pulse it walks through the pre-loaded segments, evaluates the Horner pipeline once per clock cycle, and produces one {I, Q} output sample per nanosecond. The first valid sample appears exactly 35 clock cycles after cmdstb — a latency contract baked into the surrounding QubiC gateware that this module preserves exactly.

The 35-cycle latency is composed of 8 pipeline cycles (2 BRAM read latency + 1 Stage 1a register + 4 Horner stages + 1 output register) plus a 27-stage shift register that pads the total to exactly 35. The shift register exists solely to match the existing QubiC timing contract without requiring edits to proc.sv, pulse_iface.sv, or any other surrounding file.

---

## File structure

```
QubiCSpline/
├── python/
│   ├── pulse_envelopes.py
│   ├── autoknots.py
│   ├── spline_coeff_pack.py
│   ├── spline_pulse_compiler.py
│   ├── pack_qubitcfg.py
│   ├── run_cocotb.py
│   ├── test_envelopes.py
│   ├── test_constraints.py
│   ├── test_spline_eval.py
│   ├── plot_cocotb_results.py
│   └── plot_min_dt_comparison.py
├── rtl/
│   ├── spline_eval.sv
│   └── spline_eval_top.sv
├── scripts/
│   ├── synth_check.tcl
│   └── timing.xdc
├── config/
│   └── qubitcfg.json
├── build/
│   ├── spline_coeff.mem
│   ├── spline_coeff.coe
│   ├── spline_width.mem
│   ├── spline_width.coe
│   └── spline_manifest.json
└── vivado_reports/
    ├── utilization.rpt
    ├── timing_summary.rpt
    ├── timing_worst.rpt
    ├── drc.rpt
    ├── clocks.rpt
    ├── run.log
    └── pre_pipeline_fix/
```

---

## File descriptions

### Python pipeline

**`python/pulse_envelopes.py`**
Defines every pulse shape the system supports. For each gate type (DRAG, cos_edge_square readout, square, adiabatic_ramp, rydberg_drag, atom_transport) this file provides a function that takes amplitude, duration, and shape parameters and returns a pair of callables (I_func, Q_func). It also exports feature_knots(), which returns time points where a knot must be placed regardless of the AutoKnots error check — for example, the junction between the ramp and hold phases of an adiabatic ramp is a curvature discontinuity that the midpoint test would otherwise miss.

**`python/autoknots.py`**
The mathematical core of the project. Implements the AutoKnots adaptive bisection algorithm with the three hardware extensions (min_dt, max_dt, grid_quantum). Two public entry points: autoknots() fits a single channel and returns the spline, knot locations, and convergence status; compress_multichannel() fits I and Q separately then unions their knot sets and refits both channels on the shared grid, which is required because the width BRAM has one segment-width entry per time slot and cannot represent two different segmentations simultaneously.

**`python/spline_coeff_pack.py`**
Converts a scipy CubicSpline object into Q1.15 fixed-point BRAM words. Validates h_cycles bounds, checks for d-coefficient overflow after pre-scaling, then for each segment computes recip_h, scales the a/b/c/d coefficients by powers of the segment width in seconds, rounds to Q1.15, and packs into the 128-bit coefficient word and 32-bit width word formats the RTL expects. Also writes .mem files for Icarus Verilog simulation and .coe files for Vivado BRAM initialization, and provides a verify() function that runs a software simulation of the Q1.15 Horner pipeline to confirm the packed coefficients are within the error budget.

**`python/spline_pulse_compiler.py`**
Single-pulse entry point. compile_pulse() ties together pulse_envelopes, autoknots, and spline_coeff_pack into one call: look up the default error tolerance for this shape, build the envelope functions, run AutoKnots on both channels with the appropriate hardware constraints, pack the coefficients, and return a CompiledPulse dataclass containing the BRAM words, knot locations, segment count, compression ratio, and convergence flag. Also provides verify_pulse() for a fast software error check without needing the simulator. The _DEFAULT_DELTA table records measured (delta, n_segs, compression, max_err_lsb) data for each pulse shape, with the chosen delta being the loosest value that keeps max error under 4 DAC LSBs.

**`python/pack_qubitcfg.py`**
Top-level driver. Reads config/qubitcfg.json, calls compile_pulse() on every gate, concatenates the resulting BRAM words into a single image, assigns each gate a base address, and writes the five files in build/. The --only-compression-wins flag skips gates where the spline representation is larger than the raw-sample equivalent (typically very short 32 ns DRAG pulses that hit the min_dt floor too hard to benefit); those gates fall back to the existing raw-sample path.

**`python/run_cocotb.py`**
Launches the cocotb hardware simulation. Compiles the SystemVerilog using Icarus Verilog via cocotb_tools.runner and runs test_spline_eval.py. No Makefile required. Produces results.xml, cocotb_results.npz, and cocotb_drag_results.npz.

**`python/plot_cocotb_results.py`**
Reads cocotb_results.npz and cocotb_drag_results.npz and produces two diagnostic figures: the readout pulse waveform overlay with error histogram, and the DRAG I/Q overlay with per-channel error breakdown.

**`python/plot_min_dt_comparison.py`**
Pedagogical figure. Fits the DRAG pulse twice — once with min_dt enforced and once without — and plots the resulting knot placements side by side. Shows why the hardware constraint produces a coarser knot distribution than the pure math algorithm would choose for a sharp pulse.

### Tests

**`python/test_envelopes.py`**
Pure Python unit tests, no simulator required. Compiles one of each pulse shape through compile_pulse(), runs verify_pulse(), and asserts that the spline converged and max error is under 4 DAC LSBs on both I and Q channels. Fast enough to run after every code change. 4/4 pass.

**`python/test_constraints.py`**
Adversarial constraint tests. Each test constructs a pulse that would fail in a specific, attributable way if a given constraint were absent, verifies the constraint prevents the failure, and then demonstrates the failure with the constraint disabled — proving the test is genuinely adversarial rather than coincidentally passing. Tests cover min_dt (bisection divergence on a narrow Gaussian spike), max_dt (h_cycles overflow on a 1 ms atom_transport segment), and grid_quantum (44 LSB hardware error from fractional-nanosecond knots on an adiabatic_ramp). 3/3 pass.

**`python/test_spline_eval.py`**
Cocotb hardware simulation testbench. Uses cocotb to drive Icarus Verilog cycle-by-cycle, populating the simulated BRAMs from Python using the same pack_coefficients() call that production uses, firing cmdstb, capturing every output sample, and comparing against the float reference. Five tests: reset quiescence, full readout pulse accuracy (max 3.2 LSB), DRAG pulse accuracy on both channels (max 1.6 LSB), busy signal deassertion, and end-to-end qubitcfg pipeline with three gates fired at their manifest base addresses (max 4.0 LSB). 5/5 pass.

### RTL (hardware description)

RTL stands for Register Transfer Level — the abstraction at which hardware is described in SystemVerilog. RTL describes how data moves between registers on each clock edge. Vivado compiles RTL into the actual gates, wires, and DSP slices placed on the chip.

**`rtl/spline_eval.sv`**
The only RTL file that ends up in the bitstream. Contains the segment FSM (walks through segments, maintains u_cnt), address prefetch logic (issues the next segment's BRAM address 3 cycles early to account for BRAM read latency), the Q1.15 multiplier function (synthesizes to one DSP48E2 per call site), the 5-stage Horner pipeline (Stage 1a through Stage 4), saturating output logic, and the 27-stage output shift register that pads pipeline latency to exactly 35 cycles. The two coefficient and width BRAMs are external to this module — it communicates with them through address/data ports, matching the existing QubiC envelope-memory structure and allowing runtime calibration writes to continue working via the existing path.

**`rtl/spline_eval_top.sv`**
Simulation harness, not synthesized. Defines the reg_delay1 shift-register primitive used inside spline_eval (in production this comes from the QubiC gateware library) and wraps spline_eval with two SystemVerilog memory arrays modeling the external BRAMs. Cocotb writes directly into these arrays from Python before firing cmdstb.

### Scripts

**`scripts/synth_check.tcl`**
TCL script fed to Vivado to run synthesis and implementation. Specifies source files, target part (xczu48dr on the ZCU216), top module (spline_eval), and report output locations. Run with: `vivado -mode batch -source scripts/synth_check.tcl`

**`scripts/timing.xdc`**
Xilinx Design Constraints file. Declares the 250 MHz clock constraint. This is what makes timing slack meaningful — Vivado measures all register-to-register paths against this period and reports how much margin each path has.

### Configuration and build outputs

**`config/qubitcfg.json`**
Input configuration file defining every gate in the system — envelope shape, amplitude, duration, and shape parameters for each qubit and gate type. This is the starting point for pack_qubitcfg.py.

**`build/spline_coeff.mem`**
128-bit-wide BRAM image containing the packed Q1.15 a/b/c/d coefficients for every segment of every gate, concatenated in base-address order. Read by Icarus Verilog during cocotb simulation.

**`build/spline_coeff.coe`**
Same data as spline_coeff.mem in Vivado Block Memory Generator coefficient file format. Used to initialize the coefficient BRAM at bitstream generation time during Phase B integration.

**`build/spline_width.mem`**
32-bit-wide BRAM image containing h_cycles and recip_h for every segment. Read by Icarus Verilog during simulation.

**`build/spline_width.coe`**
Same data as spline_width.mem in Vivado .coe format.

**`build/spline_manifest.json`**
Human-readable index of all compiled gates. For each gate records the BRAM base address, number of segments, compression ratio, and max error in DAC LSBs. Read by test_spline_eval.py to know which base address to use when firing each gate.

### Vivado reports

All files in vivado_reports/ are generated by synth_check.tcl and should not be edited manually.

**`vivado_reports/utilization.rpt`**
Resource usage report. Records LUT, flip-flop, DSP, and BRAM counts. The key number is 7 DSPs — confirming every multiply landed in a dedicated DSP slice rather than LUT logic.

**`vivado_reports/timing_summary.rpt`**
Full timing report. The key numbers are WNS (Worst Negative Slack) = +0.739 ns, confirming timing closure at 250 MHz with approximately 23% margin, and WHS (Worst Hold Slack) = +0.032 ns.

**`vivado_reports/timing_worst.rpt`**
Detailed breakdown of the single worst timing path. Useful for identifying which logic is on the critical path if further timing optimization is needed.

**`vivado_reports/drc.rpt`**
Design Rule Check report. Zero errors. Two expected warnings (NSTD-1, UCIO-1) about I/O standards not being set — these are artifacts of synthesizing the module in isolation and disappear when integrated into the full QubiC design where pin assignments are provided.

**`vivado_reports/clocks.rpt`**
Lists every clock in the design and its properties. One clock at 250 MHz.

**`vivado_reports/run.log`**
Full console output from the Vivado synthesis and implementation run. The first place to look if synthesis fails.

**`vivado_reports/pre_pipeline_fix/`**
Archived reports from before the Stage 1a pipeline register was inserted. Preserved for comparison — WNS was +0.387 ns before the fix and +0.739 ns after, with theoretical Fmax improving from ~277 MHz to ~307 MHz.

---

## Synthesis results

| Metric | Value |
|---|---|
| WNS (setup slack) | +0.739 ns |
| WHS (hold slack) | +0.032 ns |
| Failing timing endpoints | 0 / 895 |
| Theoretical Fmax | ~307 MHz |
| DSP slices | 7 / 4272 (0.16%) |
| LUTs | 258 / 425,280 (0.06%) |
| Flip-flops | 391 / 850,560 (0.05%) |
| BRAM tiles | 0 (BRAMs are external to module) |
| DRC errors | 0 |

## Compression results

| Pulse type | Duration | Compression | Max error |
|---|---|---|---|
| cos_edge_square (readout) | 2 µs | 48× | 3–4 LSB |
| DRAG | 1 µs | 2.6× | 1–2 LSB |
| square | 1 µs | 12–25× | ≤ 1 LSB |
| atom_transport (demo) | 1 ms | ~2700× | 1.8 LSB |

---

## Reproducing the results

**Python tests (no hardware required):**
```bash
cd python
python test_envelopes.py
python test_constraints.py
```

**Hardware simulation (requires Icarus Verilog and cocotb):**
```bash
cd python
python run_cocotb.py
```

**Vivado synthesis (requires Vivado 2022.2 and LBNL license):**
```bash
export XILINXD_LICENSE_FILE=27004@engvlic3.lbl.gov
source /path/to/Vivado/2022.2/settings64.sh
vivado -mode batch -source scripts/synth_check.tcl
```