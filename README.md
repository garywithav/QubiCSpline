# AutoKnots Spline Compression for QubiC Pulses

Adaptive cubic spline compression of quantum control waveforms for the
[QubiC](https://gitlab.com/LBL-QubiC/gateware) control system on ZCU216.
Replaces raw-sample BRAM storage with on-the-fly polynomial evaluation,
cutting envelope memory by up to 48× for typical readout pulses at 3 DAC LSB
error (well below gate-fidelity noise).

Based on Vitenti et al., "AutoKnots: Adaptive Knot Allocation for Spline
Fitting" ([arXiv:2412.13423](https://arxiv.org/abs/2412.13423)), with
hardware-specific additions for finite-resolution time DACs.

---

## Layout

```
spline/
├── README.md
├── .gitignore
│
├── python/                          # Python source + scripts
│   ├── autoknots.py                 Adaptive knot-placement core (the paper)
│   ├── pulse_envelopes.py           DRAG / cos_edge_square / square / mark
│   ├── spline_coeff_pack.py         float64 → Q1.15 BRAM packer
│   ├── spline_pulse_compiler.py     paradict → BRAM words (integration API)
│   ├── pack_qubitcfg.py             Driver: qubitcfg.json → .coe / .mem
│   ├── test_spline_eval.py          Cocotb testbench (5 tests)
│   ├── run_cocotb.py                Python runner (launches Icarus)
│   ├── plot_cocotb_results.py       Diagnostic plots: HW vs reference
│   └── plot_min_dt_comparison.py    Algorithmic study: min_dt floor
│
├── rtl/                             # SystemVerilog source
│   ├── spline_eval.sv               Pipelined Horner evaluator (FPGA module)
│   └── spline_eval_top.sv           Cocotb harness (BRAMs + DUT)
│
├── config/
│   └── qubitcfg.json                QubiC gate definitions (input)
│
├── notebooks/
│   └── autoknots_quantum.ipynb      Algorithm derivation + sandbox
│
└── build/                           # Generated outputs (populated by scripts)
    ├── spline_{coeff,width}.coe     Vivado BMG init files
    ├── spline_{coeff,width}.mem     $readmemh init for simulation
    ├── spline_manifest.json         Per-gate base address + diagnostics
    └── *.png                        Plot outputs (regeneratable, gitignored)
```

---

## Quick start

### Prerequisites
- Python 3.13 (not Windows Store — VPI loader can't dlopen WindowsApps).
  Alternatively Python 3.14 with `COCOTB_IGNORE_PYTHON_REQUIRES=1`.
- [Icarus Verilog](http://iverilog.icarus.com/) 12.0+ on PATH
- Python packages: `pip install cocotb numpy scipy matplotlib`

### Run the whole pipeline
All scripts live in `python/` and use paths relative to the project root.
Run them from inside `python/`:

```bash
cd python

# 1. Compile every gate from config/qubitcfg.json into build/
python pack_qubitcfg.py --only-compression-wins

# 2. Simulate rtl/spline_eval.sv against all compiled gates
python run_cocotb.py

# 3. Generate diagnostic plots (saved to build/)
python plot_cocotb_results.py
python plot_min_dt_comparison.py
```

Expected: **5/5 tests pass**, max hardware error ≤ 4 DAC LSBs, plots in `build/`.

---

## Architecture

Two phases, completely separate:

**Offline (Python, laptop):** `pack_qubitcfg.py` reads every gate in
`qubitcfg.json`, runs `autoknots()` to decide where to place knots, fits a
cubic spline, packs coefficients to Q1.15, and writes BRAM images.

**Online (FPGA, runtime):** [spline_eval.sv](rtl/spline_eval.sv) reads
4 coefficients from BRAM per segment and computes
`S(u) = a + u·(b + u·(c + u·d))` in a 4-stage Horner pipeline. Same 35-cycle
latency as the original envelope BRAM path in `elementconn`.

For calibration sweeps, the BRAM is runtime-writable — the same command path
QubiC already uses for envelope updates. The spline compiler is a drop-in
replacement for QubiC's raw-sample generator: same signature, different
representation.

```python
from spline_pulse_compiler import compile_pulse

# Called once per parameter change during calibration (~10 ms)
cp = compile_pulse("DRAG", paradict, twidth=32e-9, amp=0.14)
qubic.write_bram(base=addr,
                 coeff=cp.coeff_words,
                 width=cp.width_words)
```

---

## Key constraints

**Q1.15 fixed-point (16 bits signed):** range ±1.0, precision = 1 DAC LSB.
Output stage saturates on overflow so spline overshoot near peak amplitude
doesn't wrap to negative.

**`min_dt` (= `MIN_H_CYCLES / FS`):** minimum segment width set by the
time-axis DAC resolution. With `MIN_H_CYCLES = 4` and `FS = 1 GSPS`, that's a
4 ns floor. Segments narrower than this can't be resolved by the hardware
time counter. Enforced in three places that must stay in sync:

| Location | Symbol | Purpose |
|---|---|---|
| [spline_coeff_pack.py](python/spline_coeff_pack.py) | `MIN_H_CYCLES = 4` | Pack-time assertion |
| [spline_eval.sv](rtl/spline_eval.sv) | `parameter MIN_H_CYCLES = 4` | Runtime sim assertion |
| [autoknots.py](python/autoknots.py) | `min_dt` argument | Algorithm-level floor |

Short pulses (DRAG X90, 32 ns) hit this floor and can't benefit from splines
— [pack_qubitcfg.py](python/pack_qubitcfg.py) automatically skips them via
`--only-compression-wins` so they fall through to the raw-sample path.

---

## Results (deployed on qubitcfg.json)

From [spline_manifest.json](build/spline_manifest.json):

| Pulse type | Count | Duration | Compression | Max error |
|---|---|---|---|---|
| cos_edge_square (readout) | 8 | 2 µs | **48×** | 3–10 LSB |
| DRAG (Rabi calibration) | 16 | 1 µs | **2.6×** | 1–2 LSB typical |
| square (alignment tones) | ~10 | 1 µs | 12–25× | ≤ 1 LSB |
| DRAG X90 (skipped) | 8 | 32 ns | 0.25× | — uses raw samples |

Total BRAM utilization on ZCU216 coefficient BRAM: **19.4%** (796 / 4096 segments).

Hardware verification (cocotb, 5 tests):
- `test_reset_quiescent` — idle behavior ✓
- `test_full_pulse_matches_reference` — readout at 3.2 LSB ✓
- `test_drag_pulse_matches_reference` — DRAG at 1.4/1.6 LSB (I/Q) ✓
- `test_busy_deasserts_after_pulse` — FSM teardown ✓
- `test_qubitcfg_pipeline` — 3 gates from manifest, all ≤ 4 LSB ✓

### What the diagnostic plots show
Running [plot_cocotb_results.py](python/plot_cocotb_results.py) overlays the hardware
output on the float reference and plots per-sample error. The observed
structure across all pulses:

- **Flat regions are essentially exact** (< 1 LSB error). The cubic spline
  value collapses to the stored constant `a_I` when `u` is small.
- **Error concentrates on high-curvature regions** — the rising/falling
  cosine edges for readout, the Gaussian shoulders for DRAG. This is where
  the Q1.15 multiplication rounding in the Horner stages accumulates.
- **Error is zero-mean and roughly symmetric** (histogram centered at 0,
  no DC bias from the fixed-point pipeline).
- **Worst-case error sits at ~2.5 LSB theoretical** (3 multiply stages at
  0.5 LSB each + 1 LSB coefficient quantization). Observed max 3–4 LSB is
  within that budget.

### Effect of the `min_dt` floor
Running [plot_min_dt_comparison.py](python/plot_min_dt_comparison.py) compares the
current 4 ns floor against `min_dt = 0` (unconstrained) on the 32 ns DRAG
pulse:

| Setting | I knots | Q knots | Peak fit error |
|---|---|---|---|
| `min_dt = 4 ns` (current, hardware-realistic) | 9 | 9 | ~1e-4 (3 LSB) |
| `min_dt = 0` (algorithmic best, unshippable) | 23 | 18 | ~1e-5 (<1 LSB) |

The unconstrained fit is ~10× tighter, but those knot spacings (~1.4 ns) are
below what a 14-bit time DAC can address. The current floor is a
physics-of-the-DAC limit, not an algorithm choice.

---

## FPGA integration (next step, not yet done)

The RTL is ready to drop into QubiC's gateware. The integration path is
documented in the comment header of [spline_eval.sv](rtl/spline_eval.sv):

1. In `elementconn.v`, replace the envelope BRAM read path with a
   `spline_eval` instantiation. The `env_word` bit mapping
   (`[23:12]=coeff_base`, `[11:0]=n_segs`) keeps `pulse_reg.sv`,
   `pulse_iface.sv`, and `proc.sv` unchanged.
2. Vivado BMG IP for the two BRAMs (coefficient: 128b × 4096, width:
   32b × 4096, Simple Dual Port, Read Latency = 2), initialized from the
   `.coe` files this project produces.
3. Check DSP48E2 inference — each `q115_mul` should map to one DSP; the
   4-stage Horner pipeline should close timing at 1 GHz.

---

## Tuning knobs

| Parameter | Where | Default | Effect |
|---|---|---|---|
| `delta` | `autoknots(delta=...)` | `1e-3` | Relative error tolerance. Lower → more knots. |
| `eps` | `autoknots(eps=...)` | `0.5% * amp` | Near-zero floor. Prevents oversampling pulse tails. |
| `min_dt` | `autoknots(min_dt=...)` | `MIN_H_CYCLES / FS` | Hardware time-DAC floor (see above). |
| `MIN_H_CYCLES` | `spline_eval.sv`, `spline_coeff_pack.py` | `4` | Must match. Minimum segment width in clock cycles. |
| `SEG_ADDRW` | `spline_eval.sv` | `12` | BRAM depth = 2^SEG_ADDRW = 4096 segments. |
| `OUT_DELAY` | `spline_eval.sv` | `28` | Pipeline padding to hit original 35-cycle envelope latency. |
| `tol_lsb` | `test_spline_eval.py` | `4` | Cocotb pass threshold in DAC LSBs. |

---

## References

- Vitenti et al., *AutoKnots: Adaptive Knot Allocation for Spline Fitting*,
  arXiv:2412.13423
- [QubiC gateware](https://gitlab.com/LBL-QubiC/gateware) — the target platform
- [Cocotb](https://www.cocotb.org/) — Python-based HDL verification
