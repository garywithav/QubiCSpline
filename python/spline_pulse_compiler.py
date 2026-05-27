"""
spline_pulse_compiler.py
========================
End-to-end pulse → BRAM words compilation. This is the drop-in replacement for
QubiC's raw-sample generator: give it a pulse description, get back BRAM words.

    pulse_description → AutoKnots → float spline → Q1.15 pack → BRAM words

Primary entry point:
    compile_pulse(env_func, paradict, twidth, amp, ...) -> CompiledPulse

Calibration flow (called on every parameter change):
    cp = compile_pulse('DRAG', paradict, twidth=32e-9, amp=0.14)
    qubic.write_bram(COEFF_BRAM, cp.coeff_words, base=address)
    qubic.write_bram(WIDTH_BRAM, cp.width_words, base=address)
    # FPGA plays the updated pulse on the next gate fire

The compiler is pure: no I/O, no globals modified. Safe to call in a tight
calibration loop; autoknots runs in milliseconds per pulse.

────────────────────────────────────────────────────────────────────────────
Design note — three mechanisms for shape-specific tuning
────────────────────────────────────────────────────────────────────────────
Adding a new pulse shape generally needs no core-algorithm change. Three
mechanisms in this module compose to handle every shape we've seen:

  1. `pulse_envelopes.feature_knots(env_func, ...)` — interior knot times
     the spline MUST honor (e.g. C²-discontinuity junctions for ramp+hold
     shapes). compile_pulse passes the result to autoknots as `seed_knots`.

  2. `autoknots(..., grid_quantum=1/fs)` — every proposed knot (initial
     grid, bisection midpoint, refine midpoint, seed) is snapped to the
     sample-rate grid before convergence is checked. Guarantees the float
     spline lives on the same time grid the FPGA's u_cnt counter does, so
     pack_coefficients doesn't introduce a fractional-ns mismatch.

  3. `_DEFAULT_DELTA[env_func]` (below) — per-shape tolerance default.
     Some shapes need tighter delta to hit the 4 LSB HW-error budget;
     others fit cleanly at the project default 1e-3. The dict is keyed by
     env_func name, with every entry annotated by measured numbers.

Together these let a new pulse shape be added by:
  (a) writing the envelope generator in pulse_envelopes.py,
  (b) optionally registering its junction points in `_FEATURE_DISPATCH`,
  (c) optionally registering a tighter delta in `_DEFAULT_DELTA` if the
      default 1e-3 doesn't get HW error below 4 LSB.

No changes to autoknots.py, spline_coeff_pack.py, or spline_eval.sv are
required for a new shape unless it needs new convergence/quantization
behavior the existing mechanisms can't express.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from autoknots import autoknots, compress_multichannel
from pulse_envelopes import make_envelope, feature_knots
from spline_coeff_pack import pack_coefficients, MIN_H_CYCLES, MAX_H_CYCLES, FS


# ─────────────────────────────────────────────────────────────────────────────
# _DEFAULT_DELTA — per-shape AutoKnots tolerance
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is the delta at which compile_pulse() invokes autoknots when
# the caller doesn't pass an explicit `delta`. Shapes not in this dict use
# _GLOBAL_DEFAULT_DELTA (1e-3).
#
# RULE: 4 DAC LSB is the project-wide HW-error budget (verify_pulse). Pick
# the loosest delta for which max_err_I_lsb < 4 holds at representative
# parameters. Tighter wastes BRAM; looser fails the test. Each entry MUST
# be justified by a measured delta-sweep at known parameters — record the
# numbers below so a future reader doesn't need to re-derive them.
#
# Format per entry:
#   "env_func": <delta>,   # <one-line rationale + the measured row>
#
# ── adiabatic_ramp: 3e-4 ────────────────────────────────────────────────
#   Shape: minimum-jerk ramp + hold. C²-discontinuity at t=ramp_time.
#   Junction is seeded via feature_knots. The remaining error budget lives
#   on the steep curvature in the last ~6% of the ramp (worst sample at
#   t = 0.94 * ramp_time), where the spline's local 4th-derivative bound
#   forces ~6 LSB error at delta=1e-3. Tightening to 3e-4 adds 5 segments
#   and drops HW-vs-truth below 4 LSB.
#
#   Measured (twidth=1µs, ramp_time=200ns, amp=0.5):
#     delta     n_segs   compression   max_err_I (LSB)
#     1e-3       24        2.6×          6.49     ← fails 4 LSB
#     3e-4       29        2.2×          3.60     ← used here
#     1e-4       36        1.7×          1.09
#     3e-5       41        1.5×          0.97
#     1e-5       51        1.2×          0.97     ← plateaus at Q1.15 floor
#
# ── rydberg_drag: 5e-4 ──────────────────────────────────────────────────
#   Shape: Gaussian I + linearly chirped detuning on channel 2 (Levine/
#   Pichler form, NOT transmon DRAG). No junctions. The Q channel (linear
#   chirp) is trivial to fit; all error budget is on the Gaussian I.
#   delta=1e-3 leaves the algorithm with too few knots near the Gaussian
#   shoulders. A modest tighten to 5e-4 adds 2 segments and drops the
#   worst-case error from ~16 LSB to ~1.2 LSB — a sharp cliff because
#   autoknots' midpoint sampling is finally fine enough to catch the
#   shoulder curvature.
#
#   Measured (twidth=1µs, σ=200ns, amp=0.8, detuning_max=0.3):
#     delta     n_segs   compression   max_err_I (LSB)
#     1e-3       16        3.9×         15.75     ← fails 4 LSB
#     5e-4       18        3.5×          1.21     ← used here (cliff)
#     3e-4       18        3.5×          1.21
#     1e-4       22        2.8×          1.21
#     5e-5       30        2.1×          1.19
#     1e-5       40        1.6×          1.38     ← plateau
#
# ── atom_transport: 5e-4 ────────────────────────────────────────────────
#   Shape: minimum-jerk position polynomial (Lukin-group atom shuttling).
#   Same math family as adiabatic_ramp's ramp segment, but I-channel only
#   with no junction (the polynomial is C^∞ across the whole pulse). Two
#   constraints compete with delta here:
#
#     1. The shape's curvature is highest at the endpoints (where x''(t)
#        peaks), so the algorithm wants narrow segments there. At long
#        twidth, the spline-fit error dominates and delta sets the budget.
#     2. autoknots(max_dt=MAX_H_CYCLES/fs=65.5 µs) caps segment width at
#        the 16-bit h_cycles packing limit. At twidth ≥ 1 ms this binds
#        before delta does — the algorithm subdivides "for free" to meet
#        the cap, often achieving better fit accuracy than delta required.
#
#   delta=5e-4 is the cheapest value that holds < 4 LSB across the spec
#   range (100 µs to 5 ms). At twidth=100 µs the cap doesn't bind and
#   delta is the controlling tolerance. At twidth=1 ms and beyond, the
#   cap takes over and the achieved error is much better than δ would
#   predict alone — there's no benefit to tightening δ further.
#
#   Measured (x_start=0, x_end=0.7, amp=1.0, with max_dt active):
#     twidth   delta     n_segs   compression   max_err_I (LSB)
#     100 µs   1e-3        13        481×          18.08    ← fails
#     100 µs   5e-4        15        417×           2.07    ← used
#     100 µs   1e-4        21        298×           1.58
#       1 ms   1e-3        23       2717×           3.50    ← max_dt-bound
#       1 ms   5e-4        23       2717×           3.50    ← used
#       1 ms   1e-4        27       2315×           1.65
#       5 ms   1e-3        81       3858×           1.77    ← max_dt-bound
#       5 ms   5e-4        81       3858×           1.77    ← used
#
#   Compression ratios here are spectacular (300–4000×) because the
#   shape is mostly featureless polynomial; max_dt has the side effect
#   of bounding how good compression can be at long twidth, but
#   "thousands of times smaller than raw samples" is still a clean win.
#
# ─────────────────────────────────────────────────────────────────────────
_GLOBAL_DEFAULT_DELTA = 1e-3

_DEFAULT_DELTA = {
    "adiabatic_ramp":  3e-4,
    "rydberg_drag":    5e-4,
    "atom_transport":  5e-4,
}


def _resolve_delta(env_func: str, override: Optional[float]) -> float:
    """Pick delta: explicit override > per-shape entry > global default."""
    if override is not None:
        return float(override)
    return _DEFAULT_DELTA.get(env_func, _GLOBAL_DEFAULT_DELTA)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CompiledPulse:
    """Result of compile_pulse. Contains everything needed to program the FPGA."""
    coeff_words: list            # 128-bit ints, one per segment
    width_words: list            # 32-bit ints, one per segment
    knots: np.ndarray            # knot times (seconds)
    n_segments: int
    converged: bool              # True if autoknots met the tolerance
    raw_samples_equivalent: int  # what the old path would have stored
    env_func: str
    paradict: dict
    twidth: float
    amp: float

    @property
    def compression_ratio(self) -> float:
        """Raw sample bytes / spline bytes."""
        return (self.raw_samples_equivalent * 4) / (self.n_segments * 32 * 2)  # I+Q

    @property
    def ddr_bytes(self) -> int:
        """Bytes written to BRAM for this pulse (both I and Q channels)."""
        return self.n_segments * 32 * 2  # 16B per segment × 2 channels


# ─────────────────────────────────────────────────────────────────────────────
def compile_pulse(
    env_func: str,
    paradict: dict,
    twidth: float,
    amp: float,
    delta: Optional[float] = None,
    eps: Optional[float] = None,
    fs: float = FS,
    min_h_cycles: int = MIN_H_CYCLES,
    max_knots: int = 4096,
    refine: bool = True,
) -> CompiledPulse:
    """
    Compile a QubiC pulse definition to spline_eval BRAM words.

    Parameters
    ----------
    env_func  : 'DRAG', 'cos_edge_square', 'square', 'mark',
                'adiabatic_ramp', 'rydberg_drag', 'atom_transport'
    paradict  : envelope-specific parameters (from qubitcfg.json)
    twidth    : pulse duration in seconds
    amp       : peak amplitude as DAC fraction
    delta     : relative tolerance. If None (default), looked up in
                `_DEFAULT_DELTA[env_func]` with fallback to 1e-3. Pass an
                explicit value to override.
    eps       : ε floor. If None, uses max(1 DAC LSB, 0.5% * amp).
    fs        : DAC sample rate (default 1 GSPS)
    min_h_cycles : time-DAC minimum segment width; must match MIN_H_CYCLES
                   in spline_eval.sv.
    max_knots : safety cap on knot count (BRAM depth limit)
    refine    : enable plateau refinement

    Returns
    -------
    CompiledPulse
    """
    delta = _resolve_delta(env_func, delta)
    I_func, Q_func = make_envelope(env_func, twidth, amp, paradict)

    # Reasonable ε default: DAC LSB floor, lifted to 0.5% of peak so the
    # near-zero tails of DRAG/cos pulses don't cause pointless oversampling.
    dac_lsb = 2.0 / 2**16
    if eps is None:
        eps = max(dac_lsb, 0.005 * amp)

    min_dt = min_h_cycles / fs
    max_dt = MAX_H_CYCLES / fs   # 16-bit h_cycles cap — packing constraint
    grid_quantum = 1.0 / fs      # one sample period; HW knots must land here

    # Feature knots: junction points the spline MUST honor (e.g. ramp/hold
    # boundary in adiabatic_ramp). Empty for shapes without such features.
    seeds = feature_knots(env_func, twidth, paradict)

    # Compress both channels to a unified knot grid (same grid for I and Q —
    # the hardware uses one segment-width BRAM for both channels).
    compressed = compress_multichannel(
        {"I": I_func, "Q": Q_func},
        t_start=0.0,
        t_end=twidth,
        delta=delta,
        eps=eps,
        max_knots=max_knots,
        min_dt=min_dt,
        max_dt=max_dt,
        seed_knots=seeds if seeds else None,
        grid_quantum=grid_quantum,
        refine=refine,
        verbose=False,
    )

    cs_I = compressed["splines"]["I"]
    cs_Q = compressed["splines"]["Q"]

    coeff_words, width_words = pack_coefficients(
        cs_I, cs_Q, fs=fs, min_h_cycles=min_h_cycles, verbose=False
    )

    converged = all(r.converged for r in compressed["results"].values())

    return CompiledPulse(
        coeff_words=coeff_words,
        width_words=width_words,
        knots=compressed["knots"],
        n_segments=len(coeff_words),
        converged=converged,
        raw_samples_equivalent=int(round(twidth * fs)),
        env_func=env_func,
        paradict=dict(paradict),
        twidth=twidth,
        amp=amp,
    )


# ─────────────────────────────────────────────────────────────────────────────
def verify_pulse(cp: CompiledPulse, fs: float = FS, n_eval: int = 2000) -> dict:
    """
    Re-synthesize what the hardware will output (Q1.15 Horner simulated in
    float) and compare to the true pulse shape. Useful for smoke-testing a
    newly compiled pulse before uploading.

    Returns diagnostic dict with max/rms errors in DAC LSB units.
    """
    I_func, Q_func = make_envelope(cp.env_func, cp.twidth, cp.amp, cp.paradict)

    t_ref = np.linspace(0.0, cp.twidth, n_eval)
    I_true = I_func(t_ref)
    Q_true = Q_func(t_ref)

    # Simulate hardware evaluation at the exact u = u_cnt/h_cycles points
    I_hw = np.zeros_like(I_true)
    Q_hw = np.zeros_like(Q_true)

    Q_SCALE = 32768
    def _q15(raw16):
        raw16 &= 0xFFFF
        return (raw16 - 0x10000) / Q_SCALE if raw16 >= 0x8000 else raw16 / Q_SCALE

    for i, (cw, ww) in enumerate(zip(cp.coeff_words, cp.width_words)):
        h_cyc = ww & 0xFFFF
        t0, t1 = cp.knots[i], cp.knots[i + 1]
        mask = (t_ref >= t0) & (t_ref < t1)
        if i == cp.n_segments - 1:
            mask |= t_ref == t1

        aI = _q15((cw >> 64) & 0xFFFF); bI = _q15((cw >> 80) & 0xFFFF)
        cI = _q15((cw >> 96) & 0xFFFF); dI = _q15((cw >> 112) & 0xFFFF)
        aQ = _q15(cw & 0xFFFF);         bQ = _q15((cw >> 16) & 0xFFFF)
        cQ = _q15((cw >> 32) & 0xFFFF); dQ = _q15((cw >> 48) & 0xFFFF)

        u = (t_ref[mask] - t0) / (h_cyc / fs)
        I_hw[mask] = aI + u * (bI + u * (cI + u * dI))
        Q_hw[mask] = aQ + u * (bQ + u * (cQ + u * dQ))

    dac_lsb = 2.0 / 2**16
    return {
        "max_err_I_lsb": float(np.max(np.abs(I_true - I_hw)) / dac_lsb),
        "max_err_Q_lsb": float(np.max(np.abs(Q_true - Q_hw)) / dac_lsb),
        "rms_err_I_lsb": float(np.sqrt(np.mean((I_true - I_hw) ** 2)) / dac_lsb),
        "rms_err_Q_lsb": float(np.sqrt(np.mean((Q_true - Q_hw) ** 2)) / dac_lsb),
        "n_segments": cp.n_segments,
        "compression": cp.compression_ratio,
        "converged": cp.converged,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo: compile one of each pulse type and show diagnostics.
    print("Spline pulse compiler — demo")
    print("=" * 72)

    demos = [
        ("Q0 DRAG X90",  "DRAG", 32e-9,  0.12755,
         {"alpha": 0.5527, "sigmas": 3, "delta": -268e6}),
        ("Q0 readout",   "cos_edge_square", 2e-6, 0.2408,
         {"ramp_fraction": 0.25}),
        ("square ref",   "square", 1e-6, 0.5,
         {"phase": 0.0, "amplitude": 1.0}),
    ]

    fmt = "{:<18} {:>6} {:>8} {:>8} {:>8} {:>10} {:>6}"
    print(fmt.format("pulse", "segs", "ddr_B", "I_err", "Q_err", "comp", "conv"))
    print("-" * 72)

    for name, env, twidth, amp, paradict in demos:
        cp = compile_pulse(env, paradict, twidth, amp)
        v = verify_pulse(cp)
        print(fmt.format(
            name, cp.n_segments, cp.ddr_bytes,
            f"{v['max_err_I_lsb']:.1f}L", f"{v['max_err_Q_lsb']:.1f}L",
            f"{v['compression']:.1f}x", "Y" if v["converged"] else "N",
        ))
