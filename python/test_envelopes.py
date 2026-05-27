"""
test_envelopes.py
=================
Pure-Python unit tests for pulse envelope compression. Calls compile_pulse
on each new envelope type, simulates the Q1.15 hardware in float
(via verify_pulse), and asserts on compression + error budget.

Distinct from test_spline_eval.py (cocotb, runs inside the simulator):
these tests run in plain Python without Icarus, so they're fast and
suitable for adding to a CI loop or running after every code change.

Run:
    cd python
    python test_envelopes.py

Hard failure conditions (per Phase A spec):
    - max HW error > 4 LSB on either channel
    - cp.converged == False
Compression targets are HYPOTHESES — the actual value is reported but
not asserted on, so we learn whether the hypothesis holds.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

from spline_pulse_compiler import compile_pulse, verify_pulse


# ── Test infrastructure ─────────────────────────────────────────────────────
def _run_case(
    name: str,
    env_func: str,
    paradict: dict,
    twidth: float,
    amp: float,
    compression_hypothesis: float,
    *,
    pre_check: Callable | None = None,
) -> bool:
    """
    Compile + verify one pulse, print compression / error / convergence.
    Returns True on pass, False on hard failure.

    A test "passes" if:
      - compile_pulse did not raise
      - cp.converged == True
      - max_err_I_lsb < 4 and max_err_Q_lsb < 4
      - optional pre_check(cp) callback passes (or no callback)

    Compression hypothesis is for reporting only — under-performance is
    flagged but does not fail the test.
    """
    print(f"\n── {name} ──")
    print(f"  env_func={env_func}  twidth={twidth*1e9:.0f}ns  amp={amp}")
    print(f"  paradict={paradict}")

    try:
        cp = compile_pulse(env_func, paradict, twidth, amp)
    except Exception as exc:
        print(f"  FAIL: compile_pulse raised {type(exc).__name__}: {exc}")
        return False

    v = verify_pulse(cp)
    comp_hyp_note = (
        ""
        if cp.compression_ratio >= compression_hypothesis
        else f" (below {compression_hypothesis:.1f}× hypothesis — investigate)"
    )

    print(f"  n_segments  = {cp.n_segments}")
    print(f"  compression = {cp.compression_ratio:.1f}×{comp_hyp_note}")
    print(f"  err_I       = {v['max_err_I_lsb']:.2f} LSB  (rms {v['rms_err_I_lsb']:.2f})")
    print(f"  err_Q       = {v['max_err_Q_lsb']:.2f} LSB  (rms {v['rms_err_Q_lsb']:.2f})")
    print(f"  converged   = {cp.converged}")

    # Pulses >25% of BRAM depth get a soft warning.
    BRAM_DEPTH = 4096
    if cp.n_segments > 0.25 * BRAM_DEPTH:
        print(
            f"  WARNING: this pulse uses {cp.n_segments}/{BRAM_DEPTH} segments "
            f"({100*cp.n_segments/BRAM_DEPTH:.1f}% of BRAM)"
        )

    ok = True
    if not cp.converged:
        print("  FAIL: cp.converged == False")
        ok = False
    if v["max_err_I_lsb"] >= 4.0:
        print(f"  FAIL: I-channel error {v['max_err_I_lsb']:.2f} LSB ≥ 4")
        ok = False
    if v["max_err_Q_lsb"] >= 4.0:
        print(f"  FAIL: Q-channel error {v['max_err_Q_lsb']:.2f} LSB ≥ 4")
        ok = False
    if pre_check is not None:
        try:
            pre_check(cp)
        except AssertionError as exc:
            print(f"  FAIL pre_check: {exc}")
            ok = False

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ── Tests ──────────────────────────────────────────────────────────────────
def test_adiabatic_ramp() -> bool:
    """
    Adiabatic minimum-jerk ramp + hold. Reference math from Pagano, Motzoi
    et al., arXiv:2412.15173 (Dec 2024).

    History — please read before raising the hypothesis:
    -------------------------------------------------------------------
    The initial Phase A spec guessed 20× compression for this shape.
    Measured reality at the project default delta=1e-3 was only 2.6×,
    and the HW error was 6.5 LSB (above the 4 LSB project budget). After
    tightening to delta=3e-4 (now the per-shape default in
    spline_pulse_compiler._DEFAULT_DELTA), the shape compiles at:

        n_segs = 29   compression = 2.2×   HW err = 3.60 LSB

    Why the algorithmic ratio is modest for ramp+hold shapes: the spline
    must over-resolve the steep last-6%-of-ramp region (where the
    minimum-jerk 5th-order term peaks curvature) AND honor the
    C²-discontinuity junction at t=ramp_time. Twenty-nine cubic segments
    for one thousand source samples isn't a huge win compared to the
    readout pulse (40 segments for 2000 samples, 3.1× algorithmic).
    Memory savings vs raw QubiC envelope storage are still very real —
    a few-× compression on long pulses adds up across a calibration
    table — but the shape family is just not as compressible as the
    cos²-edge readout that we benchmarked against early.

    The 20× → 2.2× delta is not a failure: it's measurement correcting
    an unmotivated guess, and a useful data point for what compresses
    well (smooth shapes with broad flat regions, e.g. cos_edge_square)
    vs what doesn't (shapes with localized high-curvature features +
    long flat tails). Atom transport, with its smooth-everywhere
    polynomial and no junction, is expected to compress dramatically
    better; rydberg_drag is somewhere in between.
    """
    return _run_case(
        name="adiabatic_ramp (1 µs, 200 ns ramp)",
        env_func="adiabatic_ramp",
        paradict={"ramp_time": 200e-9},
        twidth=1e-6,
        amp=0.5,
        compression_hypothesis=2.1,  # measured 2.155 at delta=3e-4
    )


def test_rydberg_drag() -> bool:
    """
    Rydberg DRAG (Levine/Pichler form): Gaussian amplitude + linearly
    chirped detuning. Channel 2 here is DETUNING (not quadrature) — see
    pulse_envelopes._rydberg_drag for the semantic shift.

    Compression hypothesis was 5× initially (Phase A spec); measured
    reality at the per-shape default delta=5e-4 is 3.5×. Like
    adiabatic_ramp, the original number was speculation — actual is what
    matters. The Q channel (linear chirp) fits in 1 segment regardless;
    all error budget lives on the I-channel Gaussian shoulders. The cliff
    is sharp: 1e-3 → 16 LSB error, 5e-4 → 1.2 LSB. delta tuned to the
    cheapest side of the cliff.
    """
    return _run_case(
        name="rydberg_drag (1 µs, σ=200 ns, linear chirp)",
        env_func="rydberg_drag",
        paradict={
            "omega_max": 2 * 3.14159 * 1e6,   # 1 MHz (rad/s, informational)
            "sigma": 200e-9,
            "detuning_max": 0.3,               # DAC fraction
            "chirp_shape": "linear",
        },
        twidth=1e-6,
        amp=0.8,
        compression_hypothesis=3.4,  # measured 3.5× at delta=5e-4
    )


def test_atom_transport_short() -> bool:
    """
    Atom transport at the short end of the design range (100 µs).

    At this twidth the 16-bit h_cycles cap doesn't bind (max segment is
    well below 65.5 µs), so delta is the only thing controlling fit
    accuracy. The per-shape default delta=5e-4 gives 417× compression
    at ~2 LSB error.
    """
    return _run_case(
        name="atom_transport (100 µs, x: 0 → 0.7)",
        env_func="atom_transport",
        paradict={"x_start": 0.0, "x_end": 0.7},
        twidth=100e-6,
        amp=1.0,
        compression_hypothesis=400.0,  # measured 417× at delta=5e-4
    )


def test_atom_transport_long() -> bool:
    """
    Atom transport at the long end (1 ms). Specifically exercises:

    1. The d-coefficient overflow check at pack time (added Phase A but
       d_hw stays comfortably under Q1.15 range here, so the check is a
       safety net rather than a triggered constraint).

    2. The 16-bit h_cycles packing cap (MAX_H_CYCLES=65535). At twidth=1ms
       and delta=1e-3 the algorithm WANTS only ~13 segments, but max_dt
       forces it to 23 (max segment ~62500 cycles, just under the cap).
       This sets compression to ~2700×, much better than any other shape
       in the project but lower than the naive 1000-samples/13-segments
       count would suggest.

    Hypothesis was 100× pre-measurement. Actual is 2717× — the original
    guess was conservative by ~30×. atom_transport is the killer-app
    use case for spline compression and the numbers reflect it.
    """
    return _run_case(
        name="atom_transport (1 ms, x: 0 → 0.7)",
        env_func="atom_transport",
        paradict={"x_start": 0.0, "x_end": 0.7},
        twidth=1e-3,
        amp=1.0,
        compression_hypothesis=2000.0,  # measured 2717× at delta=5e-4
    )


# ── Main ────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    ("adiabatic_ramp",       test_adiabatic_ramp),
    ("rydberg_drag",         test_rydberg_drag),
    ("atom_transport_short", test_atom_transport_short),
    ("atom_transport_long",  test_atom_transport_long),
]


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 72)
    print("pulse-envelope unit tests")
    print("=" * 72)

    results = {}
    for name, fn in ALL_TESTS:
        if only is not None and only != name:
            continue
        try:
            results[name] = fn()
        except Exception:
            print(f"\n── {name} ──")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 72)
    print("summary")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:<24}  {'PASS' if ok else 'FAIL'}")
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} passed")
    sys.exit(0 if n_pass == n_total else 1)
