"""
test_constraints.py
===================
Adversarial regression tests for the three hardware-aware constraint
mechanisms in autoknots(): min_dt, max_dt, grid_quantum.

The lesson from Phase A: a passing test is only as good as what it would
catch failing. Tests like the readout cocotb test PASSED for years while
the fractional-ns / h_cycles-wrap bugs were latent — because the readout
happened to use integer-ns uniform spacings and modest segment counts,
the bugs were never exercised.

Each test in this file:

  1. Constructs a pulse that would fail in a specific, attributable
     way if the corresponding constraint were removed
  2. Verifies the constraint mechanism does its job (with constraint on)
  3. Demonstrates the failure mode (with constraint off) — proves the
     test is adversarial, not coincidentally passing

Run:
    cd python
    python test_constraints.py

If anyone later removes one of the three constraint paths from
autoknots(), exactly one of these tests fails with a clear, readable
assertion that points at the right line.
"""

from __future__ import annotations

import sys
import traceback
import warnings

import numpy as np

from autoknots import autoknots, compress_multichannel
from pulse_envelopes import make_envelope
from spline_coeff_pack import (
    FS, MIN_H_CYCLES, MAX_H_CYCLES, Q_SCALE, pack_coefficients,
)
from spline_pulse_compiler import compile_pulse, verify_pulse


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
DAC_LSB = 2.0 / 2**16


def _q15_to_float(raw16: int) -> float:
    raw16 &= 0xFFFF
    if raw16 >= 0x8000:
        return (raw16 - 0x10000) / Q_SCALE
    return raw16 / Q_SCALE


def _simulate_hw(coeff_words, width_words, knots, t_eval, fs=FS):
    """
    Re-run the Q1.15 Horner pipeline in float for arbitrary BRAM words.
    Used by tests that need to evaluate a custom-compiled pulse without
    going through compile_pulse (e.g. when we deliberately disable
    grid_quantum to reproduce the fractional-ns bug).
    """
    out = np.zeros_like(t_eval, dtype=float)
    for i, (cw, ww) in enumerate(zip(coeff_words, width_words)):
        h_cyc = ww & 0xFFFF
        t0 = knots[i]
        t1 = knots[i + 1]
        mask = (t_eval >= t0) & (t_eval < t1)
        if i == len(coeff_words) - 1:
            mask |= (t_eval == t1)
        # I-channel only — these tests don't exercise Q
        a = _q15_to_float((cw >> 64) & 0xFFFF)
        b = _q15_to_float((cw >> 80) & 0xFFFF)
        c = _q15_to_float((cw >> 96) & 0xFFFF)
        d = _q15_to_float((cw >> 112) & 0xFFFF)
        u = (t_eval[mask] - t0) / (h_cyc / fs)
        out[mask] = a + u * (b + u * (c + u * d))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial pulse: very-narrow Gaussian spike inside a longer twidth.
# Designed to push AutoKnots midpoint bisection into sub-ns territory at
# the spike center. Used by the min_dt halt test.
# ─────────────────────────────────────────────────────────────────────────────
def _narrow_spike(twidth: float, amp: float, sigma: float):
    """Return I_func, Q_func for a Gaussian spike of width sigma centered
    in the pulse. sigma=1ns in twidth=32ns is sharp enough that AutoKnots
    would want fractional-ns spacing absent the min_dt floor."""
    t_c = twidth / 2.0

    def I_func(t):
        t = np.asarray(t, dtype=float)
        return amp * np.exp(-0.5 * ((t - t_c) / sigma) ** 2)

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: min_dt halts on a steep pulse rather than diverging
# ─────────────────────────────────────────────────────────────────────────────
def test_min_dt_halts() -> bool:
    """
    A Gaussian spike with σ = 1 ns inside twidth = 32 ns has curvature
    steep enough that AutoKnots' midpoint bisection wants to descend to
    sub-ns spacing at the peak.

    With min_dt active, the algorithm MUST halt in finite iterations,
    emit the "frozen" warning, and return a result whose knot spacing is
    everywhere >= min_dt. Without min_dt, bisection runs all the way
    down to grid_quantum (1 ns), proving the test pulse genuinely
    exercises sub-min_dt subdivision.

    Removing min_dt from autoknots would cause this test to fail at
    assertion #1 (some spacing < 4 ns).
    """
    print("\n── min_dt halt on σ=1ns spike in 32ns ──")
    I_func, _ = _narrow_spike(twidth=32e-9, amp=0.5, sigma=1e-9)

    # WITH min_dt — should halt with frozen warning, all spacings >= 4 ns
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = autoknots(
            I_func, 0.0, 32e-9,
            delta=1e-3, eps=DAC_LSB,
            min_dt=MIN_H_CYCLES / FS,
            grid_quantum=1.0 / FS,
        )
    spacings = np.diff(res.knots)
    min_spacing = float(np.min(spacings))
    floor = MIN_H_CYCLES / FS
    print(f"  WITH min_dt={floor*1e9:.0f}ns:  n_knots={res.n_knots:>3}  "
          f"min spacing = {min_spacing*1e9:.2f} ns")
    if min_spacing < floor - 1e-15:
        print(f"  FAIL: knot spacing {min_spacing*1e9:.3f} ns < "
              f"min_dt {floor*1e9:.0f} ns")
        return False
    # Frozen warning should have fired
    frozen_msgs = [w for w in caught if "min_dt" in str(w.message)]
    if not frozen_msgs:
        print("  FAIL: expected min_dt frozen warning, none seen")
        return False
    print(f"  frozen warning fired: '{str(frozen_msgs[0].message)[:60]}...'")

    # WITHOUT min_dt (control) — should produce spacings < 4 ns somewhere,
    # proving the pulse really does want sub-min_dt resolution.
    res_ctrl = autoknots(
        I_func, 0.0, 32e-9,
        delta=1e-3, eps=DAC_LSB,
        min_dt=0.0,
        grid_quantum=1.0 / FS,   # keep on so it doesn't go fractional
    )
    spacings_ctrl = np.diff(res_ctrl.knots)
    min_spacing_ctrl = float(np.min(spacings_ctrl))
    print(f"  CTRL (min_dt=0):       n_knots={res_ctrl.n_knots:>3}  "
          f"min spacing = {min_spacing_ctrl*1e9:.2f} ns")
    if min_spacing_ctrl >= floor:
        print(f"  FAIL: control pulse didn't exercise sub-min_dt "
              f"subdivision (min spacing {min_spacing_ctrl*1e9:.2f} ns "
              f">= floor {floor*1e9:.0f} ns). Test is not adversarial.")
        return False
    print(f"  ✓ control exhibits {sum(spacings_ctrl < floor)} sub-{floor*1e9:.0f}ns spacings")
    print("  PASS")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: max_dt prevents the 16-bit h_cycles wrap
# ─────────────────────────────────────────────────────────────────────────────
def test_max_dt_prevents_h_cycles_wrap() -> bool:
    """
    Atom transport at twidth = 1 ms with delta = 1e-3 wants ~13 segments,
    averaging 77 µs each — well above the 16-bit MAX_H_CYCLES = 65535
    cycles = 65.5 µs cap on the width-BRAM field.

    With max_dt active, autoknots force-subdivides any segment that would
    overflow, producing 23 segments all below the cap. The pulse packs
    cleanly and HW-vs-truth error stays under 4 LSB.

    Without max_dt, autoknots converges at 13 segments and the packer
    catches the violation defensively — proving both the algorithm path
    (max_dt) and the pack-time safety net would each individually fail
    if the other were removed.

    Removing max_dt from autoknots would cause this test to fail at
    assertion #2 (pack_coefficients raises).
    """
    print("\n── max_dt prevents 16-bit h_cycles wrap (1ms atom_transport) ──")
    twidth = 1e-3
    paradict = {"x_start": 0.0, "x_end": 0.7}

    # WITH max_dt — full compile_pulse path. Should succeed with
    # HW error < 4 LSB.
    cp = compile_pulse("atom_transport", paradict, twidth, amp=1.0)
    max_h_cyc = max(w & 0xFFFF for w in cp.width_words)
    print(f"  WITH max_dt:  n_segs = {cp.n_segments:>3}  "
          f"max h_cyc = {max_h_cyc}  (cap = {MAX_H_CYCLES})")
    if max_h_cyc > MAX_H_CYCLES:
        print(f"  FAIL: max h_cyc {max_h_cyc} > cap {MAX_H_CYCLES}")
        return False
    v = verify_pulse(cp)
    print(f"                HW err = {v['max_err_I_lsb']:.2f} LSB")
    if v["max_err_I_lsb"] >= 4.0:
        print(f"  FAIL: HW error {v['max_err_I_lsb']:.2f} >= 4 LSB")
        return False

    # WITHOUT max_dt — autoknots converges with too-wide segments, then
    # pack_coefficients raises. We pack it manually using a no-max_dt
    # compression so we can observe both:
    #   (a) some segment exceeds MAX_H_CYCLES
    #   (b) pack_coefficients refuses to silently emit the wrap
    I_func, Q_func = make_envelope("atom_transport", twidth, 1.0, paradict)
    compressed_ctrl = compress_multichannel(
        {"I": I_func, "Q": Q_func},
        t_start=0.0, t_end=twidth,
        delta=1e-3, eps=DAC_LSB,
        min_dt=MIN_H_CYCLES / FS,
        max_dt=None,                 # ← disabled
        grid_quantum=1.0 / FS,
    )
    cs_I_ctrl = compressed_ctrl["splines"]["I"]
    cs_Q_ctrl = compressed_ctrl["splines"]["Q"]
    h_cyc_ctrl = np.round(np.diff(cs_I_ctrl.x) * FS).astype(int)
    max_h_ctrl = int(h_cyc_ctrl.max())
    print(f"  CTRL (no max_dt):  n_segs = {len(h_cyc_ctrl):>3}  "
          f"max h_cyc = {max_h_ctrl}")
    if max_h_ctrl <= MAX_H_CYCLES:
        print(f"  FAIL: control didn't produce a too-wide segment "
              f"(max h_cyc = {max_h_ctrl}). Test is not adversarial.")
        return False
    # And pack_coefficients should refuse to pack it.
    try:
        pack_coefficients(cs_I_ctrl, cs_Q_ctrl, verbose=False)
    except ValueError as exc:
        if "MAX_H_CYCLES" not in str(exc):
            print(f"  FAIL: pack raised wrong exception: {exc}")
            return False
        print(f"  ✓ pack-time safety net raised: '{str(exc).splitlines()[0][:70]}...'")
    else:
        print("  FAIL: pack_coefficients should have raised on >MAX_H_CYCLES h_cyc")
        return False
    print("  PASS")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: grid_quantum eliminates fractional-ns knot placement
# ─────────────────────────────────────────────────────────────────────────────
def test_grid_quantum_eliminates_fractional_ns() -> bool:
    """
    adiabatic_ramp at twidth = 1 µs, ramp_time = 200 ns: AutoKnots'
    bisection chain produces midpoints at 200, 100, 50, 25, 12.5, 6.25 ns
    — the last two are fractional ns. Before grid_quantum was introduced,
    those fractional knots silently produced a 4% mismatch between the
    float spline (built with the fractional knots) and the HW evaluation
    (using h_cycles = round(h_s * fs), an integer). The mismatch surfaced
    as ~44 LSB HW error.

    With grid_quantum active, every proposed knot is snapped to the
    integer-ns grid before the local error check runs, so the float
    spline and the HW evaluation live on the same grid. HW error stays
    < 4 LSB.

    Removing grid_quantum from autoknots would cause this test to fail
    at assertion #1 (fractional knots present) or assertion #2 (HW
    error > 4 LSB).
    """
    print("\n── grid_quantum eliminates fractional-ns knots (adiabatic_ramp) ──")
    twidth = 1e-6
    paradict = {"ramp_time": 200e-9}
    amp = 0.5
    I_func, Q_func = make_envelope("adiabatic_ramp", twidth, amp, paradict)
    seeds = [paradict["ramp_time"]]   # the C²-junction seed

    # WITH grid_quantum — every knot must land on integer ns
    compressed_on = compress_multichannel(
        {"I": I_func, "Q": Q_func},
        t_start=0.0, t_end=twidth,
        delta=3e-4, eps=max(DAC_LSB, 0.005 * amp),
        min_dt=MIN_H_CYCLES / FS,
        max_dt=MAX_H_CYCLES / FS,
        seed_knots=seeds,
        grid_quantum=1.0 / FS,
    )
    knots_on = compressed_on["knots"]
    frac_on = np.max(np.abs(knots_on * FS - np.round(knots_on * FS)))
    print(f"  WITH grid_quantum:  n_knots={len(knots_on):>3}  "
          f"max |fractional ns| = {frac_on:.3e}")
    if frac_on > 1e-6:
        print(f"  FAIL: knots not on integer-ns grid")
        return False

    cw_on, ww_on = pack_coefficients(
        compressed_on["splines"]["I"], compressed_on["splines"]["Q"],
        verbose=False,
    )
    t_eval = np.linspace(0.0, twidth, 20000)
    I_true = I_func(t_eval)
    I_hw_on = _simulate_hw(cw_on, ww_on, knots_on, t_eval)
    err_on_lsb = float(np.max(np.abs(I_hw_on - I_true))) / DAC_LSB
    print(f"                      HW err = {err_on_lsb:.2f} LSB")
    if err_on_lsb >= 4.0:
        print(f"  FAIL: with grid_quantum, HW err {err_on_lsb:.2f} LSB still >= 4")
        return False

    # WITHOUT grid_quantum — fractional knots should appear AND
    # HW error should blow up. We use compress_multichannel directly
    # since compile_pulse always sets grid_quantum.
    compressed_off = compress_multichannel(
        {"I": I_func, "Q": Q_func},
        t_start=0.0, t_end=twidth,
        delta=3e-4, eps=max(DAC_LSB, 0.005 * amp),
        min_dt=MIN_H_CYCLES / FS,
        max_dt=MAX_H_CYCLES / FS,
        seed_knots=seeds,
        grid_quantum=None,             # ← disabled
    )
    knots_off = compressed_off["knots"]
    frac_off = np.max(np.abs(knots_off * FS - np.round(knots_off * FS)))
    print(f"  CTRL (no grid_quantum): n_knots={len(knots_off):>3}  "
          f"max |fractional ns| = {frac_off:.3e}")
    if frac_off < 1e-3:
        print(f"  FAIL: control didn't produce fractional knots. "
              f"Test is not adversarial.")
        return False

    cw_off, ww_off = pack_coefficients(
        compressed_off["splines"]["I"], compressed_off["splines"]["Q"],
        verbose=False,
    )
    I_hw_off = _simulate_hw(cw_off, ww_off, knots_off, t_eval)
    err_off_lsb = float(np.max(np.abs(I_hw_off - I_true))) / DAC_LSB
    print(f"                          HW err = {err_off_lsb:.1f} LSB")
    if err_off_lsb <= 4.0:
        print(f"  FAIL: control HW err {err_off_lsb:.2f} <= 4 LSB. "
              f"Test is not adversarial.")
        return False
    print(f"  ✓ disabling grid_quantum drives HW err from "
          f"{err_on_lsb:.1f} → {err_off_lsb:.1f} LSB")
    print("  PASS")
    return True


# ─────────────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    ("min_dt halt",                    test_min_dt_halts),
    ("max_dt prevents h_cycles wrap",  test_max_dt_prevents_h_cycles_wrap),
    ("grid_quantum eliminates frac-ns", test_grid_quantum_eliminates_fractional_ns),
]


if __name__ == "__main__":
    print("=" * 72)
    print("adversarial constraint tests")
    print("=" * 72)
    results = {}
    for name, fn in ALL_TESTS:
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
        print(f"  {name:<40}  {'PASS' if ok else 'FAIL'}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"\n  {n_pass}/{len(results)} passed")
    sys.exit(0 if n_pass == len(results) else 1)
