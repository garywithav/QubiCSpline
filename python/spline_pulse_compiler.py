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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from autoknots import autoknots, compress_multichannel
from pulse_envelopes import make_envelope
from spline_coeff_pack import pack_coefficients, MIN_H_CYCLES, FS


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
    delta: float = 1e-3,
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
    env_func  : 'DRAG', 'cos_edge_square', 'square', 'mark'
    paradict  : envelope-specific parameters (from qubitcfg.json)
    twidth    : pulse duration in seconds
    amp       : peak amplitude as DAC fraction
    delta     : relative tolerance (default 0.1%)
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
    I_func, Q_func = make_envelope(env_func, twidth, amp, paradict)

    # Reasonable ε default: DAC LSB floor, lifted to 0.5% of peak so the
    # near-zero tails of DRAG/cos pulses don't cause pointless oversampling.
    dac_lsb = 2.0 / 2**16
    if eps is None:
        eps = max(dac_lsb, 0.005 * amp)

    min_dt = min_h_cycles / fs

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
