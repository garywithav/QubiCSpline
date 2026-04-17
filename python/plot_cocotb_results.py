"""
plot_cocotb_results.py
======================
Visualizes cocotb testbench output vs Python CubicSpline reference
for both the readout (cos_edge_square) and DRAG X90 pulses.

Run AFTER the cocotb test:
    python run_cocotb.py          # produces .npz files in sim_build/
    python plot_cocotb_results.py # opens the plots

Figure 1 — Readout pulse (4 panels):
  Waveform overlay, absolute error, error histogram, zoomed ramp

Figure 2 — DRAG pulse (6 panels):
  I & Q waveform overlays, I & Q error, error histogram, error budget table
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_here = Path(__file__).parent          # python/
_root = _here.parent                   # project root
_sim  = _here / "sim_build"            # cocotb puts .npz here
_build = _root / "build"


def _find(name):
    for d in [_sim, _build, _here, _root]:
        p = d / name
        if p.exists():
            return p
    return None


def _load(name):
    p = _find(name)
    if p is None:
        return None
    return np.load(p)


dac_lsb = 2.0 / 2**16

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Readout cos_edge_square
# ─────────────────────────────────────────────────────────────────────────────
rd = _load("cocotb_results.npz")
if rd is not None:
    I_hw       = rd["I_hw"];       Q_hw       = rd["Q_hw"]
    I_ref      = rd["I_ref"];      Q_ref      = rd["Q_ref"]
    knots      = rd["knots"];      f_at_knots = rd["f_at_knots"]
    dac_lsb    = float(rd["dac_lsb"])
    n_samples  = len(I_hw)
    t_us       = np.linspace(knots[0], knots[-1], n_samples) * 1e6
    knots_us   = knots * 1e6
    err_I      = I_hw - I_ref
    abs_err    = np.abs(err_I)
    max_err    = np.max(abs_err)

    print(f"READOUT: {n_samples} samples, {len(knots)} knots, "
          f"max |I err| = {max_err/dac_lsb:.1f}× LSB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Readout cos_edge_square — spline_eval.sv vs Reference\n"
                 f"{n_samples} samples · {len(knots)} knots · "
                 f"max I error = {max_err/dac_lsb:.1f}× DAC LSB",
                 fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t_us, I_ref, 'steelblue', lw=1.5, alpha=0.7, label='Python CubicSpline (float64)')
    ax.plot(t_us, I_hw,  'r--',       lw=1.2, alpha=0.9, label='spline_eval.sv (Q1.15 Horner)')
    ylim = ax.get_ylim()
    ax.vlines(knots_us, *ylim, colors='green', alpha=0.15, lw=0.6)
    ax.scatter(knots_us, f_at_knots, s=14, c='green', zorder=5, label=f'{len(knots)} knots')
    ax.set_ylim(ylim)
    ax.set_xlabel('Time (µs)'); ax.set_ylabel('I amplitude')
    ax.set_title('I-channel: Reference vs Hardware'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_us, abs_err / dac_lsb, 'steelblue', lw=0.6, alpha=0.8)
    ax.axhline(1.0, color='orange', ls='--', lw=1.2, label='1 DAC LSB')
    ax.axhline(4.0, color='red',    ls='--', lw=1.2, label='4 DAC LSB (tolerance)')
    ax.set_xlabel('Time (µs)'); ax.set_ylabel('|error| (DAC LSBs)')
    ax.set_title('Absolute Error per Sample'); ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    bins = np.linspace(-4*dac_lsb, 4*dac_lsb, 81)
    ax.hist(err_I, bins=bins, color='steelblue', alpha=0.7, edgecolor='white', lw=0.5)
    ax.axvline(0, color='black', ls='-', lw=0.8)
    ax.axvline(-dac_lsb, color='orange', ls='--', lw=1, label='±1 LSB')
    ax.axvline( dac_lsb, color='orange', ls='--', lw=1)
    ax.set_xlabel('I error (DAC fraction)'); ax.set_ylabel('Count')
    ax.set_title('Error Distribution'); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ramp_end_us = knots_us[-1] * 0.25
    mask = t_us <= ramp_end_us
    ax.plot(t_us[mask], I_ref[mask], 'steelblue', lw=2, alpha=0.7, label='Reference')
    ax.plot(t_us[mask], I_hw[mask],  'r--', lw=1.5, alpha=0.9, label='Hardware')
    knot_mask = knots_us <= ramp_end_us
    ax.scatter(knots_us[knot_mask], f_at_knots[knot_mask], s=30, c='green', zorder=5, label='Knots')
    ax.set_xlabel('Time (µs)'); ax.set_ylabel('I amplitude')
    ax.set_title('Zoom: Rising Cosine Edge'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(_build / 'cocotb_results_readout.png', dpi=150)
    print(f"  -> cocotb_results_readout.png")
else:
    print("READOUT: cocotb_results.npz not found, skipping.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: DRAG X90
# ─────────────────────────────────────────────────────────────────────────────
dd = _load("cocotb_drag_results.npz")
if dd is not None:
    I_hw       = dd["I_hw"];       Q_hw       = dd["Q_hw"]
    I_ref      = dd["I_ref"];      Q_ref      = dd["Q_ref"]
    knots      = dd["knots"]
    I_at_knots = dd["I_at_knots"]; Q_at_knots = dd["Q_at_knots"]
    I_dense    = dd["I_dense"];    Q_dense    = dd["Q_dense"]
    t_dense    = dd["t_dense"]
    dac_lsb    = float(dd["dac_lsb"])
    n_samples  = len(I_hw)

    # Time axes
    t_ns      = np.linspace(knots[0], knots[-1], n_samples) * 1e9
    knots_ns  = knots * 1e9
    td_ns     = t_dense * 1e9

    err_I     = I_hw - I_ref;    err_Q     = Q_hw - Q_ref
    max_err_I = np.max(np.abs(err_I))
    max_err_Q = np.max(np.abs(err_Q))
    rms_I     = np.sqrt(np.mean(err_I**2))
    rms_Q     = np.sqrt(np.mean(err_Q**2))

    # Peak amplitudes for context
    peak_I = np.max(np.abs(I_ref))
    peak_Q = np.max(np.abs(Q_ref))

    print(f"\nDRAG X90: {n_samples} samples, {len(knots)} knots")
    print(f"  I: peak={peak_I:.4f}  max|err|={max_err_I:.2e} ({max_err_I/dac_lsb:.1f}× LSB)  "
          f"RMS={rms_I:.2e}")
    print(f"  Q: peak={peak_Q:.5f}  max|err|={max_err_Q:.2e} ({max_err_Q/dac_lsb:.1f}× LSB)  "
          f"RMS={rms_Q:.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DRAG X90 Pulse — spline_eval.sv vs Reference\n"
                 f"{n_samples} samples · {len(knots)} knots · "
                 f"max err: I={max_err_I/dac_lsb:.1f}× LSB, "
                 f"Q={max_err_Q/dac_lsb:.1f}× LSB",
                 fontsize=12, fontweight='bold')

    # (1) I-channel waveform
    ax = axes[0, 0]
    ax.plot(td_ns, I_dense, 'steelblue', lw=1.5, alpha=0.5, label='True signal (dense)')
    ax.plot(t_ns,  I_ref,   'steelblue', lw=1, alpha=0.8, label='CubicSpline ref')
    ax.plot(t_ns,  I_hw,    'r--',       lw=1.2, alpha=0.9, label='spline_eval.sv')
    ylim = ax.get_ylim()
    ax.vlines(knots_ns, *ylim, colors='green', alpha=0.2, lw=0.6)
    ax.scatter(knots_ns, I_at_knots, s=20, c='green', zorder=5, label=f'{len(knots)} knots')
    ax.set_ylim(ylim)
    ax.set_xlabel('Time (ns)'); ax.set_ylabel('I amplitude')
    ax.set_title('I-channel (Gaussian)'); ax.legend(fontsize=7)

    # (2) Q-channel waveform
    ax = axes[0, 1]
    ax.plot(td_ns, Q_dense, 'tomato', lw=1.5, alpha=0.5, label='True signal (dense)')
    ax.plot(t_ns,  Q_ref,   'tomato', lw=1, alpha=0.8, label='CubicSpline ref')
    ax.plot(t_ns,  Q_hw,    'b--',    lw=1.2, alpha=0.9, label='spline_eval.sv')
    ylim = ax.get_ylim()
    ax.vlines(knots_ns, *ylim, colors='green', alpha=0.2, lw=0.6)
    ax.scatter(knots_ns, Q_at_knots, s=20, c='green', zorder=5)
    ax.set_ylim(ylim)
    ax.set_xlabel('Time (ns)'); ax.set_ylabel('Q amplitude')
    ax.set_title(f'Q-channel (DRAG derivative)\npeak = {peak_Q:.5f}')
    ax.legend(fontsize=7)

    # (3) Error budget summary (text panel)
    ax = axes[0, 2]
    ax.axis('off')
    lines = [
        "Error Budget Analysis",
        "─" * 36,
        "",
        f"DAC LSB  = {dac_lsb:.2e}",
        f"Q1.15 precision = {1/32768:.2e}",
        "",
        "I-channel:",
        f"  Peak amplitude  = {peak_I:.5f}",
        f"  Max |error|     = {max_err_I:.2e}",
        f"                  = {max_err_I/dac_lsb:.1f}× DAC LSB",
        f"  RMS error       = {rms_I:.2e}",
        f"  Rel error (peak)= {max_err_I/peak_I:.2e}",
        "",
        "Q-channel:",
        f"  Peak amplitude  = {peak_Q:.5f}",
        f"  Max |error|     = {max_err_Q:.2e}",
        f"                  = {max_err_Q/dac_lsb:.1f}× DAC LSB",
        f"  RMS error       = {rms_Q:.2e}",
        f"  Rel error (peak)= {max_err_Q/peak_Q:.2e}" if peak_Q > 0 else "  (Q is zero)",
        "",
        "Expected error sources:",
        f"  Coeff quantization: ~1 LSB",
        f"  3× Horner multiply: ~1.5 LSB",
        f"  ─────────────────────",
        f"  Total expected:  ~2.5 LSB",
        f"  Observed max:    {max(max_err_I,max_err_Q)/dac_lsb:.1f} LSB  ✓"
            if max(max_err_I, max_err_Q) < 4*dac_lsb
            else f"  Observed max:    {max(max_err_I,max_err_Q)/dac_lsb:.1f} LSB  ✗ EXCEEDS BUDGET",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (4) I-channel error
    ax = axes[1, 0]
    ax.plot(t_ns, np.abs(err_I) / dac_lsb, 'steelblue', lw=0.8, alpha=0.8, label='I error')
    ax.axhline(1.0, color='orange', ls='--', lw=1, label='1 LSB')
    ax.axhline(4.0, color='red',    ls='--', lw=1, label='4 LSB (tol)')
    ax.set_xlabel('Time (ns)'); ax.set_ylabel('|error| (DAC LSBs)')
    ax.set_title('I-channel Absolute Error'); ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # (5) Q-channel error
    ax = axes[1, 1]
    ax.plot(t_ns, np.abs(err_Q) / dac_lsb, 'tomato', lw=0.8, alpha=0.8, label='Q error')
    ax.axhline(1.0, color='orange', ls='--', lw=1, label='1 LSB')
    ax.axhline(4.0, color='red',    ls='--', lw=1, label='4 LSB (tol)')
    ax.set_xlabel('Time (ns)'); ax.set_ylabel('|error| (DAC LSBs)')
    ax.set_title('Q-channel Absolute Error'); ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # (6) Combined error histogram
    ax = axes[1, 2]
    max_range = max(np.max(np.abs(err_I)), np.max(np.abs(err_Q)), 4*dac_lsb)
    bins = np.linspace(-max_range*1.2, max_range*1.2, 81)
    ax.hist(err_I, bins=bins, color='steelblue', alpha=0.6, label='I error', edgecolor='white', lw=0.3)
    ax.hist(err_Q, bins=bins, color='tomato',    alpha=0.6, label='Q error', edgecolor='white', lw=0.3)
    ax.axvline(0, color='black', ls='-', lw=0.8)
    ax.axvline(-dac_lsb, color='orange', ls='--', lw=1, label='±1 LSB')
    ax.axvline( dac_lsb, color='orange', ls='--', lw=1)
    ax.set_xlabel('Error (DAC fraction)'); ax.set_ylabel('Count')
    ax.set_title('Error Distribution (I & Q)'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(_build / 'cocotb_results_drag.png', dpi=150)
    print(f"  -> cocotb_results_drag.png")
else:
    print("\nDRAG: cocotb_drag_results.npz not found, skipping.")


if rd is not None or dd is not None:
    plt.show()
else:
    print("\nNo result files found. Run: python run_cocotb.py")
    sys.exit(1)
