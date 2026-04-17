"""
plot_min_dt_comparison.py
=========================
Shows how min_dt (the minimum time-DAC spacing floor) affects spline fit
quality for the short DRAG pulse. Two fits are produced:

  1. min_dt = 4 ns  (current, what the hardware can actually resolve)
  2. min_dt = 0     (unconstrained — algorithmic best case, can't go to HW)

Both are evaluated against the true DRAG signal. This is a Python-only
comparison — it does not go through spline_eval.sv.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from autoknots import autoknots
from pulse_envelopes import make_envelope


# Q0 DRAG X90 parameters from qubitcfg.json
TWIDTH   = 32e-9
AMP      = 0.12755
PARADICT = {"alpha": 0.5527, "sigmas": 3, "delta": -268e6}

I_func, Q_func = make_envelope("DRAG", TWIDTH, AMP, PARADICT)
t_dense = np.linspace(0, TWIDTH, 50_000)
I_true  = I_func(t_dense)
Q_true  = Q_func(t_dense)

# Fit with each min_dt
configs = [
    ("min_dt = 4 ns  (14-bit time DAC floor)", 4e-9, "tomato"),
    ("min_dt = 0     (unconstrained)",          0.0,  "mediumseagreen"),
]

results = []
for label, min_dt, color in configs:
    res_I = autoknots(I_func, 0, TWIDTH, delta=1e-3, eps=1e-3,
                      min_dt=min_dt, verbose=False)
    res_Q = autoknots(Q_func, 0, TWIDTH, delta=1e-3, eps=1e-3,
                      min_dt=min_dt, verbose=False)
    results.append((label, min_dt, color, res_I, res_Q))
    print(f"{label}:  I knots={res_I.n_knots}  Q knots={res_Q.n_knots}")


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("DRAG X90 — effect of min_dt on spline fit quality\n"
             "(32 ns pulse, delta=1e-3; hardware uses top row)",
             fontsize=12, fontweight='bold')

# Row 1: I channel
ax = axes[0, 0]
ax.plot(t_dense*1e9, I_true, 'steelblue', lw=2, alpha=0.6, label='True signal')
for label, min_dt, color, res_I, _ in results:
    t_eval = np.linspace(0, TWIDTH, 2000)
    ax.plot(t_eval*1e9, res_I.evaluate(t_eval), color=color, lw=1.3,
            linestyle='--', label=f'{label}  [{res_I.n_knots} knots]')
    ax.scatter(res_I.knots*1e9, res_I.f_at_knots, s=20, c=color,
               edgecolors='black', linewidths=0.5, zorder=5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('I amplitude')
ax.set_title('I channel (Gaussian)')
ax.legend(fontsize=8, loc='upper right')

ax = axes[0, 1]
for label, min_dt, color, res_I, _ in results:
    t_eval = np.linspace(0, TWIDTH, 2000)
    err = np.abs(I_func(t_eval) - res_I.evaluate(t_eval))
    ax.semilogy(t_eval*1e9, err, color=color, lw=1.0, label=label)
ax.axhline(2/2**16, color='black', ls=':', lw=1, label='1 DAC LSB')
ax.set_xlabel('Time (ns)')
ax.set_ylabel('|error| (fit vs true)')
ax.set_title('I channel fit error')
ax.legend(fontsize=8)

# Row 2: Q channel
ax = axes[1, 0]
ax.plot(t_dense*1e9, Q_true, 'tomato', lw=2, alpha=0.6, label='True signal')
for label, min_dt, color, res_I, res_Q in results:
    t_eval = np.linspace(0, TWIDTH, 2000)
    ax.plot(t_eval*1e9, res_Q.evaluate(t_eval), color=color, lw=1.3,
            linestyle='--', label=f'{label}  [{res_Q.n_knots} knots]')
    ax.scatter(res_Q.knots*1e9, res_Q.f_at_knots, s=20, c=color,
               edgecolors='black', linewidths=0.5, zorder=5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Q amplitude')
ax.set_title('Q channel (DRAG derivative)')
ax.legend(fontsize=8, loc='upper right')

ax = axes[1, 1]
for label, min_dt, color, _, res_Q in results:
    t_eval = np.linspace(0, TWIDTH, 2000)
    err = np.abs(Q_func(t_eval) - res_Q.evaluate(t_eval))
    ax.semilogy(t_eval*1e9, err, color=color, lw=1.0, label=label)
ax.axhline(2/2**16, color='black', ls=':', lw=1, label='1 DAC LSB')
ax.set_xlabel('Time (ns)')
ax.set_ylabel('|error| (fit vs true)')
ax.set_title('Q channel fit error')
ax.legend(fontsize=8)

plt.tight_layout()
out = Path(__file__).parent.parent / "build" / "min_dt_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f"\nSaved {out}")
plt.show()
