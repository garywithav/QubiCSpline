"""
pulse_envelopes.py
==================
QubiC envelope generators. One function per env_func name seen in qubitcfg.json.

Each generator returns (I_func, Q_func) — callables f(t: ndarray) -> ndarray —
so they can be fed directly into autoknots().

Channel-2 (Q) semantic note
---------------------------
Historically "Q" meant the quadrature component of a complex IQ envelope.
For neutral-atom envelopes added in the Lukin-group extension, channel 2 is
repurposed depending on the envelope — see each generator's docstring for
its physical meaning. The spline engine is channel-agnostic: it evaluates
whatever polynomial it's given. Downstream gateware must know what each
channel represents per env_func.

  - DRAG:             channel 2 = quadrature (DRAG derivative correction)
  - cos_edge_square:  channel 2 = unused (zeros)
  - square:           channel 2 = unused (zeros)
  - mark:             channel 2 = unused (zeros)
  - adiabatic_ramp:   channel 2 = unused (zeros)
  - rydberg_drag:     channel 2 = DETUNING (linear chirp), NOT quadrature
  - atom_transport:   channel 2 = unused (zeros)

Public API:
    make_envelope(env_func, twidth, amp, paradict) -> (I_func, Q_func)

Supported env_func values:
    DRAG, cos_edge_square, square, mark,
    adiabatic_ramp, rydberg_drag, atom_transport
"""

from __future__ import annotations

import numpy as np


def _drag(twidth: float, amp: float, paradict: dict):
    """
    DRAG envelope (Derivative Removal by Adiabatic Gate).
    Gaussian I + scaled derivative Q to suppress |f> leakage.

    paradict keys: alpha, sigmas, delta (anharmonicity, Hz)
    """
    alpha = float(paradict["alpha"])
    sigmas = float(paradict["sigmas"])
    delta_hz = float(paradict["delta"])

    t_c = twidth / 2.0
    sigma = twidth / (2.0 * sigmas)

    def I_func(t):
        t = np.asarray(t, dtype=float)
        return amp * np.exp(-0.5 * ((t - t_c) / sigma) ** 2)

    def Q_func(t):
        t = np.asarray(t, dtype=float)
        gauss = np.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        dIdt = -(t - t_c) / sigma ** 2 * gauss
        return amp * (alpha / (delta_hz * 2.0 * np.pi)) * dIdt

    return I_func, Q_func


def _cos_edge_square(twidth: float, amp: float, paradict: dict):
    """
    Raised-cosine-edge square pulse (readout drive).

    paradict keys: ramp_fraction
    """
    rf = float(paradict["ramp_fraction"])
    ramp_t = twidth * rf

    def I_func(t):
        t = np.asarray(t, dtype=float)
        env = np.empty_like(t)
        rising = t < ramp_t
        flat = (t >= ramp_t) & (t <= twidth - ramp_t)
        falling = t > twidth - ramp_t
        env[rising] = 0.5 * (1 - np.cos(np.pi * t[rising] / ramp_t))
        env[flat] = 1.0
        env[falling] = 0.5 * (
            1 + np.cos(np.pi * (t[falling] - (twidth - ramp_t)) / ramp_t)
        )
        return amp * env

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


def _square(twidth: float, amp: float, paradict: dict):
    """
    Hard square pulse. paradict keys: phase, amplitude (amplitude multiplies amp).
    Mostly used for alignment/calibration tones.
    """
    amplitude = float(paradict.get("amplitude", 1.0))

    def I_func(t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, amp * amplitude)

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


def _mark(twidth: float, amp: float, paradict: dict):
    """
    Marker pulse — constant amplitude, no shape. paradict is empty.
    """
    def I_func(t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, amp)

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


def _adiabatic_ramp(twidth: float, amp: float, paradict: dict):
    """
    Adiabatic ramp for smooth state transfer (neutral-atom extension).

    Minimum-jerk ramp from 0 up to `amp` over `ramp_time`, then holds at
    `amp` for the remainder of `twidth`. Reference shape from Pagano,
    Motzoi et al., arXiv:2412.15173 (Dec 2024), which compares pulse
    shapes for neutral-atom transport and finds minimum-jerk significantly
    outperforms piecewise-linear and other forms.

        s = t / ramp_time
        shape(s) = 10·s³ - 15·s⁴ + 6·s⁵           for s in [0, 1]
        I(t) = amp · shape(t / ramp_time)          for t < ramp_time
        I(t) = amp                                  for t >= ramp_time

    The shape is C²-continuous *within* the ramp region. There is a
    C²-discontinuity at t = ramp_time (the junction with the hold region):
    the ramp's second derivative goes to zero approaching the junction
    from the left, then the hold is flat — but the third derivative is
    discontinuous. To represent that junction cleanly, the corresponding
    `feature_knots()` entry forces a knot at t = ramp_time so the spline
    can have different cubic coefficients on either side.

    Channel 2 (Q_func) returns zeros — unused for this envelope.

    paradict keys: ramp_time (seconds; must be ≤ twidth)
    """
    ramp_time = float(paradict["ramp_time"])
    if ramp_time <= 0:
        raise ValueError(f"adiabatic_ramp: ramp_time must be > 0, got {ramp_time}")
    if ramp_time > twidth + 1e-15:
        raise ValueError(
            f"adiabatic_ramp: ramp_time ({ramp_time}) exceeds twidth ({twidth})"
        )

    def I_func(t):
        t = np.asarray(t, dtype=float)
        s = np.clip(t / ramp_time, 0.0, 1.0)
        # Minimum-jerk polynomial: 10 s³ − 15 s⁴ + 6 s⁵
        shape = s ** 3 * (10.0 + s * (-15.0 + 6.0 * s))
        return amp * np.where(t < ramp_time, shape, 1.0)

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


def _rydberg_drag(twidth: float, amp: float, paradict: dict):
    """
    Rydberg DRAG pulse (Levine/Pichler form, not transmon DRAG).
    Gaussian amplitude + linearly chirped detuning.

    I_func(t) = amp * exp(-(t - t_c)² / (2 σ²))           (Gaussian envelope)
    Q_func(t) = detuning_max * (2t/twidth - 1)             (linear chirp, normalized
                                                            so Δ(0) = -detuning_max,
                                                            Δ(twidth) = +detuning_max)

    **Channel 2 semantic shift:** Q_func here is DETUNING, not quadrature
    amplitude. Downstream gateware that interprets channel 2 must know which
    env_func produced the BRAM image (the manifest carries env_func, so the
    upload path can dispatch correctly).

    paradict keys:
        omega_max     : peak Rabi frequency in rad/s (INFORMATIONAL — used
                        upstream to set `amp` in DAC fraction; the envelope
                        peak in DAC units is just `amp`).
        sigma         : Gaussian width in seconds (σ in the formula above)
        detuning_max  : peak detuning amplitude in DAC fraction (channel 2)
        chirp_shape   : 'linear' (default). Other shapes can be added later
                        — raise NotImplementedError for unknowns so callers
                        get a clear error rather than a silent fallback.
    """
    sigma = float(paradict["sigma"])
    detuning_max = float(paradict["detuning_max"])
    chirp_shape = str(paradict.get("chirp_shape", "linear"))

    if "omega_max" in paradict:
        # Informational only — present for upstream physical-units bookkeeping.
        _ = float(paradict["omega_max"])

    if chirp_shape != "linear":
        raise NotImplementedError(
            f"rydberg_drag: chirp_shape={chirp_shape!r} not supported "
            f"(only 'linear' for now)"
        )

    t_c = twidth / 2.0

    def I_func(t):
        t = np.asarray(t, dtype=float)
        return amp * np.exp(-0.5 * ((t - t_c) / sigma) ** 2)

    def Q_func(t):
        t = np.asarray(t, dtype=float)
        return detuning_max * (2.0 * t / twidth - 1.0)

    return I_func, Q_func


def _atom_transport(twidth: float, amp: float, paradict: dict):
    """
    Minimum-jerk position polynomial for neutral-atom transport
    (Lukin-group atom shuttling).

    Standard 5th-order minimum-jerk trajectory between two endpoints with
    zero velocity AND zero acceleration at both ends:

        τ = t / twidth
        x(τ) = x_start + (x_end - x_start) * (10 τ³ - 15 τ⁴ + 6 τ⁵)

    The pulse output is amp * x(τ). x_start / x_end are in DAC fraction
    (per QubiC convention), amp is the master scaling.

    Channel 2 (Q_func) returns zeros — unused for transport.

    paradict keys:
        x_start : initial position, DAC fraction (default 0.0)
        x_end   : final position, DAC fraction   (default 1.0)
    """
    x_start = float(paradict.get("x_start", 0.0))
    x_end   = float(paradict.get("x_end",   1.0))

    def I_func(t):
        t = np.asarray(t, dtype=float)
        tau = np.clip(t / twidth, 0.0, 1.0)
        # 10τ³ - 15τ⁴ + 6τ⁵ — Horner-form for slightly better numerics
        s = tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))
        return amp * (x_start + (x_end - x_start) * s)

    def Q_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))

    return I_func, Q_func


_DISPATCH = {
    "DRAG": _drag,
    "cos_edge_square": _cos_edge_square,
    "square": _square,
    "mark": _mark,
    "adiabatic_ramp": _adiabatic_ramp,
    "rydberg_drag": _rydberg_drag,
    "atom_transport": _atom_transport,
}


def make_envelope(env_func: str, twidth: float, amp: float, paradict: dict):
    """
    Dispatch to the correct envelope generator by name.

    Returns (I_func, Q_func): callables that take a time-array and return
    the I and Q envelope values. Designed to plug directly into autoknots():

        I_func, Q_func = make_envelope('DRAG', 32e-9, 0.128, paradict)
        res_I = autoknots(I_func, 0, 32e-9, delta=1e-3, min_dt=4e-9)
    """
    if env_func not in _DISPATCH:
        raise ValueError(
            f"Unknown env_func {env_func!r}. "
            f"Supported: {sorted(_DISPATCH.keys())}"
        )
    return _DISPATCH[env_func](twidth, amp, paradict)


# ─────────────────────────────────────────────────────────────────────────────
# Feature knots: interior time points where the spline must place a knot
# (e.g. C²-discontinuity junctions, hard turn-on/off edges). The autoknots
# midpoint+integral checks can otherwise miss these, leaving large residual
# error inside a wide segment that straddles the feature.
#
# Returned as a list[float] of interior times (NOT including t_start or
# t_end — those are added automatically). Pass to autoknots(seed_knots=...).
# ─────────────────────────────────────────────────────────────────────────────
def _feat_drag(twidth, paradict):              return []
def _feat_cos_edge_square(twidth, paradict):
    # The two ramp/flat junctions are C¹ but the spline already handles
    # them fine in practice (3 LSB on readout). Leave empty until proven
    # otherwise.
    return []
def _feat_square(twidth, paradict):            return []
def _feat_mark(twidth, paradict):              return []
def _feat_adiabatic_ramp(twidth, paradict):
    return [float(paradict["ramp_time"])]
def _feat_rydberg_drag(twidth, paradict):      return []
def _feat_atom_transport(twidth, paradict):    return []


_FEATURE_DISPATCH = {
    "DRAG": _feat_drag,
    "cos_edge_square": _feat_cos_edge_square,
    "square": _feat_square,
    "mark": _feat_mark,
    "adiabatic_ramp": _feat_adiabatic_ramp,
    "rydberg_drag": _feat_rydberg_drag,
    "atom_transport": _feat_atom_transport,
}


def feature_knots(env_func: str, twidth: float, paradict: dict) -> list:
    """
    Return interior knot times the spline fitter MUST place for this envelope.

    Used by compile_pulse() to seed autoknots with junction points. Empty
    list means no special seeding required. Returned times must be in the
    open interval (0, twidth) and separated by at least min_dt; both are
    validated downstream in autoknots().
    """
    if env_func not in _FEATURE_DISPATCH:
        raise ValueError(
            f"feature_knots: unknown env_func {env_func!r}. "
            f"Supported: {sorted(_FEATURE_DISPATCH.keys())}"
        )
    return list(_FEATURE_DISPATCH[env_func](twidth, paradict))
