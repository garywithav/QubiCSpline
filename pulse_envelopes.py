"""
pulse_envelopes.py
==================
QubiC envelope generators. One function per env_func name seen in qubitcfg.json.

Each generator returns (I_func, Q_func) — callables f(t: ndarray) -> ndarray —
so they can be fed directly into autoknots(). Q_func returns zeros for
envelopes without a quadrature component.

Public API:
    make_envelope(env_func, twidth, amp, paradict) -> (I_func, Q_func)

Supported env_func values: DRAG, cos_edge_square, square, mark
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


_DISPATCH = {
    "DRAG": _drag,
    "cos_edge_square": _cos_edge_square,
    "square": _square,
    "mark": _mark,
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
