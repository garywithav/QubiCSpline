"""
autoknots.py
============
AutoKnots adaptive cubic spline knot allocation (Vitenti et al. 2024).
Extracted from autoknots_quantum.ipynb as an importable module.

Public API:
    autoknots(f_func, t_start, t_end, delta=1e-3, eps=0.0, min_dt=0.0, ...)
    compress_multichannel(f_funcs, t_start, t_end, ...)
    AutoKnotsResult (dataclass)

See the notebook for mathematical derivation. Key additions vs. paper:
    - min_dt: minimum knot spacing floor (for finite time-axis DAC resolution)
    - refine: plateau/hidden-feature detection (paper Section 2.5)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────────────────────
# Exact per-segment integral of a CubicSpline in pp-form (no quadrature error)
# ─────────────────────────────────────────────────────────────────────────────
def spline_integrals(cs: CubicSpline) -> np.ndarray:
    """Exact analytic integral of each segment of a CubicSpline."""
    h = np.diff(cs.x)
    d, c, b, a = cs.c[0], cs.c[1], cs.c[2], cs.c[3]
    return h * (a + h * (b / 2.0 + h * (c / 3.0 + h * d / 4.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Paper's convergence conditions (2.8a pointwise + 2.8b integral) — vectorized
# ─────────────────────────────────────────────────────────────────────────────
def check_conditions(
    cs: CubicSpline,
    f_at_knots: np.ndarray,
    f_at_mids: np.ndarray,
    delta: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pass_f, pass_I, passed) per interval."""
    h = np.diff(cs.x)
    x_mids = 0.5 * (cs.x[:-1] + cs.x[1:])

    fhat_mids = cs(x_mids)
    err_f = np.abs(f_at_mids - fhat_mids)
    pass_f = err_f < delta * (np.abs(f_at_mids) + eps)

    I_simp = (h / 6.0) * (f_at_knots[:-1] + 4.0 * f_at_mids + f_at_knots[1:])
    I_spl = spline_integrals(cs)
    err_I = np.abs(I_simp - I_spl)
    pass_I = err_I < delta * (np.abs(I_simp) + eps * h)

    return pass_f, pass_I, pass_f & pass_I


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AutoKnotsResult:
    spline: CubicSpline
    knots: np.ndarray
    f_at_knots: np.ndarray
    converged: bool
    n_iterations: int
    n_refine_passes: int
    history: list = field(default_factory=list)

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_segments(self) -> int:
        return len(self.knots) - 1

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        return self.spline(np.asarray(t, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
def autoknots(
    f_func: Callable,
    t_start: float,
    t_end: float,
    delta: float = 1e-3,
    eps: float = 0.0,
    n_init: int = 6,
    max_knots: int = 4096,
    min_dt: float = 0.0,
    refine: bool = True,
    refine_ns: float = 5.0,
    refine_max_passes: int = 3,
    verbose: bool = False,
) -> AutoKnotsResult:
    """
    Adaptive cubic spline knot allocation.

    Parameters
    ----------
    f_func    : callable, f(t: ndarray) -> ndarray
    t_start, t_end : interval endpoints
    delta     : relative tolerance δ (primary parameter; typ. 1e-3)
    eps       : ε scale (set to ~1% of peak for signals near zero)
    n_init    : uniformly-spaced initial knots (≥ 6 for not-a-knot)
    max_knots : safety cap
    min_dt    : minimum allowed knot spacing in seconds. Intervals narrower
                than 2*min_dt won't be subdivided; if any still fail the
                error check they are "frozen" and a warning is emitted.
                For finite time-axis DAC: min_dt = 1 / (FS_time_DAC).
    refine, refine_ns, refine_max_passes : plateau-refinement (Section 2.5)
    verbose   : print iteration table

    Returns AutoKnotsResult.
    """
    assert n_init >= 6, "not-a-knot requires ≥ 6 initial knots"
    assert delta > 0
    assert eps >= 0
    assert min_dt >= 0

    history: list = []
    n_refine_passes = 0
    converged = False

    def _loop(t_k: np.ndarray, f_k: np.ndarray, tag: str = "") -> tuple:
        nonlocal converged

        frozen_reported = False

        for iteration in range(max_knots):
            cs = CubicSpline(t_k, f_k, bc_type="not-a-knot")

            x_mid = 0.5 * (t_k[:-1] + t_k[1:])
            f_mid = np.asarray(f_func(x_mid), dtype=float)

            pass_f, pass_I, passed = check_conditions(cs, f_k, f_mid, delta, eps)
            n_fail = int(np.sum(~passed))

            fhat_m = cs(x_mid)
            h = np.diff(t_k)
            maxF = float(np.max(np.abs(f_mid - fhat_m)
                                / (np.abs(f_mid) + eps + 1e-300))) / delta
            I_s = (h / 6.0) * (f_k[:-1] + 4.0 * f_mid + f_k[1:])
            I_spl = spline_integrals(cs)
            maxI = float(np.max(np.abs(I_s - I_spl)
                                / (np.abs(I_s) + eps * h + 1e-300))) / delta

            too_narrow = h < 2.0 * min_dt if min_dt > 0 else np.zeros_like(h, dtype=bool)
            subdividable = ~passed & ~too_narrow
            frozen = ~passed & too_narrow
            n_frozen = int(np.sum(frozen))
            n_sub = int(np.sum(subdividable))

            history.append((len(t_k), n_fail, maxF, maxI))

            if verbose:
                print(f"  iter={iteration:>3} knots={len(t_k):>4} "
                      f"fail={n_fail:>3} frozen={n_frozen:>3} "
                      f"maxF/δ={maxF:.2e} maxI/δ={maxI:.2e}")

            if n_fail == 0:
                converged = True
                return t_k, f_k, cs

            if n_sub == 0:
                if not frozen_reported:
                    warnings.warn(
                        f"AutoKnots: {n_frozen} interval(s) hit min_dt={min_dt:.3e} "
                        f"floor. Residual error may exceed δ. "
                        f"Relax min_dt or increase delta.",
                        RuntimeWarning,
                    )
                    frozen_reported = True
                converged = True
                return t_k, f_k, cs

            if len(t_k) + n_sub > max_knots:
                warnings.warn(
                    f"AutoKnots: would exceed max_knots={max_knots} "
                    f"(need +{n_sub}, have {len(t_k)}).",
                    RuntimeWarning,
                )
                return t_k, f_k, cs

            new_t = x_mid[subdividable]
            new_f = f_mid[subdividable]
            t_all = np.concatenate([t_k, new_t])
            f_all = np.concatenate([f_k, new_f])
            idx = np.argsort(t_all, kind="stable")
            t_k = t_all[idx]
            f_k = f_all[idx]

        cs = CubicSpline(t_k, f_k, bc_type="not-a-knot")
        return t_k, f_k, cs

    span = t_end - t_start
    if min_dt > 0:
        # If the user's n_init leaves segments too narrow to ever bisect
        # (spacing < 2*min_dt means children would fall below min_dt), bump
        # n_init up to the densest grid min_dt permits. This rescues the
        # "short pulse" case where the min_dt floor dominates — the algorithm
        # can't add detail by subdividing, so start at max density.
        max_n_allowed = int(np.floor(span / min_dt)) + 1
        spacing = span / (n_init - 1)
        if spacing < 2.0 * min_dt:
            n_init = max(n_init, max_n_allowed)
        elif spacing < min_dt:
            # Pathological: user n_init already violates min_dt. Clamp.
            n_init = max(6, max_n_allowed)

    t_k = np.linspace(t_start, t_end, n_init)
    f_k = np.asarray(f_func(t_k), dtype=float)

    t_k, f_k, cs = _loop(t_k, f_k, tag="initial")

    if refine and converged:
        for rp in range(refine_max_passes):
            h = np.diff(t_k)
            th = np.mean(h) + refine_ns * np.std(h)
            wide = (h > th) & (
                (h >= 2.0 * min_dt) if min_dt > 0 else np.ones_like(h, dtype=bool)
            )
            if int(np.sum(wide)) == 0:
                break

            mid_t = 0.5 * (t_k[:-1][wide] + t_k[1:][wide])
            mid_f = np.asarray(f_func(mid_t), dtype=float)
            t_all = np.concatenate([t_k, mid_t])
            f_all = np.concatenate([f_k, mid_f])
            idx = np.argsort(t_all, kind="stable")
            t_k = t_all[idx]
            f_k = f_all[idx]

            converged = False
            t_k, f_k, cs = _loop(t_k, f_k, tag=f"refine {rp+1}")
            n_refine_passes += 1
            if not converged:
                break

    return AutoKnotsResult(
        spline=cs,
        knots=t_k,
        f_at_knots=f_k,
        converged=converged,
        n_iterations=len(history),
        n_refine_passes=n_refine_passes,
        history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
def compress_multichannel(
    f_funcs: dict,
    t_start: float,
    t_end: float,
    delta: float = 1e-3,
    eps: float = 0.0,
    max_knots: int = 4096,
    min_dt: float = 0.0,
    refine: bool = True,
    refine_ns: float = 5.0,
    verbose: bool = False,
) -> dict:
    """
    Compress multiple channels to a shared knot grid.

    Runs autoknots on each channel independently, unions the knot sets,
    merges any pair closer than min_dt, and refits every channel on the
    unified grid. Endpoints are always preserved.

    Returns:
        {
            'knots'  : unified knot times,
            'splines': {name: CubicSpline on unified grid},
            'results': {name: AutoKnotsResult for per-channel fit},
            'ppform' : bookkeeping dict,
        }
    """
    all_knots = [np.linspace(t_start, t_end, 6)]
    per_ch: dict = {}

    for name, f in f_funcs.items():
        res = autoknots(
            f, t_start, t_end,
            delta=delta, eps=eps, max_knots=max_knots, min_dt=min_dt,
            refine=refine, refine_ns=refine_ns, verbose=verbose,
        )
        per_ch[name] = res
        all_knots.append(res.knots)

    unified = np.unique(np.concatenate(all_knots))

    if min_dt > 0 and len(unified) > 2:
        # FP tolerance: protects against spacings that are conceptually equal
        # to min_dt but land just under due to IEEE 754 rounding
        # (e.g. 16e-9 - 12e-9 evaluates to 3.999...e-9).
        tol = 1e-9 * min_dt
        kept = [unified[0]]
        for x in unified[1:-1]:
            if x - kept[-1] >= min_dt - tol:
                kept.append(x)
        if unified[-1] - kept[-1] < min_dt - tol and len(kept) > 1:
            kept.pop()
        kept.append(unified[-1])
        unified = np.asarray(kept)

    unified_splines = {}
    for name, f in f_funcs.items():
        f_u = np.asarray(f(unified), dtype=float)
        unified_splines[name] = CubicSpline(unified, f_u, bc_type="not-a-knot")

    n_segs = len(unified) - 1
    return {
        "knots": unified,
        "splines": unified_splines,
        "results": per_ch,
        "ppform": {
            "n_knots": len(unified),
            "n_segments": n_segs,
            "n_channels": len(f_funcs),
            "ddr_bytes_per_channel": n_segs * 32,
            "total_ddr_bytes": n_segs * 32 * len(f_funcs),
        },
    }
