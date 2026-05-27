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
    max_dt: Optional[float] = None,
    seed_knots: Optional[list] = None,
    grid_quantum: Optional[float] = None,
    refine: bool = True,
    refine_ns: float = 5.0,
    refine_max_passes: int = 3,
    verbose: bool = False,
) -> AutoKnotsResult:
    """
    Adaptive cubic spline knot allocation.

    Hardware-aware constraints
    --------------------------
    Three optional parameters let the caller surface hardware limits into
    the algorithm so the produced spline is guaranteed-shippable:

        min_dt       — minimum segment width (physical: time-DAC resolution
                       limit). Intervals narrower than 2*min_dt are frozen.
        max_dt       — maximum segment width (artifact: 16-bit h_cycles
                       field in the current width-BRAM layout, NOT a
                       physical limit — could be lifted by widening the
                       RTL h_cycles register). Intervals wider than max_dt
                       are force-subdivided regardless of error.
        grid_quantum — t-axis quantum (physical: 1/FS, the FPGA's clock
                       period). Every proposed knot is snapped to this
                       grid before local error checks, so the float spline
                       lives on the same grid the hardware executes.

    All three default to None/0 (no constraint), preserving pre-extension
    behaviour exactly. Each is checked at every place new knots enter t_k
    (initial uniform grid, bisection midpoint, refine midpoint, seed).
    Adding a new HW-derived constraint follows the same pattern: define
    semantics, validate at API entry, apply at every knot-introduction
    site, and document here.

    Parameters
    ----------
    f_func    : callable, f(t: ndarray) -> ndarray
    t_start, t_end : interval endpoints
    delta     : relative tolerance δ (primary parameter; typ. 1e-3)
    eps       : ε scale (set to ~1% of peak for signals near zero)
    n_init    : uniformly-spaced initial knots (≥ 6 for not-a-knot)
    max_knots : safety cap on knot count (BRAM depth)
    min_dt    : minimum allowed knot spacing in seconds. Intervals narrower
                than 2*min_dt won't be subdivided; if any still fail the
                error check they are "frozen" and a warning is emitted.
                For finite time-axis DAC: min_dt = 1 / (FS_time_DAC).
    max_dt    : maximum allowed segment width in seconds. Intervals wider
                than max_dt are subdivided unconditionally (independent of
                the error condition). Use this to honor a packing limit
                such as the 16-bit h_cycles field in spline_coeff_pack
                (max_dt = MAX_H_CYCLES / fs). Default None disables.
    seed_knots: optional list of interior times (in (t_start, t_end)) that
                MUST appear in the final knot grid. Use this to pre-place
                knots at known feature points such as junctions between
                C²-discontinuous regions (e.g. the ramp/hold boundary of
                adiabatic_ramp). Endpoints are added automatically and need
                not be included. Seeds within min_dt of each other or of
                an endpoint are an error.
    grid_quantum: optional time-axis resolution in seconds. When set, every
                new knot the algorithm proposes (initial uniform grid,
                bisection midpoints, refine-pass midpoints, seed_knots)
                is snapped to the nearest multiple of grid_quantum before
                the local error check. Guarantees the final knot grid
                is consistent with a sample-rate-quantized hardware time
                counter. Endpoints must be on-grid.
    refine, refine_ns, refine_max_passes : plateau-refinement (Section 2.5)
    verbose   : print iteration table

    Returns AutoKnotsResult.
    """
    assert n_init >= 6, "not-a-knot requires ≥ 6 initial knots"
    assert delta > 0
    assert eps >= 0
    assert min_dt >= 0
    if max_dt is not None:
        assert max_dt > 0, "max_dt must be positive"
        if min_dt > 0:
            assert max_dt > 2.0 * min_dt, (
                f"max_dt ({max_dt:.3e}) must exceed 2*min_dt "
                f"({2.0*min_dt:.3e}) — otherwise the algorithm can't "
                f"satisfy both bounds simultaneously."
            )

    # Grid validation. The grid must be at least min_dt-coarse-enough that
    # snapped bisection midpoints can satisfy min_dt; concretely, min_dt
    # should be an integer multiple of grid_quantum (or grid_quantum=0).
    if grid_quantum is not None and grid_quantum > 0:
        if min_dt > 0:
            ratio = min_dt / grid_quantum
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"grid_quantum={grid_quantum:.3e} must divide min_dt="
                    f"{min_dt:.3e} (ratio {ratio:.6f} is not integer)"
                )
        # Endpoints must already be on the grid.
        for ep_name, ep in [("t_start", t_start), ("t_end", t_end)]:
            off = ep / grid_quantum - round(ep / grid_quantum)
            if abs(off) > 1e-9:
                raise ValueError(
                    f"{ep_name}={ep} is not on grid_quantum={grid_quantum:.3e}"
                )

    def _snap(arr):
        """Snap to nearest grid_quantum multiple (no-op if grid_quantum is None)."""
        if grid_quantum is None or grid_quantum <= 0:
            return np.asarray(arr, dtype=float)
        return np.round(np.asarray(arr, dtype=float) / grid_quantum) * grid_quantum

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
            too_wide   = h > max_dt        if max_dt is not None else np.zeros_like(h, dtype=bool)

            # An interval needs subdivision if EITHER it failed the local
            # error test, OR it exceeds the packing-width cap. too_wide
            # is force-subdivided regardless of error condition.
            needs_work = ~passed | too_wide
            subdividable = needs_work & ~too_narrow
            frozen = needs_work & too_narrow
            n_frozen = int(np.sum(frozen))
            n_sub = int(np.sum(subdividable))
            n_wide = int(np.sum(too_wide))

            history.append((len(t_k), n_fail, maxF, maxI))

            if verbose:
                print(f"  iter={iteration:>3} knots={len(t_k):>4} "
                      f"fail={n_fail:>3} wide={n_wide:>3} frozen={n_frozen:>3} "
                      f"maxF/δ={maxF:.2e} maxI/δ={maxI:.2e}")

            # Converged when both: error checks pass AND no over-wide
            # segments left. The over-wide condition is independent of
            # δ but mandatory for packability.
            if n_fail == 0 and n_wide == 0:
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

            # Snap proposed midpoints to the time-DAC grid before they
            # join t_k. Snapping happens BEFORE error-checking-the-children
            # would, so the algorithm's convergence guarantee applies to
            # the actual grid-aligned spline (not a float spline that
            # would later silently shift in pack_coefficients).
            new_t = _snap(x_mid[subdividable])

            # A snapped midpoint can collide with an existing knot if
            # the grid is coarse vs the parent interval. Drop collisions.
            # (Also re-evaluate f at the snapped positions — the original
            # f_mid values were sampled at the unsnapped midpoints.)
            if grid_quantum is not None and grid_quantum > 0:
                # Distinct values only, also not already in t_k.
                new_t = np.unique(new_t)
                in_t_k = np.isin(
                    np.round(new_t / grid_quantum).astype(np.int64),
                    np.round(t_k / grid_quantum).astype(np.int64),
                )
                new_t = new_t[~in_t_k]
                if new_t.size == 0:
                    # All snapped midpoints collided with existing knots —
                    # bisection can't make progress under this grid. Treat
                    # as a frozen step and exit gracefully.
                    if not frozen_reported:
                        warnings.warn(
                            "AutoKnots: bisection midpoints collide with "
                            "existing knots after grid_quantum snapping; "
                            "no further refinement possible.",
                            RuntimeWarning,
                        )
                        frozen_reported = True
                    converged = True
                    return t_k, f_k, cs
                new_f = np.asarray(f_func(new_t), dtype=float)
            else:
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

    if max_dt is not None and max_dt > 0:
        # Ensure no initial segment exceeds max_dt. Otherwise the algorithm
        # would need a force-subdivide round just to satisfy the cap before
        # error refinement could begin. Cheaper to start dense enough.
        min_n_for_max = int(np.ceil(span / max_dt)) + 1
        if n_init < min_n_for_max:
            n_init = min_n_for_max

    t_k = np.linspace(t_start, t_end, n_init)
    if grid_quantum is not None and grid_quantum > 0:
        t_k = _snap(t_k)
        t_k[0] = t_start    # endpoints already on grid; keep exact
        t_k[-1] = t_end
        t_k = np.unique(t_k)
        # If snap+unique collapsed below 6 (rare for sane grids), expand
        # by inserting grid points until we have ≥ 6.
        while len(t_k) < 6:
            # Find widest gap and insert its (snapped) midpoint
            gaps = np.diff(t_k)
            i = int(np.argmax(gaps))
            mid = _snap(np.array([0.5 * (t_k[i] + t_k[i + 1])]))[0]
            if mid in t_k or mid == t_k[i] or mid == t_k[i + 1]:
                raise ValueError(
                    f"grid_quantum={grid_quantum:.3e} is too coarse to fit "
                    f"6 initial knots in [{t_start}, {t_end}]"
                )
            t_k = np.unique(np.concatenate([t_k, [mid]]))

    # ── Merge seed_knots into the initial grid ──────────────────────────────
    # Seeds force knots at known feature points (e.g. C²-discontinuity
    # junctions in ramp+hold pulses). We validate seeds, then drop any
    # uniform knot within min_dt of a seed so the seed wins.
    if seed_knots:
        seeds = np.sort(np.asarray(seed_knots, dtype=float))
        if grid_quantum is not None and grid_quantum > 0:
            seeds = np.unique(_snap(seeds))

        if np.any(seeds <= t_start + 1e-15) or np.any(seeds >= t_end - 1e-15):
            raise ValueError(
                f"seed_knots must be interior to (t_start={t_start}, "
                f"t_end={t_end}); got {seeds.tolist()}"
            )
        if len(seeds) >= 2 and np.min(np.diff(seeds)) < (min_dt - 1e-9 * max(min_dt, 1.0)):
            raise ValueError(
                f"seed_knots are spaced closer than min_dt={min_dt:.3e}: "
                f"{seeds.tolist()}"
            )
        if min_dt > 0:
            if (seeds[0] - t_start) < min_dt - 1e-9 * min_dt or \
               (t_end - seeds[-1]) < min_dt - 1e-9 * min_dt:
                raise ValueError(
                    f"seed_knots too close to endpoint vs min_dt={min_dt:.3e}: "
                    f"seeds={seeds.tolist()}, span=[{t_start},{t_end}]"
                )

        # Drop any uniform knot within min_dt of any seed (seeds win).
        if min_dt > 0:
            tol = min_dt - 1e-9 * min_dt
            keep = np.ones_like(t_k, dtype=bool)
            for s in seeds:
                keep &= np.abs(t_k - s) >= tol
            t_k = t_k[keep]
        t_k = np.unique(np.concatenate([t_k, seeds]))

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
            if grid_quantum is not None and grid_quantum > 0:
                mid_t = _snap(mid_t)
                # Drop any midpoint that collapsed onto an existing knot
                # under snapping (refine then has nothing to add for those).
                in_t_k = np.isin(
                    np.round(mid_t / grid_quantum).astype(np.int64),
                    np.round(t_k / grid_quantum).astype(np.int64),
                )
                mid_t = np.unique(mid_t[~in_t_k])
                if mid_t.size == 0:
                    break
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
    max_dt: Optional[float] = None,
    seed_knots: Optional[list] = None,
    grid_quantum: Optional[float] = None,
    refine: bool = True,
    refine_ns: float = 5.0,
    verbose: bool = False,
) -> dict:
    """
    Compress multiple channels to a shared knot grid.

    Runs autoknots on each channel independently, unions the knot sets,
    merges any pair closer than min_dt, and refits every channel on the
    unified grid. Endpoints are always preserved.

    `seed_knots` (if given) is passed through to every channel's autoknots
    call and protected during the unified-grid merge — these knots are
    never dropped by the min_dt merge.

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
            delta=delta, eps=eps, max_knots=max_knots,
            min_dt=min_dt, max_dt=max_dt,
            seed_knots=seed_knots, grid_quantum=grid_quantum,
            refine=refine, refine_ns=refine_ns, verbose=verbose,
        )
        per_ch[name] = res
        all_knots.append(res.knots)

    # Seed: also snap the linspace baseline if grid_quantum is on so the
    # union doesn't reintroduce off-grid knots from the bookkeeping seed.
    if grid_quantum is not None and grid_quantum > 0:
        all_knots[0] = np.unique(
            np.round(all_knots[0] / grid_quantum) * grid_quantum
        )

    unified = np.unique(np.concatenate(all_knots))

    protected = (
        set(round(float(s), 12) for s in seed_knots)
        if seed_knots else set()
    )

    if min_dt > 0 and len(unified) > 2:
        # FP tolerance: protects against spacings that are conceptually equal
        # to min_dt but land just under due to IEEE 754 rounding
        # (e.g. 16e-9 - 12e-9 evaluates to 3.999...e-9).
        tol = 1e-9 * min_dt
        kept = [unified[0]]
        for x in unified[1:-1]:
            x_key = round(float(x), 12)
            if x_key in protected:
                # Seeded knots are mandatory. If accepting this seed would
                # leave kept[-1] within min_dt, drop kept[-1] instead.
                while len(kept) > 1 and (x - kept[-1]) < min_dt - tol:
                    if round(float(kept[-1]), 12) in protected:
                        # Two seeds within min_dt is upstream user error.
                        raise ValueError(
                            f"seed_knots {kept[-1]} and {x} are within "
                            f"min_dt={min_dt:.3e}"
                        )
                    kept.pop()
                kept.append(x)
            elif x - kept[-1] >= min_dt - tol:
                kept.append(x)
        if unified[-1] - kept[-1] < min_dt - tol and len(kept) > 1:
            # Protect last endpoint; drop the prior kept knot if it's not
            # itself a seed.
            if round(float(kept[-1]), 12) not in protected:
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
