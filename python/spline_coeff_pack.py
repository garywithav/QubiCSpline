"""
spline_coeff_pack.py
====================
Converts AutoKnots CubicSpline output to the fixed-point format expected
by spline_eval.sv and writes BRAM initialization files.

COEFFICIENT SCALING
────────────────────
spline_eval.sv normalises u to [0,1) within each segment:
    u_norm = u_cnt / h_i   (u_cnt goes 0 .. h_i-1)

Coefficients are therefore scaled by h_s^k for degree k:
    a_hw = a              (dimension: signal units)
    b_hw = b * h_s        (b in 1/s, h_s in s → dimensionless × signal)
    c_hw = c * h_s^2
    d_hw = d * h_s^3

All values end up with the same magnitude as the signal itself (~0.1–1.0),
fitting cleanly in Q1.15 (range ±2.0).

FIXED-POINT FORMAT: Q1.15 signed 16-bit
    1.0  → 32767 (0x7FFF)
   -1.0  → -32768 (0x8000)
   precision = 2^-15 = 3.05e-5 = 1 DAC LSB (16-bit)

WIDTH BRAM LAYOUT (32 bits per row)
    [31:16] = recip_h_q115  = round(32768 / h_cycles)  (Q1.15 reciprocal)
    [15: 0] = h_cycles      = segment width in clock cycles (integer)

COEFFICIENT BRAM LAYOUT (128 bits per row)
    [127:112] d_I  [111:96] c_I  [95:80] b_I  [79:64] a_I
    [ 63: 48] d_Q  [ 47:32] c_Q  [31:16] b_Q  [15: 0] a_Q

FILE FORMATS
────────────
.mem files — plain text, one hex value per line.
  Used by Verilog simulators: $readmemh("coeff_init.mem", coeff_bram)
  Each line is just a hex number. Nothing special.

.coe files — Xilinx Vivado format for Block Memory Generator IP.
  Used to pre-load BRAM in the FPGA bitstream.
  When the FPGA powers on, the BRAM already has these values.
  Format: memory_initialization_radix=16;
          memory_initialization_vector=VALUE1,VALUE2,...;
  Configure BRAM IP: Width=128 (coeff) or 32 (width), Depth=4096, 
                     Simple Dual Port, Read Latency=2.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

FS        = 1e9       # QubiC ZCU216 DAC sample rate
Q_SCALE   = 32768     # 2^15 for Q1.15
Q_MAX     = 32767
Q_MIN     = -32768
BRAM_DEPTH = 4096

# Minimum allowed segment width in clock cycles. Must match the
# MIN_H_CYCLES parameter in spline_eval.sv. The hardware's prefetch FSM
# requires h_cycles >= 3 for correctness; we default to 4 for a one-cycle
# margin. Corresponding minimum time spacing is MIN_H_CYCLES / FS seconds.
MIN_H_CYCLES = 4

# Maximum allowed segment width in clock cycles. This is a packing
# constraint: the width BRAM stores h_cycles in the lower 16 bits of a
# 32-bit word, so the field can represent at most 2^16 - 1 = 65535
# cycles (= 65.5 µs at 1 GSPS). NOT a physical hardware-resolution
# limit — purely an artifact of the chosen BRAM layout. Could be lifted
# by widening h_cycles in spline_eval.sv + reformatting the width word;
# tracked as a future RTL change.
#
# autoknots(max_dt=...) enforces this on the algorithm side so the
# packer never produces a segment that wouldn't fit.
MAX_H_CYCLES = 65535


def to_q115(x: float, label: str = "") -> int:
    v = round(x * Q_SCALE)
    if v > Q_MAX:
        print(f"  OVERFLOW {label}: {x:.6f} clipped to {Q_MAX/Q_SCALE:.6f}")
        return Q_MAX
    if v < Q_MIN:
        print(f"  UNDERFLOW {label}: {x:.6f} clipped to {Q_MIN/Q_SCALE:.6f}")
        return Q_MIN
    return int(v)


def u16(v: int) -> int:
    return v & 0xFFFF


def pack_coefficients(
    cs_I: CubicSpline,
    cs_Q: CubicSpline,
    fs: float = FS,
    min_h_cycles: int = MIN_H_CYCLES,
    verbose: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Pack AutoKnots CubicSpline output into hardware BRAM words.

    cs_I and cs_Q must have IDENTICAL knot positions (unified grid).
    If they don't, compute:
        unified = np.unique(np.concatenate([result_I.knots, result_Q.knots]))
        cs_I = CubicSpline(unified, interp_I(unified), bc_type='not-a-knot')
        cs_Q = CubicSpline(unified, interp_Q(unified), bc_type='not-a-knot')

    Returns (coeff_words, width_words):
        coeff_words: list of 128-bit ints, one per segment
        width_words: list of 32-bit ints [recip_h<<16 | h_cycles]
    """
    assert np.allclose(cs_I.x, cs_Q.x), \
        "cs_I and cs_Q must have identical knot positions."

    knots  = cs_I.x
    n_segs = len(knots) - 1

    if n_segs > BRAM_DEPTH:
        raise ValueError(f"Too many segments ({n_segs} > {BRAM_DEPTH}). Relax delta.")

    # Enforce minimum segment width. The spline must have been fit with
    # min_dt >= min_h_cycles/fs in the AutoKnots call upstream.
    h_cycles_all = np.round(np.diff(knots) * fs).astype(int)
    bad = np.where(h_cycles_all < min_h_cycles)[0]
    if len(bad):
        worst = int(h_cycles_all[bad].min())
        raise ValueError(
            f"{len(bad)} segment(s) have h_cycles < MIN_H_CYCLES={min_h_cycles} "
            f"(worst: {worst} cycles at segment {int(bad[0])}). "
            f"Refit with autoknots(min_dt={min_h_cycles}/fs={min_h_cycles/fs:.3e}) "
            f"or lower MIN_H_CYCLES in spline_eval.sv."
        )

    # Enforce maximum segment width — the 16-bit h_cycles field in the
    # width BRAM word can't represent values >= 2^16. This is a packing
    # constraint, not a physical limit (see MAX_H_CYCLES comment). The
    # algorithm SHOULD have prevented this via max_dt in autoknots; we
    # check here as defense-in-depth so an upstream bug causes a clear
    # raise rather than silent BRAM wrap → corrupt HW output.
    wide = np.where(h_cycles_all > MAX_H_CYCLES)[0]
    if len(wide):
        worst = int(h_cycles_all[wide].max())
        raise ValueError(
            f"{len(wide)} segment(s) have h_cycles > MAX_H_CYCLES={MAX_H_CYCLES} "
            f"(worst: {worst} cycles at segment {int(wide[0])}). "
            f"This should have been prevented by autoknots(max_dt=...); "
            f"call compile_pulse so max_dt is set, or pass "
            f"max_dt={MAX_H_CYCLES}/fs={MAX_H_CYCLES/fs:.3e} when invoking "
            f"autoknots directly."
        )

    # Explicit Q1.15 overflow check on the d coefficient. The d term is
    # scaled by h_s³ in pack_coefficients (see below), so for long pulses
    # with wide segments d_hw = d * h_s³ can exceed the Q1.15 range ±1.0
    # even when the underlying signal is bounded. Catch this BEFORE
    # to_q115() silently clips — for transport-class shapes (atom_transport,
    # adiabatic_ramp at large twidth) this is a real failure mode, not a
    # spurious clip warning.
    h_s_all = np.diff(knots).astype(float)
    d_I_hw = cs_I.c[0] * h_s_all ** 3   # shape (n_segs,)
    d_Q_hw = cs_Q.c[0] * h_s_all ** 3
    Q115_RANGE = Q_MAX / Q_SCALE        # 0.999969... — actual representable max
    overflowing = []
    for i in range(n_segs):
        for ch_name, d_val in (("d_I", d_I_hw[i]), ("d_Q", d_Q_hw[i])):
            if abs(d_val) > Q115_RANGE:
                overflowing.append((i, ch_name, float(d_val), float(h_s_all[i])))
    if overflowing:
        msg = [
            "Q1.15 overflow on d coefficient(s) at pack time. The d term "
            "is scaled by h_s³ and saturates for wide-segment / steep-"
            "cubic combinations. Tighten delta (more knots → narrower h_s) "
            "or reduce amp.",
            "",
            f"  {'seg':>4} {'channel':>8} {'h_s (ns)':>10} {'d * h_s³':>12} "
            f"{'limit':>10}",
        ]
        for seg, ch_name, d_val, h_s in overflowing:
            msg.append(
                f"  {seg:>4} {ch_name:>8} {h_s*1e9:>10.1f} {d_val:>+12.4f} "
                f"{Q115_RANGE:>+10.4f}"
            )
        raise ValueError("\n".join(msg))

    if verbose:
        print(f"Packing {n_segs} segments | Q1.15 | fs={fs/1e9:.1f} GSPS")
        print(f"  {'Seg':>4} {'h_ns':>7} {'h_cyc':>6} {'recip':>8} | "
              f"{'a_I':>8} {'b_I':>8} {'c_I':>8} {'d_I':>8}")
        print("  " + "─" * 68)

    coeff_words, width_words = [], []

    for i in range(n_segs):
        h_s   = float(knots[i+1] - knots[i])        # segment width in seconds
        h_cyc = int(round(h_s * fs))                 # segment width in cycles
        recip = round(Q_SCALE / h_cyc)               # round(32768 / h_cycles)

        # scipy pp-form: c[0]=d (cubic), c[1]=c (quad), c[2]=b (lin), c[3]=a (const)
        # Scale by h_s^k so hardware can use u_norm = u_cnt/h_cyc in [0,1)
        a_I = float(cs_I.c[3, i])
        b_I = float(cs_I.c[2, i]) * h_s
        c_I = float(cs_I.c[1, i]) * h_s**2
        d_I = float(cs_I.c[0, i]) * h_s**3

        a_Q = float(cs_Q.c[3, i])
        b_Q = float(cs_Q.c[2, i]) * h_s
        c_Q = float(cs_Q.c[1, i]) * h_s**2
        d_Q = float(cs_Q.c[0, i]) * h_s**3

        if verbose:
            print(f"  {i:>4} {h_s*1e9:>7.1f} {h_cyc:>6} {recip:>8} | "
                  f"{a_I:>8.4f} {b_I:>8.4f} {c_I:>8.4f} {d_I:>8.4f}")

        # Convert to Q1.15
        aI = u16(to_q115(a_I, f"a_I[{i}]")); bI = u16(to_q115(b_I, f"b_I[{i}]"))
        cI = u16(to_q115(c_I, f"c_I[{i}]")); dI = u16(to_q115(d_I, f"d_I[{i}]"))
        aQ = u16(to_q115(a_Q, f"a_Q[{i}]")); bQ = u16(to_q115(b_Q, f"b_Q[{i}]"))
        cQ = u16(to_q115(c_Q, f"c_Q[{i}]")); dQ = u16(to_q115(d_Q, f"d_Q[{i}]"))

        # Pack 128-bit coefficient word
        cword = (dI<<112)|(cI<<96)|(bI<<80)|(aI<<64)|(dQ<<48)|(cQ<<32)|(bQ<<16)|aQ
        coeff_words.append(cword)

        # Pack 32-bit width word: [31:16]=recip, [15:0]=h_cycles
        width_words.append((u16(recip) << 16) | (h_cyc & 0xFFFF))

    if verbose:
        raw   = int(round((knots[-1]-knots[0])*fs)) * 4
        spline = n_segs * 16
        print(f"\n  Spline: {n_segs}×16B = {spline}B  |  Raw: {raw}B  "
              f"|  Compression: {raw/spline:.1f}×")

    return coeff_words, width_words


def verify(cs_I, cs_Q, coeff_words, width_words, n_eval=5000):
    """Simulate Q1.15 Horner in Python and compare to float spline."""
    t0, t1 = cs_I.x[0], cs_I.x[-1]
    t_e = np.linspace(t0, t1, n_eval)
    I_ref, Q_ref = cs_I(t_e), cs_Q(t_e)
    I_hw, Q_hw   = np.zeros(n_eval), np.zeros(n_eval)

    for i, (cw, ww) in enumerate(zip(coeff_words, width_words)):
        h_cyc = ww & 0xFFFF
        t_seg = cs_I.x[i]; t_end = cs_I.x[i+1]
        mask = (t_e >= t_seg) & (t_e < t_end)
        if i == len(coeff_words)-1: mask |= (t_e == t1)

        def gq(w, sh): v=(w>>sh)&0xFFFF; return (v-0x10000)/Q_SCALE if v>=0x8000 else v/Q_SCALE
        aI=gq(cw,64); bI=gq(cw,80); cI=gq(cw,96); dI=gq(cw,112)
        aQ=gq(cw,0);  bQ=gq(cw,16); cQ=gq(cw,32); dQ=gq(cw,48)

        u = (t_e[mask] - t_seg) / (h_cyc / FS)  # u_norm in [0,1)
        I_hw[mask] = aI + u*(bI + u*(cI + u*dI))
        Q_hw[mask] = aQ + u*(bQ + u*(cQ + u*dQ))

    dac_lsb = 2/2**16
    eI, eQ = np.max(np.abs(I_ref-I_hw)), np.max(np.abs(Q_ref-Q_hw))
    print(f"\nVerification (float sim of Q1.15 hardware):")
    print(f"  Max I error: {eI:.2e} = {eI/dac_lsb:.1f}× DAC LSB")
    print(f"  Max Q error: {eQ:.2e} = {eQ/dac_lsb:.1f}× DAC LSB")
    ok = eI/dac_lsb < 10 and eQ/dac_lsb < 10
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def write_mem(coeff_words, width_words,
              cf="coeff_init.mem", wf="width_init.mem"):
    """Write $readmemh-compatible .mem files for simulation."""
    with open(cf, 'w') as f:
        f.write("// Coefficient BRAM: [127:112]=d_I ... [15:0]=a_Q, all Q1.15\n")
        for i, w in enumerate(coeff_words):
            f.write(f"{w:032X}  // seg {i}\n")
        for i in range(len(coeff_words), BRAM_DEPTH):
            f.write(f"{'0':>032}\n")
    print(f"Written {cf}")

    with open(wf, 'w') as f:
        f.write("// Width BRAM: [31:16]=recip_h Q1.15, [15:0]=h_cycles\n")
        for i, w in enumerate(width_words):
            h = w & 0xFFFF; r = (w>>16)&0xFFFF
            f.write(f"{w:08X}  // seg {i}: h={h} cyc, recip={r}\n")
        for i in range(len(width_words), BRAM_DEPTH):
            f.write(f"00000000\n")
    print(f"Written {wf}")


def write_coe(coeff_words, width_words,
              cf="coeff.coe", wf="width.coe"):
    """Write Vivado Block Memory Generator .coe init files."""
    def _coe(fname, words, bits, depth):
        with open(fname, 'w') as f:
            f.write("memory_initialization_radix=16;\nmemory_initialization_vector=\n")
            all_w = [f"{w:0{bits//4}X}" for w in words]
            all_w += ["0"*(bits//4)] * (depth - len(words))
            f.write(",\n".join(all_w) + ";\n")
        print(f"Written {fname}  (Vivado BRAM init, "
              f"Width={bits}, Depth={depth}, ReadLatency=2)")
    _coe(cf, coeff_words, 128, BRAM_DEPTH)
    _coe(wf, width_words,  32, BRAM_DEPTH)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Self-test: Q0 readout cos_edge_square (twidth=2µs, amp=0.2408)")
    print("=" * 60)

    N=100_000; twidth=2e-6; rf=0.25; amp=0.2408
    t=np.linspace(0,twidth,N)
    ramp_t=twidth*rf
    env=np.zeros(N)
    r=(t<ramp_t); fl=(t>=ramp_t)&(t<=twidth-ramp_t); fa=t>twidth-ramp_t
    env[r]=0.5*(1-np.cos(np.pi*t[r]/ramp_t))
    env[fl]=1.0
    env[fa]=0.5*(1+np.cos(np.pi*(t[fa]-(twidth-ramp_t))/ramp_t))
    I_sig=amp*env; Q_sig=np.zeros(N)

    from scipy.interpolate import interp1d
    knots=np.linspace(0,twidth,41)
    fi=interp1d(t,I_sig,kind='linear'); fq=interp1d(t,Q_sig,kind='linear')
    cs_I=CubicSpline(knots,fi(knots),bc_type='not-a-knot')
    cs_Q=CubicSpline(knots,fq(knots),bc_type='not-a-knot')

    cw, ww = pack_coefficients(cs_I, cs_Q)
    verify(cs_I, cs_Q, cw, ww)
    write_mem(cw, ww)
    write_coe(cw, ww)
    print("\nDone.")
