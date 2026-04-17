"""
pack_qubitcfg.py
================
End-to-end driver: qubitcfg.json → BRAM images for spline_eval.sv.

Reads a QubiC config, compiles every gate whose envelope type we support,
allocates BRAM addresses, and writes out coefficient/width BRAM images
(.mem for simulation, .coe for Vivado bitstream init).

Also writes a manifest JSON mapping `gate_name` → `(base_addr, n_segs)` so the
QubiC control software knows where to point `coeff_base` / `n_segs` for each
gate fire.

Usage:
    python pack_qubitcfg.py qubitcfg.json
    python pack_qubitcfg.py qubitcfg.json --delta 5e-4 --only-compression-wins

Outputs (in current directory):
    spline_coeff.mem    128-bit hex, one line per segment
    spline_width.mem    32-bit hex, one line per segment
    spline_coeff.coe    Vivado BRAM init, Width=128 Depth=4096
    spline_width.coe    Vivado BRAM init, Width=32  Depth=4096
    spline_manifest.json {gate_name: {base, n_segs, env_func, compression, error_lsb}}

Flow:
    qubitcfg.json
         ↓
    for each gate: compile_pulse → (coeff_words, width_words)
         ↓
    concat into single BRAM image, record (base, n_segs) per gate
         ↓
    write .mem / .coe / manifest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from spline_pulse_compiler import compile_pulse, verify_pulse, CompiledPulse
from spline_coeff_pack import BRAM_DEPTH, MIN_H_CYCLES


SUPPORTED_ENV_FUNCS = {"DRAG", "cos_edge_square", "square", "mark"}


# ─────────────────────────────────────────────────────────────────────────────
def _compile_gate(gate_name: str, pulses: list, delta: float) -> Optional[dict]:
    """
    Compile one gate entry from qubitcfg.json. A gate may contain multiple
    pulse entries (e.g. on different channels); we compile each envelope
    separately and aggregate.

    Returns None if no supported envelopes, else:
        {
            'compiled': [CompiledPulse, ...],
            'labels':   ['<gate>.env0', '<gate>.env1', ...],
        }
    """
    compiled: list = []
    labels: list = []

    for pulse_idx, pulse in enumerate(pulses):
        envs = pulse.get("env", [])
        amp = pulse.get("amp", 1.0)
        twidth = pulse.get("twidth")
        if twidth is None or twidth <= 0:
            continue

        for env_idx, env in enumerate(envs):
            env_func = env.get("env_func")
            paradict = env.get("paradict", {})
            if env_func not in SUPPORTED_ENV_FUNCS:
                continue

            try:
                cp = compile_pulse(
                    env_func=env_func,
                    paradict=paradict,
                    twidth=float(twidth),
                    amp=float(amp),
                    delta=delta,
                )
            except Exception as exc:
                print(f"  ! {gate_name} pulse[{pulse_idx}].env[{env_idx}] "
                      f"({env_func}): compile failed — {exc}")
                continue

            compiled.append(cp)
            labels.append(f"{gate_name}.p{pulse_idx}.e{env_idx}.{env_func}")

    if not compiled:
        return None
    return {"compiled": compiled, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
def _write_mem(words: list, bits: int, path: Path, header: str) -> None:
    """Write $readmemh-compatible .mem file, padded to BRAM_DEPTH."""
    hex_width = bits // 4
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        for i, w in enumerate(words):
            f.write(f"{w:0{hex_width}X}  // {i}\n")
        for i in range(len(words), BRAM_DEPTH):
            f.write("0" * hex_width + "\n")


def _write_coe(words: list, bits: int, path: Path) -> None:
    """Write Vivado BMG .coe file, padded to BRAM_DEPTH."""
    hex_width = bits // 4
    lines = [f"{w:0{hex_width}X}" for w in words]
    lines += ["0" * hex_width] * (BRAM_DEPTH - len(words))
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(lines) + ";\n")


# ─────────────────────────────────────────────────────────────────────────────
def pack_qubitcfg(
    cfg_path: Path,
    out_dir: Path,
    delta: float = 1e-3,
    only_compression_wins: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Main pipeline. Returns the manifest dict.

    only_compression_wins: if True, skip gates whose spline representation
    is larger than the equivalent raw-sample representation (compression < 1).
    Useful for keeping the BRAM image focused on gates where splines actually
    save memory (typically the long readout pulses).
    """
    with open(cfg_path) as f:
        cfg = json.load(f)

    gates = cfg.get("Gates", {})
    if not gates:
        raise ValueError(f"No 'Gates' in {cfg_path}")

    coeff_image: list = []
    width_image: list = []
    manifest: dict = {}
    skipped: list = []

    if verbose:
        print(f"\nCompiling {len(gates)} gates from {cfg_path}")
        print(f"  delta={delta:.0e}, MIN_H_CYCLES={MIN_H_CYCLES}, "
              f"BRAM_DEPTH={BRAM_DEPTH}")
        print(f"  {'gate':<32} {'env':>16} {'segs':>5} {'comp':>6} "
              f"{'err_I':>7} {'err_Q':>7}")
        print("  " + "─" * 78)

    for gate_name, pulses in gates.items():
        result = _compile_gate(gate_name, pulses, delta)
        if result is None:
            continue

        for cp, label in zip(result["compiled"], result["labels"]):
            if only_compression_wins and cp.compression_ratio < 1.0:
                skipped.append((label, cp.compression_ratio))
                continue

            if len(coeff_image) + cp.n_segments > BRAM_DEPTH:
                print(f"  ! BRAM full ({len(coeff_image)}/{BRAM_DEPTH}), "
                      f"skipping {label}")
                skipped.append((label, cp.compression_ratio))
                continue

            base_addr = len(coeff_image)
            coeff_image.extend(cp.coeff_words)
            width_image.extend(cp.width_words)

            diag = verify_pulse(cp)

            manifest[label] = {
                "base": base_addr,
                "n_segs": cp.n_segments,
                "env_func": cp.env_func,
                "twidth": cp.twidth,
                "amp": cp.amp,
                "paradict": cp.paradict,
                "compression_ratio": cp.compression_ratio,
                "max_err_I_lsb": diag["max_err_I_lsb"],
                "max_err_Q_lsb": diag["max_err_Q_lsb"],
                "converged": cp.converged,
            }

            if verbose:
                print(f"  {label:<32} {cp.env_func:>16} {cp.n_segments:>5} "
                      f"{cp.compression_ratio:>5.1f}x "
                      f"{diag['max_err_I_lsb']:>6.1f}L "
                      f"{diag['max_err_Q_lsb']:>6.1f}L")

    # ── Write outputs ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    coeff_mem = out_dir / "spline_coeff.mem"
    width_mem = out_dir / "spline_width.mem"
    coeff_coe = out_dir / "spline_coeff.coe"
    width_coe = out_dir / "spline_width.coe"
    manifest_json = out_dir / "spline_manifest.json"

    _write_mem(coeff_image, 128, coeff_mem,
               "Spline coefficient BRAM — 128 bits: [d_I c_I b_I a_I d_Q c_Q b_Q a_Q]")
    _write_mem(width_image, 32, width_mem,
               "Spline width BRAM — 32 bits: [recip_h_q115 h_cycles]")
    _write_coe(coeff_image, 128, coeff_coe)
    _write_coe(width_image, 32, width_coe)

    with open(manifest_json, "w") as f:
        json.dump({
            "config_source": str(cfg_path),
            "delta": delta,
            "min_h_cycles": MIN_H_CYCLES,
            "bram_depth": BRAM_DEPTH,
            "total_segments_used": len(coeff_image),
            "gates": manifest,
            "skipped": [{"label": l, "compression": c} for l, c in skipped],
        }, f, indent=2)

    if verbose:
        print("  " + "─" * 78)
        print(f"  Wrote {len(coeff_image)} segments "
              f"({len(coeff_image)*16}B coeff / {len(coeff_image)*4}B width)")
        print(f"  BRAM utilisation: {100*len(coeff_image)/BRAM_DEPTH:.1f}%")
        if skipped:
            print(f"  Skipped {len(skipped)} entries "
                  f"({sum(1 for _,c in skipped if c < 1)} lost to compression)")
        print(f"\n  Outputs in {out_dir.resolve()}:")
        for p in [coeff_mem, width_mem, coeff_coe, width_coe, manifest_json]:
            print(f"    {p.name}")

    return {
        "manifest": manifest,
        "total_segments": len(coeff_image),
        "skipped": skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    HERE = Path(__file__).parent          # python/
    ROOT = HERE.parent
    DEFAULT_CFG = ROOT / "config" / "qubitcfg.json"
    DEFAULT_OUT = ROOT / "build"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cfg", type=Path, nargs="?", default=DEFAULT_CFG,
                    help=f"path to qubitcfg.json (default: {DEFAULT_CFG})")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--delta", type=float, default=1e-3,
                    help="relative tolerance (default: 1e-3)")
    ap.add_argument("--only-compression-wins", action="store_true",
                    help="skip gates where splines are larger than raw samples")
    args = ap.parse_args()

    pack_qubitcfg(
        cfg_path=args.cfg,
        out_dir=args.out,
        delta=args.delta,
        only_compression_wins=args.only_compression_wins,
    )
