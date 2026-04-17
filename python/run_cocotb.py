"""
run_cocotb.py
=============
Python runner for the spline_eval cocotb testbench. Use this instead of a
Makefile — it calls iverilog directly, so you don't need GNU make on Windows.

Prereqs:
    pip install cocotb
    iverilog on PATH (you have it at C:\\iverilog\\bin\\iverilog)

Run:
    python run_cocotb.py
"""

from pathlib import Path
from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent          # python/
ROOT = HERE.parent                    # project root
RTL  = ROOT / "rtl"


def main():
    runner = get_runner("icarus")

    runner.build(
        verilog_sources=[
            RTL / "spline_eval.sv",
            RTL / "spline_eval_top.sv",
        ],
        hdl_toplevel="spline_eval_top",
        build_args=["-g2012"],
        always=True,
    )

    runner.test(
        hdl_toplevel="spline_eval_top",
        test_module="test_spline_eval",
        test_dir=HERE,                # cocotb needs to find test_spline_eval.py
    )


if __name__ == "__main__":
    main()
